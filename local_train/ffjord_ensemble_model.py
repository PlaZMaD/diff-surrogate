import torch
from torch import nn
import torch.utils.data as dataset_utils
import copy
from base_model import BaseConditionalGenerationOracle
import sys
sys.path.append('./ffjord/')
import ffjord
import ffjord.lib
import ffjord.lib.utils as utils
from ffjord.lib.visualize_flow import visualize_transform
import ffjord.lib.layers.odefunc as odefunc
from ffjord.train_misc import standard_normal_logprob, create_regularization_fns
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
from ffjord.custom_model import build_model_tabular, get_transforms, compute_loss
import lib.layers as layers
from tqdm import tqdm, trange
from typing import Tuple
import swats
import warnings
import numpy as np
warnings.filterwarnings("ignore")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, improvement=1e-4):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self._improvement = improvement
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self._improvement:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class FFJORDModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 x_dim: int,
                 psi_dim: int,
                 y_dim: int,
                 num_blocks: int = 1,
                 lr: float = 1e-3,
                 epochs: int = 10,
                 bn_lag: float = 1e-3,
                 batch_norm: bool = True,
                 solver='fixed_adams',
                 k_models=3,
                 hidden_dims: Tuple[int] = (32, 32),

                 **kwargs):
        super(FFJORDModel, self).__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        self._x_dim = x_dim
        self._y_dim =y_dim
        self._psi_dim = psi_dim
        self._models = nn.ModuleList([
            build_model_tabular(dims=self._y_dim,
                                          condition_dim=self._psi_dim + self._x_dim,
                                          layer_type='concat_v2',
                                          num_blocks=num_blocks,
                                          rademacher=False,
                                          nonlinearity='tanh',
                                          solver=solver,
                                          hidden_dims=hidden_dims,
                                          bn_lag=bn_lag,
                                          batch_norm=batch_norm,
                                          regularization_fns=None)
                        for _ in range(k_models)
                        ])
        self._samples_fn = []
        self._densities_fn = []
        for i in range(k_models):
            _sample_fn, _density_fn = get_transforms(self._models[i])
            self._samples_fn.append(_sample_fn)
            self._densities_fn.append(_density_fn)
        self._epochs = epochs
        self._lr = lr

    def loss(self, y, condition):
        loss = 0.
        for model in self._models:
            loss = loss + compute_loss(model, data=y.detach(), condition=condition.detach())
        return loss

    def fit(self, y, condition):
        self.train()
        trainable_parameters = [
            list(_model.parameters()) for _model in self._models
        ]
        trainable_parameters = [item for sublist in trainable_parameters for item in sublist]
        optimizer = swats.SWATS(trainable_parameters, lr=self._lr, verbose=True)
        best_params = [_model.state_dict() for _model in self._models]
        best_loss = 1e6
        early_stopping = EarlyStopping(patience=200, verbose=True)
        dataset = dataset_utils.TensorDataset(condition, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=int(1024 * 128), shuffle=True)
        for epoch in range(self._epochs):
            loss_sum = 0.
            N = 0
            for condition_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = self.loss(y_batch, condition_batch)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                N += 1
            if loss_sum < best_loss:
                best_params = copy.deepcopy([_model.state_dict() for _model in self._models])
                best_loss = loss_sum
            early_stopping(loss_sum)
            if early_stopping.early_stop:
                break
            print(loss_sum / N)

        for i, _model in enumerate(self._models):
            _model.load_state_dict(best_params[i])
        self.eval()
        for i in range(len(self._models)):
            _sample_fn, _density_fn = get_transforms(self._models[i])
            self._samples_fn.append(_sample_fn)
            self._densities_fn.append(_density_fn)
        return self

    def generate(self, condition):
        n = len(condition)
        samples = []
        for _sample_fn in self._samples_fn:
            z = torch.randn(n, self._y_dim).to(self.device)
            samples.append(_sample_fn(z, condition))
        return torch.cat(samples, dim=1)

    def log_density(self, y, condition):
        return self._density_fn(y, condition)

    def train(self):
        super().train(True)
        for _model in self._models:
            for module in _model.modules():
                if hasattr(module, 'odeint'):
                    module.__setattr__('odeint', odeint_adjoint)

    def eval(self):
        super().train(False)
        # for module in self._model.modules():
        #   if hasattr(module, 'odeint'):
        #        module.__setattr__('odeint', odeint)