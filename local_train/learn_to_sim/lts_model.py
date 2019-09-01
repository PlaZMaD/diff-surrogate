import sys
import torch
import numpy as np
sys.path.append('../')
from base_model import BaseConditionalGenerationOracle

class LearnToSimModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 n_samples,
                 n_samples_per_dim,
                 y_model: BaseConditionalGenerationOracle,
                 psi_dim: int,
                 y_dim: int,
                 x_dim: int,
                 policy_std=0.05):
        super().__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        # self._psi = init_psi.clone()
        self.baselines = torch.zeros([n_samples]).to(self.device)
        self.ewma_alpha = 0.05

        self._psi_dim = psi_dim
        self._x_dim = x_dim
        self._policy_std = policy_std
        self.n_samples = n_samples # K in the initial paper
        self.n_samples_per_dim = n_samples_per_dim # train_size in the paper

    def fit(self, x, current_psi):
        pass

    # num_repetitions is not used there, only to match interface
    def grad(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        self._psi = condition
        self._psi.requires_grad_(True)
        self.policy = torch.distributions.Normal(self._psi, torch.Tensor([np.sqrt(self._policy_std)]).to(self.device))

        rewards = torch.zeros([self.n_samples]).to(self.device)
        policy_samples = self.policy.sample([self.n_samples])
        # print("!", policy_samples.shape)
        for sample_index in range(self.n_samples):
            rewards[sample_index] = - self.func(policy_samples[sample_index], self.n_samples_per_dim)

        # print ('f shape', self.func(policy_samples[sample_index], self.n_samples_per_dim))
        advantage = rewards - self.baselines
        log_grads = self.policy.log_prob(policy_samples).sum(dim=1)
        # print('lg shape', log_grads.shape)
        reinforce_loss = torch.mean(log_grads * advantage)
        psi_grad = torch.autograd.grad(reinforce_loss, self._psi)[0]

        self.baselines = rewards * self.ewma_alpha + (1 - self.ewma_alpha) * self.baselines
        return psi_grad.detach()

    def generate(self, condition):
        return self._y_model.generate(condition)

    def log_density(self, y, condition):
        pass

    def loss(self, y, condition):
        pass