from abc import ABC, abstractmethod
import numpy as np
from base_model import BaseConditionalGenerationOracle
from numpy.linalg import LinAlgError
from line_search_tool import LineSearchTool, get_line_search_tool
from torch import optim
from logger import BaseLogger
from collections import defaultdict
import copy
import scipy
import matplotlib.pyplot as plt
import torch
import time
SUCCESS = 'success'
ITER_ESCEEDED = 'iterations_exceeded'
COMP_ERROR = 'computational_error'


class BaseOptimizer(ABC):
    """
    Base class for optimization of some function with logging
    functionality spread by all classes
    """
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 x_step: float = np.inf,  # step_data_gen
                 tolerance: torch.Tensor = torch.tensor(1e-4),
                 trace: bool = True,
                 num_repetitions: int = 1000,
                 max_iters: int = 1000,
                 *args, **kwargs):
        self._oracle = oracle
        self._oracle.eval()
        self._history = defaultdict(list)
        self._x = x
        self._x_init = copy.deepcopy(x)
        self._x_step = x_step
        self._tolerance = tolerance
        self._trace = trace
        self._max_iters = max_iters
        self._num_repetitions = num_repetitions
        self._num_iter = 0.
        self._alpha_k = 0.

    def _update_history(self, init_time):
        self._history['time'].append(
            time.time() - init_time
        )
        self._history['func'].append(
            self._oracle.func(self._x,
                              num_repetitions=self._num_repetitions).detach().cpu().numpy()
        )
        self._history['grad'].append(
            self._oracle.grad(self._x,
                              num_repetitions=self._num_repetitions).detach().cpu().numpy()
        )
        self._history['x'].append(
            self._x.detach().cpu().numpy()
        )
        self._history['alpha'].append(
            self._alpha_k
        )

    def optimize(self):
        """
        Run optimization procedure
        :return:
            torch.Tensor:
                x optim
            str:
                status_message
            defaultdict(list):
                optimization history
        """
        for i in range(self._max_iters):
            status = self._step()
            if status == COMP_ERROR:
                return self._x.detach().clone(), status, self._history
            elif status == SUCCESS:
                return self._x.detach().clone(), status, self._history
        return self._x.detach().clone(), ITER_ESCEEDED, self._history

    def update(self, oracle: BaseConditionalGenerationOracle, x: torch.Tensor):
        self._oracle = oracle
        self._x = x
        self._x_init = copy.deepcopy(x)
        self._history = defaultdict(list)

    @abstractmethod
    def _step(self):
        """
        Compute update of optimized parameter
        :return:
        """
        raise NotImplementedError('_step is not implemented.')

    def _post_step(self, init_time):
        """
        This function saves stats in history and forces
        :param init_time:
        :return:
        """
        self._x = torch.max(torch.min(self._x, self._x_init + self._x_step), self._x_init - self._x_step)
        self._num_iter += 1
        if self._trace:
            self._update_history(init_time=init_time)


class GradientDescentOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr
        self._alpha_k = None
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)

    def _step(self):
        # seems like a bad dependence...
        init_time = time.time()
        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        d_k = -self._oracle.grad(x_k, num_repetitions=self._num_repetitions)

        if self._alpha_k is None:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=None,
                                                               num_repetitions=self._num_repetitions)
        else:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=2 * self._alpha_k,
                                                               num_repetitions=self._num_repetitions)
        if self._alpha_k is None:
            print('alpha_k is None!')
            self._alpha_k = self._lr
        with torch.no_grad():
            x_k = x_k + d_k * self._alpha_k
        grad_norm = torch.norm(d_k).item()
        self._x = x_k

        super()._post_step(init_time)
        # seems not cool to call super method in the middle of function...

        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


class NewtonOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1.,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr  # in newton method learning rate used to initialize line search tool
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)
        self._alpha_k = None

    def _step(self):
        # seems like a bad dependence...
        init_time = time.time()
        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        d_k = -self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        h_d = self._oracle.hessian(x_k, num_repetitions=self._num_repetitions)
        try:
            c_and_lower = scipy.linalg.cho_factor(h_d.detach().cpu().numpy())
            d_k = scipy.linalg.cho_solve(c_and_lower, d_k.detach().cpu().numpy())
            d_k = torch.tensor(d_k).float().to(self._oracle.device)
        except LinAlgError:
            pass
        self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                           x_k,
                                                           d_k,
                                                           previous_alpha=self._lr,
                                                           num_repetitions=self._num_repetitions)
        if self._alpha_k is None:
            print('alpha_k is None!')
            self._alpha_k = self._lr

        with torch.no_grad():
            x_k = x_k + d_k * self._alpha_k
        self._x = x_k
        super()._post_step(init_time)

        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


def d_computation_in_lbfgs(d, history):
    l = len(history)
    mu = list()
    for i in range(l)[::-1]:
        s = history[i][0]
        y = history[i][1]
        mu.append(s.dot(d) / s.dot(y))
        d -= y * mu[-1]
    mu = mu[::-1]
    s = history[-1][0]
    y = history[-1][1]
    d = d * s.dot(y) / y.dot(y)
    for i in range(l):
        s = history[i][0]
        y = history[i][1]
        beta = y.dot(d) / s.dot(y)
        d += (mu[i] - beta) * s
    return d


class LBFGSOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 memory_size: int = 20,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr
        self._sy_history = list()
        self._alpha_k = None
        self._memory_size = memory_size
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)

    def _step(self):
        init_time = time.time()

        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        g_k = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)

        if len(self._sy_history) > 0:
            d_k = d_computation_in_lbfgs(-g_k.clone().detach(), self._sy_history)
        else:
            d_k = - g_k.clone().detach()

        if self._alpha_k is None:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=None,
                                                               num_repetitions=self._num_repetitions)
        else:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=2 * self._alpha_k,
                                                               num_repetitions=self._num_repetitions)

        if self._alpha_k is None:
            print('alpha_k is None!')
            self._alpha_k = self._lr
        x_k = x_k + d_k * self._alpha_k
        self._x = x_k.clone().detach()
        g_k_new = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        self._sy_history.append((self._alpha_k * d_k, g_k_new - g_k))
        if len(self._sy_history) > self._memory_size:
            self._sy_history.pop(0)

        super()._post_step(init_time)
        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


class ConjugateGradientsOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr
        self._alpha_k = None
        self._d_k = None
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)

    def _step(self):
        init_time = time.time()

        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        g_k = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        if self._d_k is None:
            self._d_k = -g_k.clone().detach()

        norm_squared = g_k.pow(2).sum()

        if self._alpha_k is None:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               self._d_k,
                                                               previous_alpha=None,
                                                               num_repetitions=self._num_repetitions)
        else:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               self._d_k,
                                                               previous_alpha=2 * self._alpha_k,
                                                               num_repetitions=self._num_repetitions)
        if self._alpha_k is None:
            print('alpha_k is None!')
            self._alpha_k = self._lr

        x_k = x_k + self._d_k * self._alpha_k
        g_k_next = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        beta_k = g_k_next.dot((g_k_next - g_k)) / norm_squared
        self._d_k = -g_k_next + beta_k * self._d_k
        self._x = x_k.clone().detach()

        super()._post_step(init_time)
        grad_norm = torch.norm(g_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(self._d_k).all()):
            return COMP_ERROR


class TorchOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 torch_model: str = 'Adam',
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._x.requires_grad_(True)
        self._lr = lr
        self._alpha_k = self._lr
        self._torch_model = torch_model
        self._base_optimizer = getattr(optim, self._torch_model)(
            params=[self._x], lr=lr
        )

    def _step(self):
        init_time = time.time()

        d_k = self._oracle.grad(self._x, num_repetitions=self._num_repetitions).detach()
        self._x.grad = d_k
        self._base_optimizer.step()
        self._base_optimizer.zero_grad()
        print(self._x)
        super()._post_step(init_time)
        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(self._d_k).all()):
            return COMP_ERROR


class GPOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1.,
                 base_estimator="gp",
                 acq_func='gp_hedge',
                 acq_optimizer="sampling",
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        from skopt import Optimizer
        self._x.requires_grad_(True)
        self._lr = lr
        self._alpha_k = self._lr
        self._opt_result = None
        borders = []
        x_step = self._x_step
        if x_step is None:
            x_step = self._lr
        for xi in x.detach().cpu().numpy():
            borders.append((xi - x_step, xi + x_step))
        self._base_optimizer = Optimizer(borders,
                                         base_estimator=base_estimator,
                                         acq_func=acq_func,
                                         acq_optimizer=acq_optimizer)


    def optimize(self):
        # d_k = self._oracle.grad(self._x, num_repetitions=self._num_repetitions).detach().cpu().numpy()
        f_k = self._oracle.func(self._x, num_repetitions=self._num_repetitions).item()
        self._base_optimizer.tell(
            self.bound_x(self._x.detach().cpu().numpy().tolist()),
            f_k
        )
        x, status, history = super().optimize()
        self._x = torch.tensor(self._opt_result.x).float().to(self._oracle.device)
        return self._x.detach().clone(), status, history

    def bound_x(self, x):
        x_new = []
        for xi, space in zip(x, self._base_optimizer.space):
            if xi in space:
                pass
            else:
                xi = np.clip(xi, space.low + 1e-3, space.high - 1e-3)
            x_new.append(xi)
        return x_new

    def _step(self):
        init_time = time.time()

        x_k = self._base_optimizer.ask()
        x_k = torch.tensor(x_k).float().to(self._oracle.device)
        d_k = self._oracle.grad(x_k, num_repetitions=self._num_repetitions) # .detach().cpu().numpy()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)

        self._opt_result = self._base_optimizer.tell(
            self.bound_x(x_k.detach().cpu().numpy().tolist()),
            f_k.item()
        )
        self._x = x_k.detach().clone()
        super()._post_step(init_time)
        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR



class HMCOptimizer(BaseLogger):
    pass