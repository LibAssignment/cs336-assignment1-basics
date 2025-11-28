from collections.abc import Iterable
import math
from typing import NotRequired, TypedDict, cast
from torch import Tensor
import torch
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class _GroupParams(TypedDict):
  params: Iterable[Parameter]

class _SGDParams(TypedDict):
  lr: float

class _SDGGroupParams(_SGDParams, _GroupParams):
  pass

class SGD(Optimizer):
  def __init__(self, params: ParamsT, lr: float = 0.01):
    super().__init__(params, dict(lr=lr))

  def step(self): # type: ignore
    for group in cast(list[_SDGGroupParams], self.param_groups):
      lr = group["lr"]
      for p in group["params"]:
        if p.grad is None:
          continue
        grad = p.grad.data
        p.data -= lr * grad


class _AdamWParams(TypedDict):
  alpha: float
  beta1: float
  beta2: float
  epsilon: float
  lamda: float

class _AdamWGroupParams(_AdamWParams, _GroupParams):
  pass

class _AdamState(TypedDict):
  m: NotRequired[Tensor]
  v: NotRequired[Tensor]
  step_count: NotRequired[Tensor]

class AdamW(Optimizer):
  def __init__(self, params: ParamsT, lr: float = 0.001, weight_decay: float = 0.1, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
    defaults = _AdamWParams(
      alpha = lr,
      beta1 = betas[0],
      beta2 = betas[1],
      epsilon = eps,
      lamda = weight_decay,
    )
    super().__init__(params, cast(dict, defaults))

  def step(self): # type: ignore
    for group in cast(list[_AdamWGroupParams], self.param_groups):
      beta1 = group["beta1"]
      beta2 = group["beta2"]
      alpha = group['alpha']
      epsilon = group['epsilon']
      decay = group['lamda']
      for param in group["params"]:
        if param.grad is None:
          continue
        state = cast(_AdamState, self.state[param])
        step_count = state.get('step_count', torch.tensor(0)) + 1
        alpha_t = alpha * (1 - beta2**step_count).sqrt() / (1 - beta1**step_count)
        grad = param.grad
        m = state.get('m', torch.zeros_like(param))
        v = state.get('v', torch.zeros_like(param))
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        state['m'] = m
        state['v'] = v
        state['step_count'] = step_count

        param.data -= alpha_t * m / (v.sqrt() + epsilon)
        param.data *= 1 - alpha * decay
