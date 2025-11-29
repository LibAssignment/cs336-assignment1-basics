from collections.abc import Callable, Iterable
import math
from typing import NotRequired, TypedDict, cast
from torch import Tensor
import torch
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT
from torch.optim.lr_scheduler import LRScheduler


class _GroupParams(TypedDict):
  params: Iterable[Parameter]

class _SGDParams(TypedDict):
  lr: float

class _SGDGroupParams(_SGDParams, _GroupParams):
  pass

class _SGDState(TypedDict):
  t: NotRequired[int]

class SGD(Optimizer):
  def __init__(self, params: ParamsT, lr: float = 0.01):
    super().__init__(params, dict(lr=lr))

  def step(self, closure: Callable[[], float] | None = None): # type: ignore
    loss = None if closure is None else closure()
    for group in cast(list[_SGDGroupParams], self.param_groups):
      lr = group["lr"]
      for p in group["params"]:
        if p.grad is None:
          continue
        state = cast(_SGDState, self.state[p])
        t = state.get('t', 0) + 1
        grad = p.grad.data
        p.data -= lr / math.sqrt(t) * grad
        state['t'] = t

    return loss


class _AdamWParams(TypedDict):
  lr: float
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
      lr = lr,
      beta1 = betas[0],
      beta2 = betas[1],
      epsilon = eps,
      lamda = weight_decay,
    )
    super().__init__(params, cast(dict, defaults))

  def step(self, closure: Callable[[], float] | None = None): # type: ignore
    loss = None if closure is None else closure()
    for group in cast(list[_AdamWGroupParams], self.param_groups):
      beta1 = group["beta1"]
      beta2 = group["beta2"]
      alpha = group['lr']
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
    return loss

class CosineLR(LRScheduler):
  def __init__(self,
      optimizer: Optimizer,
      last_epoch: int = -1,
      *,
      alpha_range: tuple[float, float],
      warmup_steps: int = 0,
      cosine_steps: int = 1000
    ):
    super().__init__(optimizer, last_epoch)
    self.a_min = alpha_range[0]
    self.a_max = alpha_range[1]
    self.warmup_steps = warmup_steps
    self.cosine_steps = cosine_steps

  def _lr(self):
    return _lr_cosine(self._step_count, self.a_min, self.a_max, self.warmup_steps, self.cosine_steps)

  def get_lr(self):
    lr = self._lr() / self.a_max
    return [lr * group["initial_lr"] for group in self.optimizer.param_groups]

def _lr_cosine(t: int, a_min: float, a_max: float, t_warmup: int, t_cosine: int):
  if t < t_warmup:
    return t / t_warmup * a_max
  if t < t_cosine:
    k = (t - t_warmup) / (t_cosine - t_warmup)
    return a_min + (1 + math.cos(k * math.pi)) * (a_max -a_min) / 2
  return a_min

def _gradient_clipping(params: Iterable[Parameter], max_l2_norm: float, eps=1e-6):
  grads = [p.grad for p in params if p.grad is not None]
  g = torch.stack([(p * p).sum() for p in grads])
  g = g.sum().sqrt()
  if g < max_l2_norm:
    return
  g = max_l2_norm / (g + eps)
  for p in params:
    if p.grad is None:
      continue
    p.grad.data *= g
