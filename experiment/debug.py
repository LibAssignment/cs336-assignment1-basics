# %%
import numpy as np
from pathlib import Path
name = "TinyStoriesV2-GPT4-train"
vocab_size = 10000
idx_filename = f"{name}.idx.npz"
idx = np.load(Path("out")/idx_filename)['arr_0']

# %%
from cs336_basics.training import RandomTokenDataLoader
from dataclasses import dataclass, asdict

@dataclass
class Config:
  vocab_size: int
  batch_size = 32
  context_length = 256
  d_model = 512
  d_ff = 1344 # d_model * 2.625
  num_heads = 16
  theta = 10000
  num_layers = 4
  tokens = 327680000

  @property
  def epochs(self):
    return self.tokens // (self.batch_size * self.context_length)

device = "cuda"

config = Config(vocab_size=vocab_size)

dataset = RandomTokenDataLoader(idx, batch_size=config.batch_size, context_length=config.context_length, device=device)

# %%
from cs336_basics.modules import LLM, _cross_entory
from cs336_basics.optimizer import AdamW
silu = LLM(
  vocab_size=config.vocab_size,
  num_layers=config.num_layers,
  context_length=config.context_length,
  d_model=config.d_model,
  d_ff=config.d_ff,
  num_heads=config.num_heads,
  theta=config.theta,
  device=device
)
optimizer = AdamW(silu.parameters())

# %%
list(silu.named_parameters())

# %%
t = dict(silu.named_parameters())['layers.3.ffn.gated_lu.weight']
t.isnan().count_nonzero()

# %%
import torch
from torch.nn import Parameter
params = dict(silu.named_parameters())

def _stats(v: torch.Tensor):
  return torch.stack([v.mean(), v.std(), v.min(), v.max()])
def stats(v: Parameter):
  if v.grad is None:
    return torch.stack([_stats(v.data)])
  return torch.stack([_stats(v.data), _stats(v.grad)])
def show_params_stats(params: dict[str, Parameter]):
  return {k: stats(v) for k, v in params.items()}

show_params_stats(params)

# %%
activations = {}
def get_activation(name):
  def hook(model, input, output: torch.Tensor):
    activations[name] = (tuple(i.detach() for i in input), output.detach())
  return hook

for name, module in silu.named_modules():
  module.register_forward_hook(get_activation(name))


# %%
with torch.autograd.detect_anomaly():
  x, y = dataset[0]
  y_hat = silu.forward(x)
  loss = _cross_entory(y_hat, y).mean()
  loss.backward()

# %%
list(activations.keys())

# %%
_stats(activations['layers.3.ffn.gated_lu.sig'][1])

# %%
_stats(activations['layers.3.ffn.gated_lu.sig'][0][0])

# %%
from cs336_basics.modules import SiLU
SiLU()(torch.tensor(-1.2785))

# %%
silu = SiLU()
x = activations['layers.3.ffn.gated_lu.sig'][0][0].clone().cpu()
x = torch.tensor(x, requires_grad=True)
y = silu.forward(x)
y.sum().backward()
assert x.grad is not None
x.grad

# %%
x[x.grad.isnan()]

# %%
