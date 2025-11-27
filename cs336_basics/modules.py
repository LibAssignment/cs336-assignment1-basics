from einops import einsum
import torch.nn
from torch.nn import Module, Parameter
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(Module):
  __constants__ = ["d_in", "d_out"]
  def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
    kwargs = {"device": device, "dtype": dtype}
    super().__init__()
    self.d_in = in_features
    self.d_out = out_features
    self.weight = Parameter(
      torch.empty(self.d_out, self.d_in, **kwargs)
    )
    torch.nn.init.trunc_normal_(self.weight)

  def forward(self, in_features: Float[Tensor, " ... d_in"]):
    return _linear(self.d_in, self.d_out, self.weight, in_features)


class Embedding(Module):
  __constants__ = ["n_embed", "d_embed"]
  def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
    kwargs = {"device": device, "dtype": dtype}
    super().__init__()
    self.n_embed = num_embeddings
    self.d_embed = embedding_dim
    self.weight = Parameter(
      torch.empty(self.n_embed, self.d_embed, **kwargs)
    )
    torch.nn.init.trunc_normal_(self.weight)

  def forward(self, token_ids: Int[Tensor, " ... "]):
    return self.weight[token_ids]


class RMSNorm(Module):
  __constants__ = ["d_model", "eps"]
  def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    kwargs = {"device": device, "dtype": dtype}
    super().__init__()
    self.d_model = d_model
    self.eps = eps
    self.weight = Parameter(
      torch.ones(d_model, **kwargs)
    )

  def forward(self, x: Float[Tensor, " ... "]):
    in_dtype = x.dtype
    x = x.to(torch.float32)
    rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
    result = x / rms * self.weight
    return result.to(in_dtype)


class Relu(Module):
  def forward(self, x: Tensor):
    return torch.max(x, torch.tensor(0))


class SiLU(Module):
  def forward(self, x: Tensor):
    return x / (1 + (-x).exp())


class GatedLU(Module):
  def __init__(self, d_input: int, d_output: int, sig: Module, device=None, dtype=None):
    kwargs = {"device": device, "dtype": dtype}
    super().__init__()
    self.d_input = d_input
    self.d_output = d_output
    self.sig = sig
    self.weight = Parameter(
      torch.empty(self.d_output, self.d_input, **kwargs)
    )
    self.v = Parameter(
      torch.empty(self.d_output, self.d_input, **kwargs)
    )
    torch.nn.init.trunc_normal_(self.weight)
    torch.nn.init.trunc_normal_(self.v)

  def forward(self, x: Float[Tensor, '... d_input']):
    x1 = torch.einsum("...i,ji->...j", x, self.weight)
    g = torch.einsum("...i,ji->...j", x, self.v)
    return self.sig(x1) * g


class FFN(Module):
  def __init__(self, d_model: int, d_hidden: int, sig: Module, device=None, dtype=None):
    super().__init__()
    self.gated_lu = GatedLU(d_model, d_hidden, sig, device=device, dtype=dtype)
    self.linear = Linear(d_hidden, d_model, device=device, dtype=dtype)

  def forward(self, x: Float[Tensor, "... d_input"]):
    return self.linear(self.gated_lu(x))


def _linear(
  d_in: int,
  d_out: int,
  weights: Float[Tensor, " d_out d_in"],
  in_features: Float[Tensor, " ... d_in"],
):
  assert weights.shape == (d_out, d_in)
  assert in_features.shape[-1] == d_in
  return in_features @ weights.T
