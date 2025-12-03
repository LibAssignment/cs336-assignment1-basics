from typing import Any, TypedDict
from einops import einsum, rearrange, repeat
import torch.nn
from torch.nn import Module, Parameter
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

class DeviceParams(TypedDict):
  device: Any
  dtype: Any

class Linear(Module):
  __constants__ = ["d_in", "d_out"]
  def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
    kwargs = DeviceParams(device=device, dtype=dtype)
    super().__init__()
    self.d_in = in_features
    self.d_out = out_features
    self.weight = Parameter(
      torch.empty(self.d_out, self.d_in, **kwargs)
    )
    torch.nn.init.trunc_normal_(self.weight)

  def forward(self, in_features: Float[Tensor, " ... d_in"]):
    # einsum(in_features, self.weight, "... d_in, d_out d_in -> ... d_out")
    return _linear(self.d_in, self.d_out, self.weight, in_features)


class Embedding(Module):
  __constants__ = ["n_embed", "d_embed"]
  def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
    kwargs = DeviceParams(device=device, dtype=dtype)
    super().__init__()
    self.n_embed = num_embeddings
    self.d_embed = embedding_dim
    self.weight = Parameter(
      torch.empty(self.n_embed, self.d_embed, **kwargs)
    )
    torch.nn.init.trunc_normal_(self.weight)

  def forward(self, token_ids: Int[Tensor, " ... vocab"]) -> Float[Tensor, " ... d_embed"]:
    return self.weight[token_ids]


class RMSNorm(Module):
  __constants__ = ["d_model", "eps"]
  def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    kwargs = DeviceParams(device=device, dtype=dtype)
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
  """
  $"SiLU"(x) = x dot sigma(x) = x / (1+e^(-x))$
  """
  def forward(self, x: Tensor):
    return x / (1 + (-x).exp())


class GatedLU(Module):
  """
  $"GLU"(x, W_1, V; sigma) = sigma(W_1 x) dot.o (V x)$
  """
  __constants__ = ["d_input", "d_output"]
  def __init__(self, d_input: int, d_output: int, sig: Module, device=None, dtype=None):
    kwargs = DeviceParams(device=device, dtype=dtype)
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


class RoPE(Module):
  __constants__ = ["theta", "d_k", "d_n"]
  def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
    kwargs = DeviceParams(device=device, dtype=dtype)
    super().__init__()
    self.Theta = theta
    self.d_k = d_k
    self.d_n = max_seq_len
    R = _rope_rotate(self.Theta, self.d_n, self.d_k)
    self.R = R.to_dense().to(device=device, dtype=dtype)

  def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"] | None = None):
    assert x.shape[-1] == self.d_k
    if token_positions is None:
      token_positions = torch.arange(x.shape[-2])
    assert x.shape[-2] == token_positions.shape[-1]
    m = self.R[token_positions]
    return einsum(x, m, "... k, ... k2 k -> ... k2")


class Softmax(Module):
  __constants__ = ["dim"]
  def __init__(self, dim = -1) -> None:
    super().__init__()
    self.dim = dim
  def forward(self, x: Float[Tensor, "... d"]):
    return _softmax(x, dim=self.dim)


class MultiHeadAttention(Module):
  __constants__ = ["d_embed", "d_k", "d_v", "n_heads", "auto_mask"]
  def __init__(self, d_model: int, d_k: int, d_v: int, num_heads: int = 1, pos_embed: Module | None = None, mask: bool = True, device=None, dtype=None):
    kwargs = DeviceParams(device=device, dtype=dtype)
    super().__init__()
    self.d_embed = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = num_heads
    self.auto_mask = mask
    self.pos_embed = pos_embed
    self.linear_q = Linear(d_model, d_k*num_heads, **kwargs)
    self.linear_k = Linear(d_model, d_k*num_heads, **kwargs)
    self.linear_v = Linear(d_model, d_v*num_heads, **kwargs)
    self.linear_o = Linear(d_v*num_heads, d_model, **kwargs)

  def forward(
      self,
      q_input: Float[Tensor, "... queries d_embed"],
      v_input: Float[Tensor, "... values d_embed"] | None = None,
      mask: Bool[Tensor, " ... queries keys"] | None = None,
  ) -> Float[Tensor, "... queries d_v"]:
    if v_input is None:
      v_input = q_input
    Q = self.linear_q.forward(q_input) # Float[Tensor, " ... queries d_k"]
    K = self.linear_k.forward(v_input) # Float[Tensor, " ... keys d_k"]
    V = self.linear_v.forward(v_input) # Float[Tensor, " ... keys d_v"]
    Q = rearrange(Q, "... queries (h d_k) -> ... h queries d_k", h=self.n_heads)
    K = rearrange(K, "... keys (h d_k) -> ... h keys d_k", h=self.n_heads)
    V = rearrange(V, "... keys (h d_v) -> ... h keys d_v", h=self.n_heads)
    if self.pos_embed is not None:
      Q: Tensor = self.pos_embed(Q)
      K: Tensor = self.pos_embed(K)
    if mask is None and self.auto_mask:
      n_queries = q_input.shape[-2]
      n_keys = q_input.shape[-2]
      assert n_queries == n_keys
      mask = ~torch.triu(torch.ones(n_queries, n_keys, dtype=torch.bool, device=V.device), diagonal=1)
    result = _scaled_dot_product_attention(Q, K, V, mask)
    result = rearrange(result, "... h queries d_v -> ... queries (h d_v)")
    return self.linear_o.forward(result)


class TransformerBlock(Module):
  def __init__(self, d_model: int, d_ff: int, d_k: int, d_v: int = 0, sig: Module | None = None, pos_embed: Module | None = None, num_heads: int = 1, device=None, dtype=None):
    kwargs = DeviceParams(device=device, dtype=dtype)
    super().__init__()
    d_v = d_v or d_k
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = num_heads
    if sig is None:
      sig = SiLU()

    self.attn = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, pos_embed=pos_embed, num_heads=num_heads, mask=True, **kwargs)
    self.norm1 = RMSNorm(d_model=d_model, **kwargs)
    self.ffn = FFN(d_model=d_model, d_hidden=d_ff, sig=sig, **kwargs)
    self.norm2 = RMSNorm(d_model=d_model, **kwargs)

  def forward(self, x: Tensor):
    attn = x + self.attn.forward(self.norm1.forward(x))
    return attn + self.ffn.forward(self.norm2.forward(attn))


class LLM(Module):
  def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, d_ff: int, theta: float = 1, num_heads: int = 1, sig: Module | None = None, pos_embed: Module | None = None, device=None, dtype=None):
    kwargs = DeviceParams(device=device, dtype=dtype)
    super().__init__()
    self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **kwargs)
    self.vocab_size = vocab_size
    self.context_length = context_length
    self.d_model = d_model
    self.d_ff = d_ff
    d_k = d_model // num_heads
    if pos_embed is None:
      pos_embed = RoPE(theta=theta, d_k=d_k, max_seq_len=context_length, **kwargs)
    self.layers = torch.nn.ModuleList([
      TransformerBlock(d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_k, num_heads=num_heads, sig=sig, pos_embed=pos_embed, **kwargs)
      for _ in range(num_layers)
    ])
    self.norm1 = RMSNorm(d_model=d_model, **kwargs)
    self.out_embed = Linear(d_model, vocab_size, **kwargs)
    self.out_softmax = Softmax()

  def forward(self, token_ids: Int[Tensor, "... vocab"], prob: bool = False) -> Float[Tensor, "... vocab"]:
    x = self.embedding.forward(token_ids)
    for layer in self.layers:
      x = layer.forward(x)
    x = self.norm1.forward(x)
    x = self.out_embed.forward(x)
    if prob:
      return self.out_softmax.forward(x)
    return x


class CrossEntropy(Module):
  def __init__(self, dim=-1, reduce="mean") -> None:
    self.dim = -1
    self.reduce = "mean"

  def forward(self, x: Float[Tensor, "... pred"], k: Int[Tensor, "..."]):
    y = _cross_entory(x, k, dim=self.dim)
    if self.reduce is None:
      return y
    elif self.reduce == "mean":
      return y.mean(dim=self.dim)
    elif self.reduce == "sum":
      return y.sum(dim=self.dim)
    assert False


def _linear(
  d_in: int,
  d_out: int,
  weights: Float[Tensor, " d_out d_in"],
  in_features: Float[Tensor, " ... d_in"],
):
  assert weights.shape == (d_out, d_in)
  assert in_features.shape[-1] == d_in
  return einsum(in_features, weights, "... d_in, d_out d_in -> ... d_out")

def _rope_rotate(Theta: float, n: int, k: int) -> Float[Tensor, "d_n d_k d_k"]:
  """
  $ R_{i,k} &= mat(cos theta_(i,k), -sin theta_(i,k);
                   sin theta_(i,k),  cos theta_(i,k);) \
    theta_(i,k) &= i / Theta^((2k-2)/d) $
  """
  _is = torch.arange(n)
  _ks = torch.arange(k)
  _krs = _ks + 1 - (_ks % 2) * 2
  thetas = einsum(_is, torch.pow(Theta, -_ks[::2] / k), "n, k -> n k")
  thetas = rearrange(thetas, "n k -> (n k)")
  # idx goes like (0, 0), (1, 1), ... (d_k-1, d_k-1)
  _k_idx = repeat(_ks, "k -> (n k) 1", n=n)
  _kr_idx = repeat(_krs, "k -> (n k) 1", n=n)
  _i_idx = repeat(_is, "n -> (n k) 1", k=k)
  idx1 = torch.cat([_i_idx, _k_idx, _k_idx], dim=-1)
  idx2 = torch.cat([_i_idx, _k_idx, _kr_idx], dim=-1)
  idx = torch.stack([idx1[::2], idx2[::2], idx2[1::2], idx1[1::2]])
  v = torch.stack([thetas.cos(), -thetas.sin(), thetas.sin(), thetas.cos()])
  idx = rearrange(idx, "group t p -> p (t group)")
  v = rearrange(v, "group t -> (t group)")
  return torch.sparse_coo_tensor(idx, v, size=(n, k, k))

def _softmax(x: Float[Tensor, "..."], dim = -1) -> Float[Tensor, "..."]:
  x_max = torch.max(x, dim=dim, keepdim=True).values.detach()
  x = (x - x_max).exp()
  return x / x.sum(dim=dim, keepdim=True)

def _scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
  d_k = torch.tensor(K.shape[-1])
  atten = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
  if mask is not None:
    atten = atten.masked_fill(~mask, -torch.inf)
  atten = _softmax(atten / d_k.sqrt(), dim=-1)
  return einsum(atten, V, "... queries keys, ... keys d_v -> ... queries d_v")

def _cross_entory(pred: Float[Tensor, "... pred"], target: Int[Tensor, "..."], dim=-1) -> Float[Tensor, "..."]:
  # -_softmax(pred)[target].log()
  x = pred - pred.max(dim=dim, keepdim=True).values.detach()
  x_sum = x.exp().sum(dim=dim).log()
  return x_sum - x.gather(dim=dim, index=target.unsqueeze(dim)).squeeze(dim)
