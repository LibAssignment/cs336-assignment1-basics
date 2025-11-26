import torch.nn
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(torch.nn.Module):
  def __init__(self, d_in: int, d_out: int, *, weights: Float[Tensor, "d_out d_in"] | None = None):
    self.d_in = d_in
    self.d_out = d_out
    if weights is None:
      self.weights = torch.randn(d_out, d_in)
    else:
      self.weights = weights

  def forward(self, in_features: Float[Tensor, " ... d_in"]):
    return linear(self.d_in, self.d_out, self.weights, in_features)

def linear(
  d_in: int,
  d_out: int,
  weights: Float[Tensor, " d_out d_in"],
  in_features: Float[Tensor, " ... d_in"],
):
  assert weights.shape == (d_out, d_in)
  assert in_features.shape[-1] == d_in
  return in_features @ weights.T
