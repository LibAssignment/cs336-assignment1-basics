import numpy
import torch
from tests.adapters import run_silu
from torch.nn.functional import silu

def test_silu_backward():
  x = torch.arange(200) * 2. - 200
  x1 = x.clone().detach().requires_grad_()
  y1 = run_silu(x1)
  y1.sum().backward()
  assert y1.isnan().count_nonzero() == 0
  assert x1.grad is not None
  # assert x1.grad.isnan().count_nonzero() == 0

  x2 = x.clone().detach().requires_grad_()
  y2 = silu(x2)
  y2.sum().backward()
  assert x2.grad is not None
  numpy.testing.assert_allclose(
    y1.detach().numpy(), y2.detach().numpy(),
    atol=1e-6,
  )
  numpy.testing.assert_allclose(
    x1.grad.numpy(), x2.grad.numpy(),
    atol=1e-6,
  )
