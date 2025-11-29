# %%
import torch
from einops import rearrange, einsum

D = torch.randn((4, 4))
D
# %%
a = torch.tensor([[1,2]])
D[a]

# %% relu
torch.max(torch.randn(10), torch.tensor(0))

# %%
from cs336_basics.modules import GatedLU, Relu

g = GatedLU(10, 20, sig=Relu())
g(torch.randn(2, 10))

# %%
from einops import einsum, repeat
Theta = 1/1000
n, k = 30, 16
_is = torch.arange(n)
_ks = torch.arange(k)
_krs = _ks + 1 - (_ks % 2) * 2
thetas = einsum(_is+1, torch.pow(Theta, _ks[::2] / k), "n, k -> n k")
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
R2 = torch.sparse_coo_tensor(idx, v, size=(n, k, k))

# %%
_R = R2.to_dense()
_R[0, 0, 0:2]

# %%
_R[0]

# %%
thetas[0].cos(), thetas[0].sin()

# %%
torch.cat([_ks, _krs])

# %%
torch.sparse_coo_tensor(torch.tensor([[1,1], [2,2], [3, 3]]).T, [3, 3, 3], (10, 10))
# %%
import torch
mask = torch.tensor([True, False])
x = torch.randn(2)
x_masked = x.masked_fill(~mask, -torch.inf)
x_masked

# %%
~torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)

# %%
import torch
x = torch.randn(2, 5, 4)
t = torch.tensor([1,2,3,2,1,0,0,0,0,1]).reshape(2, 5)
# select values from the last dimension using t: result shape (2, 5)
x_selected = torch.gather(x, 2, t.unsqueeze(-1)).squeeze(-1)
# %%
# gather along last dim using t (indices), result shape (2, 5)
x_selected = torch.gather(x, -1, t.unsqueeze(-1)).squeeze(-1)
x_selected

# %%
from cs336_basics.modules import _cross_entory
_cross_entory(torch.randn(4), torch.tensor(2))
_cross_entory(torch.randn(2, 4), torch.tensor([2, 3]))
_cross_entory(torch.randn(2, 3, 4), torch.randint(0, 3, (2, 3)))

# %%
import torch
from cs336_basics.optimizer import SGD, AdamW
from cs336_basics.modules import Linear
m = Linear(5, 3)
s = AdamW(m.parameters(), lr=0.001, weight_decay=0.9)
print(s)
beta1 = s.defaults['beta1']
beta2 = s.defaults['beta2']
steps = torch.arange(10000) + 1
(1 - beta1 ** steps).sqrt() / (1 - beta2 ** steps)

# %%
import math
import matplotlib.pyplot as plt
def _lr_cosine(t: int, alpha_range: tuple[float, float], t_warmup: int, t_cosine: int):
  a_min, a_max = alpha_range
  if t < t_warmup:
    return t / t_warmup * a_max
  if t < t_cosine:
    k = (t - t_warmup) / (t_cosine - t_warmup)
    return a_min + (1 + math.cos(k * math.pi)) * (a_max -a_min) / 2
  return a_min
t = torch.arange(1000)
lr = [_lr_cosine(int(t), alpha_range=(0.1, 1), t_warmup=50, t_cosine=1000) for t in t]
plt.plot(lr)

# %%
import torch
g = torch.stack([torch.tensor(0), torch.tensor(1)]).sum().sqrt()
if g < 1:
  print("g < 1")

# %%
