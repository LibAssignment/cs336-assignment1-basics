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
