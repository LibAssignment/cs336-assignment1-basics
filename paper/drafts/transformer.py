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
