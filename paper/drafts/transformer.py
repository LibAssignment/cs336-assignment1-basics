# %%
import torch
from einops import rearrange, einsum

D = torch.randn((4, 4))
D
# %%
a = torch.tensor([[1,2]])
D[a]
# %%
a
# %%
