# %%
from cs336_basics.utils import find_chunk_boundaries

with open('../tests/fixtures/tinystories_sample.txt', 'rb') as f:
  boundaries = find_chunk_boundaries(f, 10, b"<|endoftext|>")

print(f"[{len(boundaries)}]: {boundaries}")

# %%
