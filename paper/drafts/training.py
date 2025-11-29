# %%
from collections.abc import Iterable
import numpy as np
from jaxtyping import Int

class DataSlice:
  tokens: Int[np.ndarray, "idx"]
  sp: list[int]

  def __init__(self, arr: Int[np.ndarray, "idx"], special_token: int):
    self.tokens = arr
    self.sp = np.where(arr == special_token)[0].tolist()

def _token_iter(array: Int[np.ndarray, "idx"], special_token: int, max_length: int) -> Iterable[Int[np.ndarray, "idx"]]:
  current = DataSlice(array[:max_length], special_token)
  i = max_length
  while i < array.size:
    j1 = 0
    if not current.sp:
      yield current.tokens
      i += max_length
      while i < array.size and not current.sp:
        current = DataSlice(array[i:i+max_length], special_token)
        i += max_length
      if current.sp:
        j1 = current.sp[0] + 1
      else:
        break
    for j2 in current.sp:
      if j1 < j2:
        yield current.tokens[j1:j2]
      j1 = j2 + 1
    current = DataSlice(np.concat([current.tokens[j1:], array[i:i+max_length]]), special_token)
    i += max_length

sample = np.arange(1000) % 93
list(_token_iter(sample, 0, 10))
# %%
from cs336_basics.training import TokenDataLoader
loader = TokenDataLoader(sample, 0, 4, 10)
list(loader)

# %%
next(iter(loader))
# %%
