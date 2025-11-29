from collections.abc import Iterable
from torch import Tensor
import torch
from torch.utils.data import IterableDataset, Dataset
import numpy as np
from jaxtyping import Bool, Float, Int

class TokenDataLoader(IterableDataset):
  def __init__(self, array: Int[np.ndarray, "idx"], special_token: int, batch_size: int, context_length: int, device=None):
    self.array = array
    self.special_token = special_token
    self.batch_size = batch_size
    self.context_length = context_length
    self.device = device

  def __iter__(self):
    def send(x): return _split_xy(x, self.device)
    inputs = _token_iter(self.array, self.special_token, self.context_length)
    batchs = _batch_iter(inputs, batch_size=self.batch_size, max_length=self.context_length)
    return map(send, batchs)

class RandomTokenDataLoader(Dataset):
  def __init__(self, array: Int[np.ndarray, "idx"], batch_size: int, context_length: int, device=None):
    self.array = array
    self.batch_size = batch_size
    self.context_length = context_length
    self.device = device

  def __getitem__(self, _):
    idx = torch.randint(0, len(self.array) - self.context_length, (self.batch_size,))
    batch = []
    for i in idx:
      batch.append(self.array[i:i+self.context_length+1])
    batch = np.stack(batch)
    return _split_xy(batch, self.device)


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

def _batch_iter(inputs: Iterable[Int[np.ndarray, "idx"]], batch_size: int, max_length: int) -> Iterable[Int[np.ndarray, "batch idx"]]:
  batch = []
  for s in inputs:
    batch.append(s[:max_length+1])
    if len(batch) == batch_size:
      batch_arr = np.zeros(shape=(len(batch), max_length+1), dtype=batch[0].dtype)
      for i, j in enumerate(batch):
        batch_arr[i, :len(j)] = j
      batch.clear()
      yield batch_arr
  batch_arr = np.zeros(shape=(len(batch), max_length+1), dtype=batch[0].dtype)
  for i, j in enumerate(batch):
    batch_arr[i, :len(j)] =  j
  yield batch_arr

def _split_xy(input: np.ndarray, device=None):
  return Tensor(input[:, :-1], device=device), Tensor(input[:, 1:], device=device)
