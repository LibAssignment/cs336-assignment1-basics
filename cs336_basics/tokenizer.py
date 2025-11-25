from collections import Counter
from collections.abc import Iterable
from dataclasses import field, dataclass
from typing import Self, TypeAlias
import regex
import os

from .utils import get_words_parallel
from typing import overload

Idx: TypeAlias = int

@dataclass
class PreToken:
  src: bytes
  idxs: list[Idx]
  freq: int

@dataclass
class PreMerge:
  tp: tuple[Idx, Idx]
  content: tuple[bytes, bytes]
  occurs_in: set[int] = field(default_factory=set)
  freq: int = 0

  def add(self, i: int, v: int):
    self.occurs_in.add(i)
    self.freq += v

  def remove(self, i: int, v: int):
    self.occurs_in.discard(i)
    self.freq -= v

  @property
  def key(self):
    return self.freq, self.content

class Tokenizer:
  def __init__(self, words: dict[str, int], *, special_tokens: list[str] | None = None) -> None:
    self.special_tokens = [] if special_tokens is None else special_tokens
    special_tokens_len = len(self.special_tokens)
    self.vocabs = [s.encode() for s in self.special_tokens] + [bytes([c]) for c in range(256)]
    self.merges = list[PreMerge]()
    self.pre_merges = dict[tuple[Idx, Idx], PreMerge]()
    self.current_tuples = [
      PreToken(src=k.encode(), idxs=[special_tokens_len+c for c in k.encode()], freq=v)
      for k, v in words.items()
    ]
    self._tmp_merge = None

  def display_tuples(self, t: list[PreToken] | None = None):
    if t is None:
      t = self.current_tuples
    return Counter({self.display_tuple(k.idxs): k.freq for k in t})

  @overload
  def display_tuple(self, t: tuple[int, int]) -> tuple[bytes, bytes]: ...
  @overload
  def display_tuple(self, t: Iterable[Idx]) -> tuple[bytes, ...]: ...
  def display_tuple(self, t):
    return tuple(self.vocabs[c] for c in t)

  @classmethod
  def load_file(cls, input_path: str | os.PathLike, special_tokens: list[str], desired_num_chunks=1024) -> Self:
    re_special_tokens = '|'.join(regex.escape(s) for s in special_tokens)

    final_words = get_words_parallel(
      input_path,
      desired_num_chunks=desired_num_chunks,
      split_special_token=special_tokens[0].encode(),
      re_special_tokens=re_special_tokens
    )

    return cls(final_words, special_tokens=special_tokens)

  @property
  def vocabs_dict(self):
    return {i: v for i, v in enumerate(self.vocabs)}

  @property
  def merges_display(self):
    return [(self.vocabs[m.tp[0]], self.vocabs[m.tp[1]]) for m in self.merges]


  def merge_tuples(self, m: PreMerge) :
    n = len(self.vocabs)
    def merge_tuple(f: list[Idx]) -> list[Idx]:
      i = 0
      result = list[Idx]()
      while i < len(f):
        if i < len(f) - 1 and (f[i], f[i+1]) == m.tp:
          result.append(n)
          i += 2
        else:
          result.append(f[i])
          i += 1
      if len(f) == len(result):
        return f
      f.clear()
      f.extend(result)
      return f
    for i in m.occurs_in:
      t = self.current_tuples[i]
      t.idxs = merge_tuple(t.idxs)
    self.pre_merges.pop(m.tp)
    self.vocabs.append(self.vocabs[m.tp[0]] + self.vocabs[m.tp[1]])
    self.merges.append(m)

  def init(self):
    self.pre_merges.clear()
    for i, t in enumerate(self.current_tuples):
      if len(t.idxs) <= 1:
        continue
      # TODO: 0. utf8?
      # TODO: 2. nnn => counter for b"nn"
      for tp in zip(t.idxs, t.idxs[1:]):
        m = self.pre_merges.get(tp)
        if m is None:
          m = PreMerge(tp, self.display_tuple(tp))
          self.pre_merges[tp] = m
        m.add(i, t.freq)

  def step(self):
    if len(self.pre_merges) == 0:
      self.init()
    if len(self.pre_merges) == 0:
      return
    # most_common = heapq.nlargest(1, self.pre_merges.values(), lambda m: (m.freq, self.vocabs[m.tp[0]], self.vocabs[m.tp[1]]))
    # current_merge = most_common[0] if most_common else None
    current_merge = max(self.pre_merges.values(), key=lambda m: m.key)
    if current_merge is None:
      return
    # occurs_in = sorted(current_merge.occurs_in)
    for i in current_merge.occurs_in:
      t = self.current_tuples[i]
      for tp in zip(t.idxs, t.idxs[1:]):
        if tp != current_merge.tp:
          self.pre_merges[tp].remove(i, t.freq)
    self.merge_tuples(current_merge)
    for i in current_merge.occurs_in:
      t = self.current_tuples[i]
      for tp in zip(t.idxs, t.idxs[1:]):
        m = self.pre_merges.get(tp)
        if m is None:
          m = PreMerge(tp, self.display_tuple(tp))
          self.pre_merges[tp] = m
        m.add(i, t.freq)
    return current_merge

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  tokenizer = Tokenizer.load_file(input_path, special_tokens)
  while len(tokenizer.vocabs) < vocab_size:
    current = tokenizer.step()
    if current is None:
      break

  return tokenizer.vocabs_dict, tokenizer.merges_display
