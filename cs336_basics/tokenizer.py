from collections import Counter
from collections.abc import Iterable
from dataclasses import field, dataclass
from typing import Self, TypeAlias
import regex
import heapq
import os

from .utils import chunk_to_pretokenizer, chunks_iter

Idx: TypeAlias = int

@dataclass
class PreToken:
  src: bytes
  idxs: list[Idx]
  freq: int

@dataclass
class PreMerge:
  tp: tuple[Idx, Idx]
  occurs_in: set[int] = field(default_factory=set)
  freq: int = 0

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

  def display_tuple(self, t: Iterable[Idx]) -> tuple[bytes, ...]:
    return tuple(self.vocabs[c] for c in t)

  @classmethod
  def load_file(cls, input_path: str | os.PathLike, special_tokens: list[str], desired_num_chunks=1000) -> Self:
    re_special_tokens = '|'.join(regex.escape(s) for s in special_tokens)

    with open(input_path, 'rb') as f:
      final_words = Counter[str]()
      for chunk in chunks_iter(f, desired_num_chunks=desired_num_chunks, split_special_token=special_tokens[0].encode()):
        for c in regex.split(re_special_tokens, chunk):
          words = chunk_to_pretokenizer(c)
          final_words += words

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
    # self.current_tuples = Counter({merge_tuple(k):v for k, v in self.current_tuples.items()})
    for t in self.current_tuples:
      t.idxs = merge_tuple(t.idxs)
    self.vocabs.append(self.vocabs[m.tp[0]] + self.vocabs[m.tp[1]])
    self.merges.append(m)

  def step(self):
    self.pre_merges.clear()
    for i, t in enumerate(self.current_tuples):
      if len(t.idxs) <= 1:
        continue
      # TODO: 0. utf8?
      # TODO: 2. nnn => counter for b"nn"
      for tp in zip(t.idxs, t.idxs[1:]):
        m = self.pre_merges.get(tp)
        if m is None:
          m = PreMerge(tp)
          self.pre_merges[tp] = m
        m.freq += t.freq
        m.occurs_in.add(i)
    most_common = heapq.nlargest(1, self.pre_merges.values(), lambda m: (m.freq, self.vocabs[m.tp[0]], self.vocabs[m.tp[1]]))
    current_merge = None
    if most_common:
      current_merge = most_common[0]
      self.merge_tuples(current_merge)
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
