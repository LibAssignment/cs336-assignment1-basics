from collections import Counter
from collections.abc import Iterable
from dataclasses import field, dataclass
import json
from typing import Self, TypeAlias
import regex
import os

from .utils import get_words_parallel, gpt2_bytes_to_unicode
from typing import overload

Idx: TypeAlias = int
Vocabs: TypeAlias = dict[int, bytes]

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

  @classmethod
  def create(cls, tp: tuple[Idx, Idx], *, vocabs: Vocabs):
    return cls(tp, (vocabs[tp[0]], vocabs[tp[1]]))

  @classmethod
  def create_lookup(cls, content: tuple[bytes, bytes], *, vocabs: Vocabs):
    def find_idx(vocabs: Vocabs, s: bytes):
      for k, v in vocabs.items():
        if v == s: return k
      return -1
    return cls((find_idx(vocabs, content[0]), find_idx(vocabs, content[1])), content)

  def add(self, i: int, v: int):
    self.occurs_in.add(i)
    self.freq += v

  def remove(self, i: int, v: int):
    self.occurs_in.discard(i)
    self.freq -= v

  @property
  def key(self):
    return self.freq, self.content

  def __gt__(self, other: Self) -> bool:
    return self.key > other.key

class Tokenizer:
  def __init__(self, vocabs: dict[int, bytes] | list[bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
    self.vocabs = vocabs if isinstance(vocabs, dict) else {i: v for i,v in enumerate(vocabs)}
    self.max_vocab_idx = max(self.vocabs.keys())
    self.merges = [PreMerge.create_lookup(tp, vocabs=self.vocabs) for tp in merges]
    self.pre_merges = dict[tuple[Idx, Idx], PreMerge]()
    self.current_tokens = list[PreToken]()
    self.special_tokens = [] if special_tokens is None else special_tokens

  @classmethod
  def create_training(cls, words: dict[str, int], *, special_tokens: list[str] | None = None) -> Self:
    vocabs = [bytes([c]) for c in range(256)] + [s.encode() for s in special_tokens or []]
    this = cls(vocabs, [], special_tokens=special_tokens)
    this.current_tokens.extend(
      PreToken(src=k.encode(), idxs=list(k.encode()), freq=v)
      for k, v in words.items()
    )
    return this

  def add_vocab(self, vocab: bytes, *, idx: int | None = None):
    if idx is None:
      self.max_vocab_idx += 1
      idx = self.max_vocab_idx
    self.vocabs[idx] = vocab
    return idx

  def tokens_counter(self, t: list[PreToken] | None = None):
    if t is None:
      t = self.current_tokens
    return Counter({self.apply_idx(k.idxs): k.freq for k in t})

  @overload
  def apply_idx(self, t: tuple[int, int]) -> tuple[bytes, bytes]: ...
  @overload
  def apply_idx(self, t: Iterable[Idx]) -> tuple[bytes, ...]: ...
  def apply_idx(self, t):
    return tuple(self.vocabs[c] for c in t)

  @staticmethod
  def to_visable(b: bytes, *, d: dict[int, str] = gpt2_bytes_to_unicode()) -> str:
    return "".join(d[i] for i in b)

  @staticmethod
  def from_visable(s: str, *, d: dict[str, int] = {v:k for k, v in gpt2_bytes_to_unicode().items()}) -> bytes:
    return bytes([d[i] for i in s])

  @classmethod
  def from_files(cls, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, special_tokens: list[str] | None = None):
    with open(vocab_filepath, "r") as f:
      data: dict[str, int] = json.load(f) # typing: dict
    vocabs = {k: cls.from_visable(v) for v,k in data.items()}
    merges = list[tuple[bytes, bytes]]()
    with open(merges_filepath, "r") as f:
      for line in f:
        (a, b) = line.strip().split(' ')
        merges.append((cls.from_visable(a), cls.from_visable(b)))
    return cls(vocabs, merges, special_tokens)

  @classmethod
  def training_from_file(cls, input_path: str | os.PathLike, special_tokens: list[str], desired_num_chunks=1024) -> Self:
    re_special_tokens = '|'.join(regex.escape(s) for s in special_tokens)

    final_words = get_words_parallel(
      input_path,
      desired_num_chunks=desired_num_chunks,
      split_special_token=special_tokens[0].encode(),
      re_special_tokens=re_special_tokens
    )

    return cls.create_training(final_words, special_tokens=special_tokens)

  @property
  def visible_vocabs_dict(self):
    return {self.to_visable(v): i for i, v in self.vocabs.items()}

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
      t = self.current_tokens[i]
      t.idxs = merge_tuple(t.idxs)
    self.pre_merges.pop(m.tp)
    self.add_vocab(self.vocabs[m.tp[0]] + self.vocabs[m.tp[1]])
    self.merges.append(m)

  def init_training(self):
    self.pre_merges.clear()
    for i, t in enumerate(self.current_tokens):
      if len(t.idxs) <= 1:
        continue
      # TODO: 0. utf8?
      # TODO: 2. nnn => counter for b"nn"
      for tp in zip(t.idxs, t.idxs[1:]):
        m = self.pre_merges.get(tp)
        if m is None:
          m = PreMerge.create(tp, vocabs=self.vocabs)
          self.pre_merges[tp] = m
        m.add(i, t.freq)

  def step(self):
    if len(self.pre_merges) == 0:
      self.init_training()
    if len(self.pre_merges) == 0:
      return
    # most_common = heapq.nlargest(1, self.pre_merges.values(), lambda m: (m.freq, self.vocabs[m.tp[0]], self.vocabs[m.tp[1]]))
    # current_merge = most_common[0] if most_common else None
    # TODO: bottleneck here
    current_merge = max(self.pre_merges.values())
    if current_merge is None:
      return
    # occurs_in = sorted(current_merge.occurs_in)
    for i in current_merge.occurs_in:
      t = self.current_tokens[i]
      for tp in zip(t.idxs, t.idxs[1:]):
        if tp != current_merge.tp:
          self.pre_merges[tp].remove(i, t.freq)
    self.merge_tuples(current_merge)
    for i in current_merge.occurs_in:
      t = self.current_tokens[i]
      for tp in zip(t.idxs, t.idxs[1:]):
        m = self.pre_merges.get(tp)
        if m is None:
          m = PreMerge.create(tp, vocabs=self.vocabs)
          self.pre_merges[tp] = m
        m.add(i, t.freq)
    return current_merge

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  tokenizer = Tokenizer.training_from_file(input_path, special_tokens)
  while len(tokenizer.vocabs) < vocab_size:
    current = tokenizer.step()
    if current is None:
      break

  return dict(tokenizer.vocabs), tokenizer.merges_display
