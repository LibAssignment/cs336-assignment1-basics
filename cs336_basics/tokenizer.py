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

ENCODING = "utf-8"

@dataclass
class PreToken:
  src: bytes
  idxs: list[Idx]
  freq: int

@dataclass
class PreMerge:
  tp: tuple[Idx, Idx]
  content: tuple[bytes, bytes]
  tar: Idx = -1
  occurs_in: set[int] = field(default_factory=set)
  freq: int = 0

  @classmethod
  def create(cls, tp: tuple[Idx, Idx], *, vocabs: Vocabs):
    return cls(tp, (vocabs[tp[0]], vocabs[tp[1]]))

  @classmethod
  def create_lookup(cls, content: tuple[bytes, bytes], *, vocab_rev: dict[bytes, int]):
    return cls((vocab_rev[content[0]], vocab_rev[content[1]]), content, tar=vocab_rev.get(content[0]+content[1], -1))

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
    self.vocab_rev = {v: i for i, v in self.vocabs.items()}
    self.merges = [PreMerge.create_lookup(tp, vocab_rev=self.vocab_rev) for tp in merges]
    self.pre_merges = dict[tuple[Idx, Idx], PreMerge]()
    self.current_tokens = list[PreToken]()
    self.special_tokens = [] if special_tokens is None else special_tokens
    self.completed = False

    self.finish()

  @classmethod
  def create_training(cls, words: dict[str, int], *, special_tokens: list[str] | None = None) -> Self:
    vocabs = [bytes([c]) for c in range(256)] + [s.encode(ENCODING) for s in special_tokens or []]
    this = cls(vocabs, [], special_tokens=special_tokens)
    this.current_tokens.extend(
      PreToken(src=k.encode(ENCODING), idxs=list(k.encode(ENCODING)), freq=v)
      for k, v in words.items()
    )
    return this

  def finish(self):
    if self.completed:
      return
    self.vocab_rev = {v: i for i, v in self.vocabs.items()}
    sorted_special_tokens = sorted(self.special_tokens, reverse=True)
    self.re_special_tokens = regex.compile('|'.join(map(regex.escape, sorted_special_tokens)))
    self.re_special_tokens_capture = regex.compile('(' + '|'.join(map(regex.escape, sorted_special_tokens)) + ')')
    for m in self.merges:
      m.tar = self.vocab_rev[m.content[0] + m.content[1]]
    self.completed = True

  @classmethod
  def from_files(cls, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, special_tokens: list[str] | None = None):
    with open(vocab_filepath, "r") as f:
      data: dict[str, int] = json.load(f) # typing: dict
    vocabs = {k: cls.from_visable(v) for v,k in data.items()}
    merges = list[tuple[bytes, bytes]]()
    with open(merges_filepath, "r") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        (a, b) = line.strip().split(' ')
        merges.append((cls.from_visable(a), cls.from_visable(b)))
    return cls(vocabs, merges, special_tokens)

  def save_to_files(self, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike):
    vocabs= self.visible_vocabs_dict
    with open(vocab_filepath, "w") as f:
      json.dump(vocabs, f, ensure_ascii=False, indent=2)
    with open(merges_filepath, "w") as f:
      for line in self.visible_merges_list:
        f.write(f"{line}\n")

  @classmethod
  def training_from_file(cls, input_path: str | os.PathLike, special_tokens: list[str], desired_num_chunks=1024) -> Self:
    re_special_tokens = '|'.join(regex.escape(s) for s in special_tokens)

    final_words = get_words_parallel(
      input_path,
      desired_num_chunks=desired_num_chunks,
      split_special_token=special_tokens[0].encode(ENCODING),
      re_special_tokens=re_special_tokens
    )

    return cls.create_training(final_words, special_tokens=special_tokens)

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

  @property
  def visible_vocabs_dict(self):
    return {self.to_visable(v): i for i, v in self.vocabs.items()}

  @property
  def visible_merges_list(self) -> list[str]:
    return [f"{self.to_visable(m.content[0])} {self.to_visable(m.content[1])}" for m in self.merges]

  @property
  def merges_display(self):
    return [(self.vocabs[m.tp[0]], self.vocabs[m.tp[1]]) for m in self.merges]

  def encode_byte(self, b: int) -> Idx:
    if rev := getattr(self, "_vocab_byte_rev", None):
      return rev[b]
    self._vocab_byte_rev = {i: self.vocab_rev[bytes([i])] for i in range(256)}
    return self._vocab_byte_rev[b]

  def _encode(self, s: str) -> list[Idx]:
    if not self.completed:
      self.finish()
    # TODO handle special_tokens
    idxs = [self.encode_byte(i) for i in s.encode(ENCODING)]
    if not idxs:
      return idxs
    idx_len = len(idxs)
    nxt = [i+1 for i in range(idx_len)]
    for m in self.merges:
      i = 0
      while i < idx_len:
        j = nxt[i]
        # no next idx
        if j >= idx_len:
          break
        if (idxs[i], idxs[j]) == m.tp:
          idxs[i] = m.tar
          nxt[i] = nxt[j]
        i = nxt[i]
    i = 0
    result = list[Idx]()
    while i < idx_len:
      result.append(idxs[i])
      i = nxt[i]
    return result

  def encode(self, s: str) -> list[Idx]:
    result = list[Idx]()
    for chunk in regex.split(self.re_special_tokens_capture, s):
      if chunk not in self.special_tokens:
        result.extend(self._encode(chunk))
      else:
        result.append(self.vocab_rev[chunk.encode()])
    return result

  def decode(self, ids: list[int]) -> str:
    bstr = b"".join(self.vocabs[i] for i in ids)
    return bstr.decode(ENCODING, errors="replace")

  # below all training related
  def merge_tuples(self, m: PreMerge):
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
    self.completed = False
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
