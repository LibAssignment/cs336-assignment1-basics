from collections import Counter
from typing import Self
import regex

from .utils import chunk_to_pretokenizer, chunks_iter

class Tokenizer:
  def __init__(self, words: dict[str, int]) -> None:
    self.vocabs = [bytes([c]) for c in range(256)]
    self.merges = list[tuple[int, int]]()
    self.current_tuples = Counter({tuple(k.encode()): v for k, v in words.items()})
    self._tmp_merge = None

  def display_tuples(self, t: dict[tuple[int, ...], int] | None = None):
    if t is None:
      t = self.current_tuples
    return Counter({self.display_tuple(k): v for k, v in t.items()})

  def display_tuple(self, t: tuple[int, ...]) -> tuple[bytes, ...]:
    return tuple(self.vocabs[c] for c in t)

  @classmethod
  def load_file(cls, input_path: str, special_tokens: list[str]) -> Self:
    re_special_tokens = '|'.join(regex.escape(s) for s in special_tokens)

    with open(input_path, 'rb') as f:
      final_words = Counter[str]()
      for chunk in chunks_iter(f, 1000, special_tokens[0].encode()):
        for c in regex.split(re_special_tokens, chunk):
          words = chunk_to_pretokenizer(c)
          final_words += words

    return cls(final_words)

  @property
  def vocabs_dict(self):
    return {i: v for i, v in enumerate(self.vocabs)}

  @property
  def merges_display(self):
    return [(self.vocabs[a], self.vocabs[b]) for a, b in self.merges]


  def merge_tuples(self, p: tuple[int, int]) :
    n = len(self.vocabs)
    def merge_tuple(f: tuple[int, ...]) -> tuple[int, ...]:
      i = 0
      result = list[int]()
      while i < len(f):
        if i < len(f) - 1 and f[i] == p[0] and f[i+1] == p[1]:
          result.append(n)
          i += 2
        else:
          result.append(f[i])
          i += 1
      if len(f) == len(result):
        return f
      return tuple(result)
    self.current_tuples = Counter({merge_tuple(k):v for k, v in self.current_tuples.items()})
    self.vocabs.append(self.vocabs[p[0]] + self.vocabs[p[1]])
    self.merges.append(p)

  def step(self):
    tmp_merge = Counter[tuple[int, int]]()
    for k, v in self.current_tuples.items():
      if len(k) <= 1:
        continue
      # TODO: 0. utf8?
      # TODO: 1. in+g, i+ng, and ing
      # TODO: 2. nnn => counter for b"nn"
      for a, b in zip(k, k[1:]):
        tmp_merge[(a, b)] = tmp_merge.get((a, b), 0) + v
    self._tmp_merge = tmp_merge
    current_merge = tmp_merge.most_common(1)[0][0]
    self.merge_tuples(current_merge)
    # merges.append(current_merge)
    return current_merge

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  tokenizer = Tokenizer.load_file(input_path, special_tokens)
  while len(tokenizer.vocabs) < vocab_size:
    tokenizer.step()

  return tokenizer.vocabs_dict, tokenizer.merges_display
