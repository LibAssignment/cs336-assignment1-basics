# %%
from cs336_basics.utils import chunks_iter, chunk_to_pretokenizer
from collections import Counter
import regex

special_token = b"<|endoftext|>"
special_tokens = [special_token, b"<||>"]
re_special_tokens = '|'.join(regex.escape(s.decode()) for s in special_tokens)
with open('../tests/fixtures/tinystories_sample_5M.txt', 'rb') as f:
  final_words = Counter[str]()
  for chunk in chunks_iter(f, 10, b"<|endoftext|>"):
    for c in regex.split(re_special_tokens, chunk):
      words = chunk_to_pretokenizer(c)
      final_words += words
    # break
  print(final_words)
  print(len(final_words))

# %%
from collections import Counter
def bytes_tuple(b: bytes) -> tuple[bytes, ...]:
  return tuple(b[i:i+1] for i in range(len(b)))

def display_tuples(t: dict[tuple[int, ...], int]):
  return Counter({display_tuple(k): v for k, v in t.items()})
def display_tuple(t: tuple[int, ...]) -> tuple[bytes, ...]:
  return tuple(vocab[c] for c in t)

vocab = [bytes([c]) for c in range(256)]
final_tuples = Counter({tuple(k.encode()): v for k, v in final_words.items()})
merges = list[tuple[int, int]]()
display_tuples(final_tuples)

# %%
# %%
def merge_tuple(f: tuple[int, ...], p: tuple[int, int], n: int) -> tuple[int, ...]:
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

def merge_tuples(f: dict[tuple[int, ...], int], p: tuple[int, int], n: int) -> Counter[tuple[int, ...]]:
  return Counter({merge_tuple(k, p, n):v for k, v in f.items()})

def merge_step(final_tuples: dict[tuple[int, ...], int]):
  tmp_merge = Counter[tuple[int, int]]()
  for k, v in final_tuples.items():
    if len(k) <= 1:
      continue
    # TODO: 0. utf8?
    # TODO: 1. in+g, i+ng, and ing
    # TODO: 2. nnn => counter for b"nn"
    for a, b in zip(k, k[1:]):
      tmp_merge[(a, b)] = tmp_merge.get((a, b), 0) + v
  current_merge = tmp_merge.most_common(1)[0][0]
  final_tuples = merge_tuples(final_tuples, current_merge, len(vocab))
  vocab.append(vocab[current_merge[0]] + vocab[current_merge[1]])
  merges.append(current_merge)
  # merges.append(current_merge)
  return final_tuples, current_merge

# %%
for i in range(10000):
  final_tuples, current_merge = merge_step(final_tuples)
  print(display_tuple(current_merge))

# %%
vocab
# %%
len(vocab)
# %%
