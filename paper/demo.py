# %%
from cs336_basics.utils import chunks_iter, chunk_to_pretokenizer

special_token = b"<|endoftext|>"
with open('../tests/fixtures/tinystories_sample.txt', 'rb') as f:
  for chunk in chunks_iter(f, 10, b"<|endoftext|>"):
    for c in chunk.split(special_token.decode("utf8")):
      words = chunk_to_pretokenizer(c)
      print(words)
    # break

# %%
