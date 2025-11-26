# %%
from cs336_basics.tokenizer import Tokenizer

special_tokens = ["<|endoftext|>"]
# input_path = '../tests/fixtures/corpus.en'
input_path = '../tests/fixtures/tinystories_sample_5M.txt'
# input_path = './fixtures/TinyStories-train.txt'
# input_path = "./chinese.txt"
vocab_size = 2000

tokenizer = Tokenizer.training_from_file(input_path, special_tokens)
while len(tokenizer.vocabs) < vocab_size:
  current_merge = tokenizer.step()
  if current_merge is None:
    break
  # print(f"merge {tokenizer.apply_idx(current_merge.tp)} => [{len(tokenizer.vocabs)}] {current_merge.freq}")
tokenizer.vocabs

# %%
"""
" hello, world!" => (b' he', b'll', b'o', b',', b' world', b'!')
"hello, world!"  => (b'he', b'll', b'o', b',', b' world', b'!')
"Hello, world!"  => (b'Hello', b',', b' world', b'!')
"""
input_str = "Hello, world! aaa my<|endoftext|>\n\nworld."
idxs = tokenizer.encode(input_str)
print(tokenizer.apply_idx(idxs))
# (b'Hello', b',', b' world', b'!', b' a', b'a', b'a', b' my', b'<|endoftext|>', b'\n', b'\n', b'w', b'or', b'ld', b'.')
assert tokenizer.decode(idxs) == input_str

# %%
with open(input_path) as fin:
  content = fin.read()
print(len(content))

# %%
import time
starttime = time.time()
idxs = tokenizer.encode(content)
elapsed = time.time() - starttime
print(f"encode {len(idxs)} in {elapsed} second")

# %%
