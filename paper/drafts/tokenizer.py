# %%
from collections import Counter
from cs336_basics.tokenizer import Tokenizer
from pathlib import Path

workspace_dir = Path(__file__).parent.parent.parent

special_tokens = ["<|endoftext|>"]
# input_path = '../tests/fixtures/corpus.en'
input_path = workspace_dir / 'tests/fixtures/tinystories_sample_5M.txt'
# input_path = workspace_dir / 'experiment/fixtures/TinyStoriesV2-GPT4-train.txt'
# input_path = '../experiment/fixtures/TinyStories-train.txt'
# input_path = "../experiment/chinese.txt"
name = input_path.stem
vocab_size = 2000

tokenizer = Tokenizer.training_from_file(input_path, special_tokens)
import json
tokens = Counter({v.src.decode():v.freq for v in tokenizer.current_tokens})
with open(f"../tokens.{name}.json", "w") as f:
  json.dump(tokens, f, ensure_ascii=False, indent=2)

# %%
while len(tokenizer.vocabs) < vocab_size:
  current_merge = tokenizer.step()
  if current_merge is None:
    break
  # print(f"merge {tokenizer.apply_idx(current_merge.tp)} => [{len(tokenizer.vocabs)}] {current_merge.freq}")
tokenizer.vocabs

# %%
tokenizer.save_to_files(f"../vocab.{name}.json", f"../merges.{name}.txt", freq=True)

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
