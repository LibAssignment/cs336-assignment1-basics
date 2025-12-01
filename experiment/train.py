# %%
import os
os.environ['HF_HUB_OFFLINE'] = "1"
from datasets import load_dataset

# dataset = load_dataset("roneneldan/TinyStories")
# %%
# from cs336_basics.tokenizer import Tokenizer, train_bpe
# from pathlib import Path

# fixture = "TinyStoriesV2-GPT4-train.txt"
# fixture_url = [
#   "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
#   "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",

#   "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz",
#   "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz",
# ]

# if not Path("tokenizer.json").exists():
#   tokenizer = train_bpe()

# %%
from cs336_basics.tokenizer import Tokenizer, train_bpe
from pathlib import Path
import logging

logging.basicConfig(filename="train.log", level=logging.DEBUG)
fixture = Path(__file__).parent.parent / "tests/fixtures/tinystories_sample_5M.txt"
name = fixture.stem
tokenizer_filename = f"tokenizer.{name}.json"
merges_filename = f"merges.{name}.txt"

if not Path(tokenizer_filename).exists():
  tokenizer = train_bpe(fixture, 1000, ["<|endoftext|>"], chunks=10)
  tokenizer.save_to_files(tokenizer_filename, merges_filename)

# %%
import numpy as np
idx_filename = f"{name}.idx.npz"
tokenizer = Tokenizer.from_files(tokenizer_filename, merges_filename)

if not Path(idx_filename).exists():
  result = []
  with open(fixture, "r") as f:
    result.extend(tokenizer.encode_iterable(f))
  result = np.array(result)
  np.savez(idx_filename, result, allow_pickle=False)

# %%
idx = np.load(idx_filename)['arr_0'] # type: np.ndarray

# %%
from cs336_basics.training import RandomTokenDataLoader
context_length = 1024
vocab_size = tokenizer.vocab_size
batch_size = 32
d_model = 256
d_ff = d_model * 4
num_heads = 4
theta = 10000
num_layers = 3
device = "mps"

dataset = RandomTokenDataLoader(idx, batch_size=batch_size, context_length=context_length, device=device)

# %%
from cs336_basics.modules import LLM, _cross_entory
from cs336_basics.optimizer import AdamW
a = LLM(vocab_size=vocab_size, num_layers=3, context_length=context_length, d_model=d_model, d_ff=d_ff, num_heads=num_heads, theta=theta, device=device)
optimizer = AdamW(a.parameters())

# %%
x, y = dataset[0]

# %%
y_hat = a.forward(x)
loss = _cross_entory(y_hat, y).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()
loss
# %%
