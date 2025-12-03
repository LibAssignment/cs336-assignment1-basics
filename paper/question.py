# %%
c = chr(0)
print(c) # '\x00'
print(repr(c)) # "'\\x00'"
print(f"<{c}>") # <>

# %%
test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode('utf-8')
print(utf8_encoded) # b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
print(type(utf8_encoded)) # <class 'bytes'>
print(list(utf8_encoded)) # [104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
print(len(test_string)) # 13
print(len(utf8_encoded)) # 23

# %%
utf16_encoded = test_string.encode('utf-16')
print(utf16_encoded) # b'\xff\xfeh\x00e\x00l\x00l\x00o\x00!\x00 \x00S0\x930k0a0o0!\x00'
print(list(utf16_encoded)) # [255, 254, 104, 0, 101, 0, 108, 0, 108, 0, 111, 0, 33, 0, 32, 0, 83, 48, 147, 48, 107, 48, 97, 48, 111, 48, 33, 0]
print(len(utf16_encoded)) # 28

# %%
utf32_encoded = test_string.encode('utf-32')
print(utf32_encoded) # b'\xff\xfe\x00\x00h\x00\x00\x00e\x00\x00\x00l\x00\x00\x00l\x00\x00\x00o\x00\x00\x00!\x00\x00\x00 \x00\x00\x00S0\x00\x00\x930\x00\x00k0\x00\x00a0\x00\x00o0\x00\x00!\x00\x00\x00'
print(list(utf32_encoded)) # [255, 254, 0, 0, 104, 0, 0, 0, 101, 0, 0, 0, 108, 0, 0, 0, 108, 0, 0, 0, 111, 0, 0, 0, 33, 0, 0, 0, 32, 0, 0, 0, 83, 48, 0, 0, 147, 48, 0, 0, 107, 48, 0, 0, 97, 48, 0, 0, 111, 48, 0, 0, 33, 0, 0, 0]
print(len(utf32_encoded)) # 56

# %%
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  return "".join([bytes([b]).decode("utf-8") for b in bytestring])

decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")) # 'hello'

# %%
try:
  decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
except UnicodeDecodeError:
  pass

# %%
def cannot_decode(bytestring: bytes):
  try:
    return bytestring.decode('utf-8')
  except UnicodeDecodeError:
    print(f"cannot decode {bytestring}")

cannot_decode(b'\xff\x00')
cannot_decode(b'\xff')

# %%
from cs336_basics.tokenizer import Tokenizer
import logging
import os
import time

special_tokens = ["<|endoftext|>"]
input_path = '../experiment/fixtures/TinyStoriesV2-GPT4-train.txt'
vocab_size = 10_000

tokenizer = None
if os.path.exists(input_path):
  os.makedirs("logs", exist_ok=True)
  logging.basicConfig(level=logging.DEBUG, filename="logs/tokenizer.log")
  starttime = time.time()
  tokenizer = Tokenizer.training_from_file(input_path, special_tokens)
  while len(tokenizer.vocabs) < vocab_size:
    current_merge = tokenizer.step()
    if current_merge is None:
      break
    logging.debug(f"merge {tokenizer.apply_idx(current_merge.tp)} => [{len(tokenizer.vocabs)}] {current_merge.freq}")
  elapsed = time.time() - starttime
  print(elapsed, len(tokenizer.vocabs))
  logging.info(f"finished in {elapsed} seconds")

  tokenizer.save_to_files("tokenizer.json", "merges.txt")

# %%
from cs336_basics.tokenizer import Tokenizer
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files("tokenizer.json", "merges.txt", special_tokens)
print(max(tokenizer.vocabs.values(), key=len)) # b' accomplishment'

# %%
import torch
from cs336_basics.optimizer import SGD

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
  opt.zero_grad() # Reset the gradients for all learnable parameters.
  loss = (weights**2).mean() # Compute a scalar loss value.
  print(loss.cpu().item())
  loss.backward() # Run backward pass, which computes gradients.
  opt.step() # Run optimizer step.

# %%
