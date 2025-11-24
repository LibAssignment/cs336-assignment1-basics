# %%
from cs336_basics.tokenizer import Tokenizer

special_token = "<|endoftext|>"
special_tokens = [special_token, "<||>"]
input_path = '../tests/fixtures/tinystories_sample_5M.txt'
vocab_size = 1000

tokenizer = Tokenizer.load_file(input_path, special_tokens)
while len(tokenizer.vocabs) < vocab_size:
  current_merge = tokenizer.step()
  print(tokenizer.display_tuple(current_merge))
tokenizer.vocabs

# %%
