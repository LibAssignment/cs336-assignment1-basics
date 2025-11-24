# %%
from cs336_basics.tokenizer import Tokenizer

special_tokens = ["<|endoftext|>"]
input_path = '../tests/fixtures/corpus.en'
vocab_size = 500

tokenizer = Tokenizer.load_file(input_path, special_tokens)
while len(tokenizer.vocabs) < vocab_size:
  current_merge = tokenizer.step()
  if current_merge is None:
    break
  print(tokenizer.display_tuple(current_merge[0]), current_merge[1])
tokenizer.vocabs

# %%
