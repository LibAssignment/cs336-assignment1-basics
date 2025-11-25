# %%
from cs336_basics.tokenizer import Tokenizer

special_tokens = ["<|endoftext|>"]
input_path = '../tests/fixtures/corpus.en'
# input_path = '../tests/fixtures/tinystories_sample_5M.txt'
# input_path = "./chinese.txt"
vocab_size = 500

tokenizer = Tokenizer.load_file(input_path, special_tokens)
while len(tokenizer.vocabs) < vocab_size:
  current_merge = tokenizer.step()
  if current_merge is None:
    break
  print(tokenizer.display_tuple(current_merge.tp), current_merge.freq)
tokenizer.vocabs

# %%
tokenizer.display_tuples()
# %%
