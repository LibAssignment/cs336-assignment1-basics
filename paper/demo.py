# %%
from cs336_basics.tokenizer import Tokenizer
import logging

special_tokens = ["<|endoftext|>"]
# input_path = '../tests/fixtures/corpus.en'
input_path = '../tests/fixtures/tinystories_sample_5M.txt'
# input_path = './fixtures/TinyStories-train.txt'
# input_path = "./chinese.txt"
vocab_size = 500

os.makedirs("logs")
logging.basicConfig(level=logging.DEBUG, filename="logs/tokenizer.log")
tokenizer = Tokenizer.load_file(input_path, special_tokens)
while len(tokenizer.vocabs) < vocab_size:
  current_merge = tokenizer.step()
  if current_merge is None:
    break
  logging.debug(f"merge {tokenizer.display_tuple(current_merge.tp)} => [{len(tokenizer.vocabs)}] {current_merge.freq}")
tokenizer.vocabs

# %%
tokenizer.display_tuples()
# %%
