# %%
from pathlib import Path

from cs336_basics.config import Config
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import _load_checkpoint

def mkpath(s: str) -> Path:
  d = Path(s)
  d.mkdir(parents=True, exist_ok=True)
  return d

cp_dir = mkpath("checkpoints/current")
out_dir = mkpath("out")
name = "TinyStoriesV2-GPT4-train"
device = "cuda"
config = Config.load(cp_dir/f"_config.json")
def get_int(s: str) -> int:
  try:
    return int(s.split(".")[-2])
  except ValueError:
    return -1
checkpoint_filenames = sorted(map(lambda x: x.name, cp_dir.glob(f"a.{name}.*.pt")), key=get_int)
checkpoint_filenames

# %%
from cs336_basics.tokenizer import Tokenizer
vocab_size = 10000
tokenizer = Tokenizer.from_files(out_dir/f"vocab.{name}.{vocab_size}.json", out_dir/f"merges.{name}.{vocab_size}.txt")

# %%
from cs336_basics.inference import gen_text
a, _ = config.create_llm("cpu")
results = {}
for i in checkpoint_filenames:
  _load_checkpoint(cp_dir/i, a, None)
  text = gen_text("Long time ago", llm=a, tokenizer=tokenizer)
  print(i, text, "\n\n")
  results[i] = text

# %%
results
# %%
