# %%
from cs336_basics.tokenizer import Tokenizer, train_bpe
from pathlib import Path
import logging
import time

def mkpath(s: str) -> Path:
  d = Path(s)
  d.mkdir(parents=True, exist_ok=True)
  return d

logs_dir = mkpath("logs")
logging.basicConfig(filename=logs_dir/f"train-{int(time.time())}.log", level=logging.DEBUG)
fixture = Path(__file__).parent.parent / "experiment/fixtures/TinyStoriesV2-GPT4-train.txt"
out_dir = mkpath("out")
name = fixture.stem
vocab_size = 10000
tokenizer_filename = f"tokenizer.{name}.{vocab_size}.json"
merges_filename = f"merges.{name}.{vocab_size}.txt"

# tinystories_sample_5M: 23s
if not (out_dir/tokenizer_filename).exists():
  tokenizer = train_bpe(fixture, vocab_size, ["<|endoftext|>"])
  tokenizer.save_to_files(out_dir/tokenizer_filename, out_dir/merges_filename)
else:
  tokenizer = Tokenizer.from_files(out_dir/tokenizer_filename, out_dir/merges_filename)

# %%
import numpy as np
idx_filename = f"{name}.idx.npz"
tokenizer = Tokenizer.from_files(out_dir/tokenizer_filename, out_dir/merges_filename)

# out/tinystories_sample_5M.npz: 10s
# out/TinyStoriesV2-GPT4-train.idx.npz: 7m59s
if not (out_dir/idx_filename).exists():
  result = []
  with open(fixture, "r") as f:
    result.extend(tokenizer.encode_iterable(f))
  result = np.array(result)
  np.savez(out_dir/idx_filename, result, allow_pickle=False)
idx = np.load(out_dir/idx_filename)['arr_0'] # type: np.ndarray

# %%
from cs336_basics.config import Config
from cs336_basics.training import RandomTokenDataLoader

config = Config(
  vocab_size=tokenizer.vocab_size,
  batch_size=32,
  context_length=256,
  d_model=512,
  d_ff=1344,
  num_heads=16,
  num_layers=4,
  _tokens=327680000,
)

cp_dir = mkpath("checkpoints/current")
config_save_filename = cp_dir/f"_config.json"
if config_save_filename.exists():
  saved_config = Config.load(config_save_filename)
  if saved_config != config:
    print("use saved config")
    logging.warning(f"config mismatch: saved={saved_config}, current={config}")
    config = saved_config
else:
  config.save(config_save_filename)
config.to_dict()

# %%
device = "cuda"
dataset = RandomTokenDataLoader(idx, batch_size=config.batch_size, context_length=config.context_length, device=device)
a, optimizer = config.create_llm(device=device)

# %%
from cs336_basics.training import _load_checkpoint
checkpoint_filenames = [i.name for i in cp_dir.iterdir() if i.name.startswith(f"a.{name}.") and i.name.endswith(".pt")]
def get_int(i: str):
  try:
    return int(i.split(".")[-2])
  except:
    return -1
start_iteration = 0
if checkpoint_filenames:
  last_checkpoint_filenames = max(checkpoint_filenames, key=get_int)
  start_iteration = _load_checkpoint(cp_dir/last_checkpoint_filenames, a, optimizer)["iteration"]
start_iteration

# %%
import torch
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table
x = torch.zeros((config.batch_size, config.context_length), dtype=torch.int)
flops = FlopCountAnalysis(a, x)
act = ActivationCountAnalysis(a, x)
print(flop_count_table(flops, activations=act))

if torch.cuda.is_available():
  torch.cuda.empty_cache()

# %%
import math
import wandb
from cs336_basics.training import _save_checkpoint
from cs336_basics.optimizer import _gradient_clipping
from cs336_basics.modules import LLM, _cross_entory
epoch = config.epochs
run_id = None
# run_id = "1h45b6ce"
assert (start_iteration == 0) == (run_id is None)
resume = "must" if run_id else "allow"

# torch.cuda.memory._record_memory_history()

with wandb.init(project="llm-assignment1", id=run_id, resume=resume, config=config.to_dict()) as run:
  for i in range(start_iteration, epoch):
    optimizer.zero_grad()
    x, y = dataset[i]
    y_hat = a.forward(x)
    loss = _cross_entory(y_hat, y).mean()

    if math.isnan(loss.item()):
      logging.info(f"save epoch {i}")
      _save_checkpoint(a, optimizer, i, cp_dir/f"a.{name}.{i}.pt")
      logging.error("loss is nan")
      break

    if i % 10 == 0:
      logging.debug(f"allocated {torch.cuda.memory_allocated() / 2**30:.3}, cached: {torch.cuda.memory_reserved() / 2**30:.3}")

    loss.backward()
    if config.gradient_clipping is not None:
      _gradient_clipping(a.parameters(), config.gradient_clipping)
    optimizer.step()
    logging.info(f"epoch {i}: loss={loss.item()}")
    run.log({'loss': loss})

    if i % 100 == 0:
      logging.info(f"save epoch {i}")
      _save_checkpoint(a, optimizer, i, cp_dir/f"a.{name}.{i}.pt")

# torch.cuda.memory._dump_snapshot("my_snapshot2.pickle")
# torch.cuda.memory._record_memory_history(None)

# %%
from cs336_basics.training import _save_checkpoint
_save_checkpoint(a, optimizer, epoch, out=out_dir/f"a.{name}.{epoch}.pt")
# %%
config.save(out_dir/f"a.config.{name}.{vocab_size}.json")

# %%
import torch
if torch.cuda.is_available():
  torch.cuda.empty_cache()

# %%
from cs336_basics.config import Config
from cs336_basics.training import _load_checkpoint
device = "cpu"
config = Config.load(out_dir/f"a.config.{name}.{vocab_size}.json")
a, optimizer = config.create_llm(device=device)
_load_checkpoint(out_dir/f"a.{name}.{config.epochs}.pt", a, optimizer, device=device)

# %%\
import torch
from cs336_basics.inference import gen_text

output = gen_text(
  prefix_text="Once upon a time",
  n=200,
  llm=a,
  tokenizer=tokenizer,
  device=device
)
print(output)

# %%
