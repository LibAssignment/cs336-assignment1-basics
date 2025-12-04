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

# %%
idx = np.load(out_dir/idx_filename)['arr_0'] # type: np.ndarray

# %%
from cs336_basics.training import RandomTokenDataLoader
from dataclasses import dataclass, asdict

@dataclass
class Config:
  vocab_size: int
  batch_size = 32
  context_length = 256
  d_model = 512
  d_ff = 1344 # d_model * 2.625
  num_heads = 16
  theta = 10000
  num_layers = 4
  tokens = 327680000

  @property
  def epochs(self):
    return self.tokens // (self.batch_size * self.context_length)

device = "cuda"

config = Config(vocab_size = tokenizer.vocab_size)

dataset = RandomTokenDataLoader(idx, batch_size=config.batch_size, context_length=config.context_length, device=device)

# %%
from cs336_basics.modules import LLM, _cross_entory
from cs336_basics.optimizer import AdamW
a = LLM(
  vocab_size=config.vocab_size,
  num_layers=config.num_layers,
  context_length=config.context_length,
  d_model=config.d_model,
  d_ff=config.d_ff,
  num_heads=config.num_heads,
  theta=config.theta,
  device=device
)
optimizer = AdamW(a.parameters())

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
import wandb
from cs336_basics.training import _save_checkpoint
import math
epoch = config.epochs
cp_dir = mkpath("checkpoints")
# torch.cuda.memory._record_memory_history()
with wandb.init("clouds56", "llm-assignment1", config={
  **asdict(config)
}) as run:
  for i in range(epoch):
    x, y = dataset[i]
    y_hat = a.forward(x)
    loss = _cross_entory(y_hat, y).mean()

    if math.isnan(loss.item()):
      logging.info(f"save epoch {i}")
      _save_checkpoint(a, optimizer, i, cp_dir/f"a.{name}.{i}.pt")
      logging.error("loss is nan")
      break

    if i % 100 == 0:
      logging.debug(f"allocated {torch.cuda.memory_allocated() / 2**30:.3}, cached: {torch.cuda.memory_reserved() / 2**30:.3}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    run.log({'loss': loss})
    logging.info(f"epoch {i}: loss={loss.item()}")

    if i % 1 == 0:
      logging.info(f"save epoch {i}")
      _save_checkpoint(a, optimizer, i, cp_dir/f"a.{name}.{i}.pt")

# torch.cuda.memory._dump_snapshot("my_snapshot2.pickle")

# %%
from cs336_basics.training import _save_checkpoint
_save_checkpoint(a, optimizer, epoch, out=out_dir/f"a.{name}.{epoch}.pt")

# %%
if torch.cuda.is_available():
  torch.cuda.empty_cache()

# %%
from cs336_basics.training import _load_checkpoint
_load_checkpoint("a.pt", a, optimizer)

# %%
import torch
inputs = tokenizer.encode("I")

def _choice(prob: torch.Tensor) -> int:
  p = np.arange(prob.size(-1))
  return np.random.choice(p, p=prob.cpu().detach().numpy()).item()

for i in range(100):
  x = torch.ones(1024, dtype=torch.int)
  for k, v in enumerate(inputs):
    x[k] = v
  y_pred = a.forward(x.to(device=device), prob=True)
  # next_i = y_pred[-1].argmax().item()
  next_i = _choice(y_pred[-1])
  assert isinstance(next_i, int)
  logging.info(f"pred {i} => {next_i}")
  inputs.append(next_i)
tokenizer.decode(inputs)

# %%
