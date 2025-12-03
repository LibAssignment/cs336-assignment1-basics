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
from dataclasses import dataclass, asdict

@dataclass
class Config:
  vocab_size: int
  context_length = 1024
  batch_size = 32
  d_model = 256
  d_ff = d_model * 4
  num_heads = 4
  theta = 10000
  num_layers = 3

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

# %%
import wandb
from cs336_basics.training import _save_checkpoint
epoch = 100
with wandb.init("clouds56", "llm-assignment1", config={
  **asdict(config)
}) as run:
  for i in range(epoch):
    x, y = dataset[i]
    y_hat = a.forward(x)
    loss = _cross_entory(y_hat, y).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    run.log({'loss': loss})
    logging.info(f"epoch {i}: loss={loss.item()}")
# %%
from cs336_basics.training import _save_checkpoint
_save_checkpoint(a, optimizer, epoch, out="a.pt")

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
