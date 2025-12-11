import torch
import numpy as np
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.modules import LLM
import logging

def _choice(prob: torch.Tensor) -> int:
  p = np.arange(prob.size(-1))
  return np.random.choice(p, p=prob.cpu().detach().numpy()).item()

def gen_text(prefix_text: str, n: int, *, llm: LLM, tokenizer: Tokenizer, device):
  inputs = tokenizer.encode(prefix_text)

  for i in range(n):
    x = torch.tensor(inputs, dtype=torch.int)
    y_pred = llm.forward(x.to(device=device), prob=True)
    # next_i = y_pred[-1].argmax().item()
    next_i = _choice(y_pred[-1])
    assert isinstance(next_i, int)
    logging.info(f"pred {i} => {next_i}")
    inputs.append(next_i)
    if tokenizer.vocabs[next_i] == "<|endoftext|>":
      break
  return tokenizer.decode(inputs)
