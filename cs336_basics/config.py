from dataclasses import asdict, dataclass
from typing import Self
from torch.types import FileLike
import os

@dataclass
class Config:
  vocab_size: int
  batch_size: int = 32
  context_length: int = 256
  d_model: int = 512
  d_ff: int = 1344 # d_model * 2.625
  num_heads: int = 16
  theta: int = 10000
  num_layers: int = 4
  _tokens: int | None = None # 327680000

  @property
  def tokens(self):
    if self._tokens is None:
      raise ValueError("tokens not set")
    return self._tokens

  @property
  def epochs(self):
    return self.tokens // (self.batch_size * self.context_length)

  @property
  def d_ff_ratio(self):
    return self.d_ff / self.d_model

  @property
  def approx_params(self):
    """
    see writeup
    ```
    let params = (
      vocab: vocab_size * d_model * 2,
      model: d_model * d_model * (4 + 3 * d_ff_ratio) * num_layers,
      const: d_model * (2 * num_layers + 1)
    )
    params.total = params.vocab + params.model + params.const
    ```
    """
    return (
      self.vocab_size * self.d_model * 2 +
      self.d_model * self.d_model * (4 + 3 * self.d_ff_ratio) * self.num_layers +
      self.d_model * (2 * self.num_layers + 1)
    )

  @property
  def approx_activation_coeff(self):
    """
    see writeup
    ```
    let act = (
      model: d_model * context_length * ((ffn_count * d_ff_ratio + qkv) * num_layers + 1),
      seq: context_length*context_length * num_heads * (qk * num_layers),
      vocab: (vocab_size + 1) * context_length,
    )
    act.total = act.model + act.seq + act.vocab
    ```
    """
    ffn_count = 2 # w1, silu, w3, multiplies
    qkv = 2 + 3 + 2 + 1 #  rms, qkv, sum/output, w2
    qk = 2 # qk, softmax
    return (
      self.d_model * self.context_length * ((ffn_count * self.d_ff_ratio + qkv) * self.num_layers + 1) +
      self.context_length * self.context_length * self.num_heads * (qk * self.num_layers) +
      (self.vocab_size + 1) * self.context_length
    )

  @property
  def approx_memory(self):
    bytes_f32 = 4
    adamw_factor = 2
    return (self.approx_params * (2 + adamw_factor) + self.approx_activation_coeff * self.batch_size) * bytes_f32

  def to_dict(self):
    return asdict(self) | {
      "info": {
        "d_ff_ratio": self.d_ff_ratio,
        "epochs": self.epochs if self._tokens is not None else None,
        "approx_params": self.approx_params,
        "approx_activation_coeff": self.approx_activation_coeff,
        "approx_memory": self.approx_memory,
      }
    }

  def save(self, filename: FileLike):
    import json
    data = self.to_dict()
    if isinstance(filename, (os.PathLike, str)):
      with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    else:
      s = json.dumps(data)
      filename.write(s.encode('utf-8'))

  @classmethod
  def load(cls, filename: FileLike) -> Self:
    import json
    if isinstance(filename, (os.PathLike, str)):
      with open(filename, 'r') as f:
        data = json.load(f)
    else:
      s = filename.read().decode('utf-8')
      data = json.loads(s)
    data.pop("info", None)
    return cls(**data)
