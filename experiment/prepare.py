# %%
import requests
import logging, time
from pathlib import Path

Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f"logs/prepare-{int(time.time())}.log")
fixture_folder = Path(__file__).parent.parent / "data/fixtures"

fixture_url = [
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",

  "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz",
  "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz",
]

def _download(url: str, filename: Path, chunk_size: int = 1024 * 1024):
  tmp = filename.with_name(filename.name + ".tmp")
  tmp.parent.mkdir(parents=True, exist_ok=True)

  existing = tmp.stat().st_size if tmp.exists() else 0
  headers = {}
  if existing:
    headers["Range"] = f"bytes={existing}-"

  resp = requests.get(url, headers=headers, stream=True, timeout=30)
  resp.raise_for_status()

  # If server ignored Range and returned full file, restart
  if resp.status_code == 200 and existing:
    tmp.unlink()
    existing = 0

  total = None
  if resp.status_code == 206:
    cr = resp.headers.get("Content-Range")
    if cr and "/" in cr:
      try:
        total = int(cr.split("/")[-1])
      except Exception:
        total = None
  elif "Content-Length" in resp.headers:
    try:
      total = int(resp.headers["Content-Length"]) + existing
    except Exception:
      total = None

  mode = "ab" if existing else "wb"
  with tmp.open(mode) as f:
    for chunk in resp.iter_content(chunk_size=chunk_size):
      if not chunk:
        continue
      f.write(chunk)

  try:
    if total is None or tmp.stat().st_size == total:
      tmp.replace(filename)
    else:
      # incomplete: leave .tmp so future runs can resume
      pass
  except Exception:
    # on any rename/finalize error leave .tmp for resuming later
    pass
  pass

for i in fixture_url:
  filename = fixture_folder / i.split('/')[-1]
  if filename.exists():
    continue
  _download(i, filename)

# %%
from pathlib import Path
fixture_folder = Path(__file__).parent.parent / "tests/fixtures"

# %%
from unitoken import PreTokenizer, BpeTrainer
from pathlib import Path
tokenizer_dir = Path(__file__).parent.parent / "data/tokens"

def create_tokenizer(name: str, input_dir: Path = fixture_folder, output_dir: Path = tokenizer_dir, special_tokens: list[str] | None = None, vocab_size: int = 10000):
  input_file = input_dir / f"{name}.txt"
  if special_tokens is None:
    special_tokens = ["<|endoftext|>"]
  words = PreTokenizer(special_tokens, None).get_words_from_file(input_file, 1024)
  trainer = BpeTrainer(special_tokens)
  trainer.add_words(words)
  trainer.train(vocab_size)
  output_dir.mkdir(parents=True, exist_ok=True)
  trainer.save(name, outdir=output_dir)

create_tokenizer("tinystories_sample_5M", vocab_size=1000)
# %%
from unitoken import BpeEncoder
import numpy as np
def create_idx_file(name: str, input_dir: Path = fixture_folder, output_dir: Path = tokenizer_dir, chunks: int = 1024):
  input_file = input_dir / f"{name}.txt"
  encoder = BpeEncoder.load(name, input_dir=output_dir)
  idxs = encoder.encode_file(input_file, chunks)
  np.save(output_dir/f"idx.{name}.npy", idxs)
  if "train" in name:
    name_valid = name.replace("train", "valid")
    valid_file = input_dir / name_valid
    if valid_file.exists():
      valid_idxs = encoder.encode_file(valid_file, chunks)
      np.save(output_dir/f"idx.{name_valid}.npy", valid_idxs)

create_idx_file("tinystories_sample_5M")
# %%
