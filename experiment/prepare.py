# %%
import requests
import logging, time
from pathlib import Path

Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f"logs/prepare-{int(time.time())}.log")
fixture_folder = Path(__file__).parent / "fixtures"

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
