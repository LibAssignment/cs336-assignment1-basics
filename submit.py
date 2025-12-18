# %%
from pathlib import Path

folder = Path(__file__).parent

# %%
import git
repo = git.Repo(folder)
repo.head

# %%
repo.head.commit.hexsha

# %%
import tempfile
tmp_dir = Path(tempfile.mkdtemp(prefix="srun-"))

# %%
shasum = repo.head.commit.hexsha
code_filename = f"code-{shasum}.zip"
with open(tmp_dir / code_filename, "wb") as f:
  repo.archive(f, shasum)
print(f"code saved to {tmp_dir / code_filename}")

# %%
data_dir = folder / "data"
