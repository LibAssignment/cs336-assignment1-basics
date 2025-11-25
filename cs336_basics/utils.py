import os
from typing import BinaryIO
import regex
from collections import Counter
import logging

def find_chunk_boundaries(
  file: BinaryIO,
  desired_num_chunks: int,
  split_special_token: bytes,
) -> list[int]:
  """
  Chunk the file into parts that can be counted independently.
  May return fewer chunks if the boundaries end up overlapping.
  """
  assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

  # Get total file size in bytes
  file.seek(0, os.SEEK_END)
  file_size = file.tell()
  file.seek(0)

  chunk_size = file_size // desired_num_chunks

  # Initial guesses for chunk boundary locations, uniformly spaced
  # Chunks start on previous index, don't include last index
  chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
  chunk_boundaries[-1] = file_size

  mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

  for bi in range(1, len(chunk_boundaries) - 1):
    initial_position = chunk_boundaries[bi]
    file.seek(initial_position)  # Start at boundary guess
    while True:
      mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

      # If EOF, this boundary should be at the end of the file
      if mini_chunk == b"":
        chunk_boundaries[bi] = file_size
        break

      # Find the special token in the mini chunk
      found_at = mini_chunk.find(split_special_token)
      if found_at != -1:
        chunk_boundaries[bi] = initial_position + found_at
        break
      initial_position += mini_chunk_size

  # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
  return sorted(set(chunk_boundaries))

def chunks_iter(
  file: BinaryIO,
  desired_num_chunks: int,
  split_special_token: bytes,
):
  boundaries = find_chunk_boundaries(file, desired_num_chunks, split_special_token)

  for (start, end) in zip(boundaries, boundaries[1:]):
    file.seek(start)
    data = file.read(end-start).removeprefix(split_special_token)
    yield data.decode('utf-8', errors="ignore")

# TODO: check the latest https://github.com/openai/tiktoken/blame/main/tiktoken_ext/openai_public.py
# r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
def chunk_to_pretokenizer(chunk: str, pat: regex.Pattern[str] | str = PAT, re_special_token: regex.Pattern[str] | str | None = None):
  if re_special_token is None:
    return Counter(i[0] for i in regex.finditer(pat, chunk))
  words = Counter[str]()
  for c in regex.split(re_special_token, chunk):
    words.update(Counter(i[0] for i in regex.finditer(pat, c)))
  return words

def get_words(
  file: BinaryIO,
  desired_num_chunks: int,
  split_special_token: bytes,
  re_special_tokens: regex.Pattern[str] | str,
) -> Counter[str]:
  final_words = Counter[str]()
  for chunk in chunks_iter(file, desired_num_chunks=desired_num_chunks, split_special_token=split_special_token):
    final_words.update(chunk_to_pretokenizer(chunk, re_special_token=re_special_tokens))
  return final_words


from multiprocessing import Pool
from dataclasses import dataclass
@dataclass
class GetWordsParam:
  idx: int
  filename: str | os.PathLike
  start: int
  end: int
  remove_prefix: bytes
  pat: regex.Pattern[str] | str = PAT
  re_special_token: regex.Pattern[str] | str | None = None


def _get_words_parallel_run(param: GetWordsParam):
  with open(param.filename, 'rb') as f:
    f.seek(param.start)
    data = f.read(param.end - param.start).removeprefix(param.remove_prefix)
    chunk = data.decode('utf-8', errors='ignore')
  result = chunk_to_pretokenizer(chunk, param.pat, param.re_special_token)
  logging.debug(f"processing {param.idx}: {param.start}-{param.end} => {len(result)}")
  return result


def get_words_parallel(
    filename: str | os.PathLike,
    desired_num_chunks: int,
    split_special_token: bytes,
    re_special_tokens: regex.Pattern[str] | str,
    parallel_count = 8,
) -> Counter[str]:
  with open(filename, 'rb') as file:
    boundaries = find_chunk_boundaries(file, desired_num_chunks, split_special_token)

  logging.debug(f"split boundaries: {len(boundaries)} {boundaries[-1]}")

  with Pool(parallel_count) as p:
    words = p.map(_get_words_parallel_run, [GetWordsParam(idx=i, filename=filename, start=a, end=b, remove_prefix=split_special_token, re_special_token=re_special_tokens) for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))])

  final_words = Counter[str]()
  for w in words:
    final_words.update(w)

  logging.info(f"proceed {filename} => {len(final_words)}")

  return final_words
