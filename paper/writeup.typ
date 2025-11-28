= BPE Tokenizer

== Problem (unicode1): Understanding Unicode
=== a) What Unicode character does `chr(0)` return?
It would return `'\x00'`.

=== b) How does this character's string representation (`__repr__()`) differ from its printed representation?
The repr of `chr(0)` would be `'\\x00'`, while `print(chr(0))` would produce no visible output, as it is a non-printable character.

=== c) What happens when this character occurs in text?
When this character occurs in text, it is visibly represented as a space.

== Problem (unicode2): Unicode Encodings
=== a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32?
The raw bytes of UTF-8 are more compact, as it uses a variable-length encoding scheme that is more efficient for common characters. UTF-8 is also backward compatible with ASCII, making it easier to handle in many systems and applications.

=== b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect?
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```
`decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))` would throw `UnicodeDecodeError`

=== c) Give a two byte sequence that does not decode to any Unicode character(s).
`b'\xff\x00'` and `b'\xff'`, since according to utf-8, any byte starting with `0xff` is invalid.

== Problem (train_bpe): BPE Tokenizer Training
Done, see: `file:cs336_basics/tokenizer.py`

== Problem (train_bpe_tinystories): BPE Training on TinyStories
=== a) How many hours and memory did training take? What is the longest token in the vocabulary? Does it make sense?
1. Training took around 160 seconds and ~1GB of memory.
2. The longest token is `b' accomplishment'`
3. Make sense, since the tokenizer would try to merge frequent pairs, and this word might appear frequently in the dataset.

=== b) Profile your code. What part of the tokenizer training process takes the most time?
```python
def step(self):
  ...
  # TODO: bottleneck here
  current_merge = max(self.pre_merges.values())
  ...
```

== Problem (train_bpe_expts_owt): BPE Training on OpenWebText
TODO

== Problem (tokenizer): Implementing the tokenizer
Done, see: `file:cs336_basics/tokenizer.py`

NOTE: there's performance issue with the current implementation.

== Problem (tokenizer_experiments): Experiments with tokenizers

= Transformer Language Model Architecture
=== Problem (linear): Implementing the linear module
```python
einsum(x, w, "... d_in, d_out d_in -> ... d_out")
```
=== Problem (embedding): Implement the embedding module
```python
w[x]
```
=== Problem (rmsnorm): Root Mean Square Layer Normalization
  $ "RMSNorm"(a_i) &= a_i / "RMS"(a) g_i \
    "RMS"(a) &= sqrt(1/d sum_1^d a_i^2 + epsilon) $
=== Problem (positionwise_feedforward): Implement the position-wise feed-forward network
  $ "ReLU"(x) &= max(0, x) \
    "SiLU"(x) &= x dot sigma(x) = x / (1+e^(-x)) \
    "GLU"(x; W, V, sigma) &= sigma(W x) dot.o (V x) \
    "FFN"(x; W_1, W_2, V, sigma) &= W_2 dot "GLU"(x; W_1, V, sigma) $
=== Problem (rope): Implement RoPE
  $ R_(i,k) &= mat(cos theta_(i,k), -sin theta_(i,k);
                   sin theta_(i,k),  cos theta_(i,k);) \
    theta_(i,k) &= i / Theta^((2k-2)/d) $
Notes:
1. $i$ in $R_(i,k)$ starts from 0, not 1.
2. Be careful with the order of index in the RoPE matrix multiplication.
3.
=== Problem (softmax): Implement softmax
Notes:
1. `x - x_max` is used to prevent overflow.
2. `x_max = torch.max(x, dim=self.dim, keepdim=True).values.detach()`
=== Problem (scaled_dot_product_attention): Implement scaled dot-product attention
Notes:
1. `1/sqrt(d_k)`
2. `x.masked_fill(~mask, -torch.inf)`
=== Problem (multihead_self_attention): Implement causal multi-head self-attention
Notes:
1. set `mask` if non-present
