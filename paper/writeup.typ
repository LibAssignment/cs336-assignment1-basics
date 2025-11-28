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
3. $Theta$ should be inverse, or use $-(2k-2)/d$ in `pow`
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
1. $W_o$ applied to concatenated output
2. set `mask` if non-present
3. allow `rope` which applied on $Q$ and $K$
=== Problem (transformer_block): Implement the Transformer block
=== Problem (transformer_lm): Implementing the Transformer LM
Notes:
1. NO pass into `softmax` layer at the end, see also #link("https://github.com/stanford-cs336/assignment1-basics/issues/37")[\#37]
=== Problem (transformer_accounting): Transformer LM resource accounting
a) Suppose we constructed our model using this configuration.
#let vocab_size = 50257
#let d_model = 1600
#let d_ff = 6400
#let num_heads = 25
#let num_layers = 48
#let context_length = 1024

$"vocab_size"&: #vocab_size \
"context_length"&: #context_length \
"num_layers"&: #num_layers \
"d_model"&: #d_model \
"num_heads"&: #num_heads \
"d_ff"&: #d_ff \
$
#let calc_M(b, k, i) = calc.round(b / calc.pow(1000, k) * calc.pow(10, i)) / calc.pow(10, i)
#let calc_MiB(b, k, i) = calc.round(b / calc.pow(1024, k) * calc.pow(10, i)) / calc.pow(10, i)
- How many trainable parameters would our model have?
  - embedding: #let embedding_param = vocab_size * d_model;  `vocab_size * d_model =` $#vocab_size times #d_model = #embedding_param approx 80Mu$
  - transformer block:
    #let layer_attn_params = 4 * d_model * d_model
    #let layer_ffn_params = 3 * d_model * d_ff
    #let layer_rmsnorm_params = 2 * d_model
    #let layer_params = layer_attn_params + layer_ffn_params + layer_rmsnorm_params
    $~ #(calc_M(layer_params, 2, 0))Mu$
    - attention: `4 * d_model * d_model =` $4 times #d_model times #d_model approx #calc_M(layer_attn_params, 2, 0)Mu$
    - ffn: `3 * d_model * d_ff =` $3 times #d_model times #d_ff approx #calc_M(layer_ffn_params, 2, 0)Mu$
    - rmsnorm: `2 * d_model =` $2 times #d_model approx #calc_M(layer_rmsnorm_params, 2, 3)Mu$
  - head: #let head_params = d_model + d_model * vocab_size; $~#calc_M(embedding_param, 2, 0)Mu$
    - norm: `d_model =` $#d_model approx #calc_M(d_model, 2, 3)Mu$
    - embedding proj: `d_model * vocab_size =` $#d_model times #vocab_size approx #calc_M(embedding_param, 2, 0)Mu$
  - total: `embedding + num_layers * layer_params + head` #let total_params = embedding_param + num_layers * layer_params + head_params; $approx #calc_M(total_params, 3, 3)"B"$
- Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?
  - $#calc_MiB(total_params * 4, 3, 3) "GiB"$ memory.

b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. Assume that our input
sequence has context_length tokens.
#let attn_proj_calc = 2*d_model*d_model*context_length
#let ffn_linear_calc = 2*d_model*d_ff*context_length
#let attn_total_calc = 4*attn_proj_calc + 3*ffn_linear_calc
#let head_embedding_calc = 2*d_model*vocab_size*context_length
#let total_calc = attn_total_calc * num_layers + head_embedding_calc
#table(
  columns: (auto, auto, auto, auto),
  table.header("name", "matrix", "multiple", "value (TFLOPs)"),
  "attention_q_proj", [$W_q in RR^(h d_k times d_"model"), x in RR^("seq"times d_"model")$], [`2*d_model^2*seq_len`], [#calc_M(attn_proj_calc, 4, 3)],
  "attention_k_proj", [$W_k in RR^(h d_k times d_"model"), x in RR^("seq"times d_"model")$], [`2*d_model^2*seq_len`], [#calc_M(attn_proj_calc, 4, 3)],
  "attention_v_proj", [$W_v in RR^(h d_v times d_"model"), x in RR^("seq"times d_"model")$], [`2*d_model^2*seq_len`], [#calc_M(attn_proj_calc, 4, 3)],
  "attention_o_proj", [$W_o in RR^(d_"model" times h d_v), x in RR^("seq"times h d_v)$], [`2*d_model^2*seq_len`], [#calc_M(attn_proj_calc, 4, 3)],
  "ffn_linear1", [$W_1 in RR^(d_"ff" times d_"model"), x in RR^("seq"times d_"model")$], [`2*d_model*d_ff*seq_len`], [#calc_M(ffn_linear_calc, 4, 3)],
  "ffn_linear_gate", [$W_3 in RR^(d_"ff" times d_"model"), x in RR^("seq"times d_"model")$], [`2*d_model*d_ff*seq_len`], [#calc_M(ffn_linear_calc, 4, 3)],
  "ffn_linear2", [$W_3 in RR^(d_"model" times d_"ff"), x in RR^("seq"times d_"ff")$], [`2*d_model*d_ff*seq_len`], [#calc_M(ffn_linear_calc, 4, 3)],
  "attn_total", [], [$4 times #calc_M(attn_proj_calc,4,3) + 3times#calc_M(ffn_linear_calc, 4, 3)$], [#calc_M(attn_total_calc, 4, 3)],
  "head_embedding", [$W_e in RR^(d_"model" times"vocab"), x in RR^("seq"times d_"model")$], [`2*d_model*vocab*seq_len`], [#calc_M(head_embedding_calc, 4, 3)],
  "total", [], [$#num_layers times #calc_M(attn_total_calc,4,3) + #calc_M(head_embedding_calc, 4, 3)$], [#calc_M(total_calc, 4, 3)]
)

c) Based on your analysis above, which parts of the model require the most FLOPs?
attention and feed-forward layers, since they repeat for 48 layers.
