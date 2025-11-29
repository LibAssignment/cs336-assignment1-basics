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
== Problem (transformer_accounting): Transformer LM resource accounting
=== a) Suppose we constructed our model using this configuration.
#let config_xl_base = (
  vocab_size: 50257,
  d_model: 1600,
  d_ff: 6400,
  num_heads: 25,
  num_layers: 48,
  context_length: 1024
)
#let calc_params(config) = {
  let params = (
    embedding: config.vocab_size * config.d_model,
    layer: (
      attn: 4 * config.d_model * config.d_model,
      ffn: 3 * config.d_model * config.d_ff,
      rmsnorm: 2 * config.d_model,
    ),
    head_embedding: (
      norm: config.d_model,
      proj: config.d_model * config.vocab_size,
    ),
  )
  params.layer.total = params.layer.attn + params.layer.ffn + params.layer.rmsnorm
  params.head_embedding.total = params.head_embedding.norm + params.head_embedding.proj
  params.total = params.embedding + params.layer.total * config.num_layers + params.head_embedding.total
  params.total_bytes = params.total * 4  // assuming float32
  (
    ..config,
    params: params,
  )
}
#let config_xl = calc_params(config_xl_base)

$"vocab_size"&: #config_xl.vocab_size \
"context_length"&: #config_xl.context_length \
"num_layers"&: #config_xl.num_layers \
"d_model"&: #config_xl.d_model \
"num_heads"&: #config_xl.num_heads \
"d_ff"&: #config_xl.d_ff \
$
#let calc_M(b, k: 2, i: 3) = calc.round(b / calc.pow(1000, k) * calc.pow(10, i)) / calc.pow(10, i)
#let calc_MiB(b, k: 2, i: 3) = calc.round(b / calc.pow(1024, k) * calc.pow(10, i)) / calc.pow(10, i)
- How many trainable parameters would our model have?
  - embedding:  `vocab_size * d_model =` $#config_xl.vocab_size times #config_xl.d_model = #config_xl.params.embedding approx 80Mu$
  - transformer block:
    $~ #(calc_M(config_xl.params.layer.total))Mu$
    - attention: `4 * d_model * d_model =` $4 times #config_xl.d_model times #config_xl.d_model approx #calc_M(config_xl.params.layer.attn)Mu$
    - ffn: `3 * d_model * d_ff =` $3 times #config_xl.d_model times #config_xl.d_ff approx #calc_M(config_xl.params.layer.ffn)Mu$
    - rmsnorm: `2 * d_model =` $2 times #config_xl.d_model approx #calc_M(config_xl.params.layer.rmsnorm)Mu$
  - head: $~#calc_M(config_xl.params.embedding)Mu$
    - norm: `d_model =` $#config_xl.d_model approx #calc_M(config_xl.d_model)Mu$
    - embedding proj: `d_model * vocab_size =` $#config_xl.d_model times #config_xl.vocab_size approx #calc_M(config_xl.params.embedding, i: 0)Mu$
  - total: `embedding + num_layers * layer_params + head` $approx #calc_M(config_xl.params.total, k: 3)"B"$
- Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?
  - $#calc_MiB(config_xl.params.total_bytes, k: 3) "GiB"$ memory.

=== b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. Assume that our input sequence has context_length tokens.

#let calc_tflops(config) = {
  let d_model = config.d_model
  let context_length = config.context_length
  let vocab_size = config.vocab_size
  let d_ff = config.d_ff
  let tflops = (
    attn: (
      proj: 2*d_model*d_model*context_length,
      ffn_linear: 2*d_model*d_ff*context_length,
    ),
    head_embedding: 2*d_model*vocab_size*context_length,
  )
  tflops.attn.total = 5*tflops.attn.proj + 3*tflops.attn.ffn_linear
  tflops.total = tflops.attn.total * config.num_layers + tflops.head_embedding
  tflops.total_tflops = calc.round(tflops.total / calc.pow(10, 9)) / 1000 // convert to TFLOPs
  (
    ..config,
    tflops: tflops,
  )
}
#let config_xl = calc_tflops(config_xl)

// #let vocab_size = 50257
// #let d_model = 1600
// #let d_ff = 6400
// #let num_heads = 25
// #let num_layers = 48
// #let context_length = 1024

#table(
  columns: (auto, auto, auto, auto),
  table.header("name", "matrix", "multiple", "value (TFLOPs)"),
  "attn_rope", [$W_q in RR^(d_"model" times d_"model"), x in RR^(d_"model")$], [`2*d_model^2*seq_len`], [#calc_M(config_xl.tflops.attn.proj, k: 4)],
  "attn_q_proj", [$W_q in RR^(h d_k times d_"model"), x in RR^(d_"model")$], [`2*d_model^2*seq_len`], [#calc_M(config_xl.tflops.attn.proj, k: 4)],
  "attn_k_proj", [$W_k in RR^(h d_k times d_"model"), x in RR^(d_"model")$], [`2*d_model^2*seq_len`], [#calc_M(config_xl.tflops.attn.proj, k: 4)],
  "attn_v_proj", [$W_v in RR^(h d_v times d_"model"), x in RR^(d_"model")$], [`2*d_model^2*seq_len`], [#calc_M(config_xl.tflops.attn.proj, k: 4)],
  "attn_o_proj", [$W_o in RR^(d_"model" times h d_v), x in RR^(h d_v)$], [`2*d_model^2*seq_len`], [#calc_M(config_xl.tflops.attn.proj, k: 4)],
  "ffn_linear1", [$W_1 in RR^(d_"ff" times d_"model"), x in RR^(d_"model")$], [`2*d_model*d_ff*seq_len`], [#calc_M(config_xl.tflops.attn.ffn_linear, k: 4)],
  "ffn_linear_gate", [$W_3 in RR^(d_"ff" times d_"model"), x in RR^(d_"model")$], [`2*d_model*d_ff*seq_len`], [#calc_M(config_xl.tflops.attn.ffn_linear, k: 4)],
  "ffn_linear2", [$W_3 in RR^(d_"model" times d_"ff"), x in RR^(d_"ff")$], [`2*d_model*d_ff*seq_len`], [#calc_M(config_xl.tflops.attn.ffn_linear, k: 4)],
  "attn_total", [], [$5 times #calc_M(config_xl.tflops.attn.proj, k:4, i:3) + 3times#calc_M(config_xl.tflops.attn.ffn_linear, k: 4)$], [#calc_M(config_xl.tflops.attn.total, k: 4)],
  "head_embedding", [$W_e in RR^(d_"model" times"vocab"), x in RR^(d_"model")$], [`2*d_model*vocab*seq_len`], [#calc_M(config_xl.tflops.head_embedding, k: 4)],
  "total", [], [$#config_xl.num_layers times #calc_M(config_xl.tflops.attn.total, k: 4) + #calc_M(config_xl.tflops.head_embedding, k: 4)$], [#calc_M(config_xl.tflops.total, k: 4)]
)

=== c) Based on your analysis above, which parts of the model require the most FLOPs?
Attention and feed-forward layers, since they repeat for 48 layers.

=== d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads).
#let calc_config(config) = calc_tflops(calc_params(config))
#let config_small = (
  ..config_xl_base,
  d_model: 768,
  d_ff: 3072,
  num_heads: 12,
  num_layers: 12,
)
#let config_small = calc_config(config_small)

#let config_medium = (
  ..config_xl_base,
  d_model: 1024,
  d_ff: 4096,
  num_heads: 16,
  num_layers: 24,
)
#let config_medium = calc_config(config_medium)

#let config_large = (
  ..config_xl_base,
  d_model: 1280,
  d_ff: 5120,
  num_heads: 20,
  num_layers: 36,
)
#let config_large = calc_config(config_large)

Memory:
#table(
  columns: (auto, auto, auto, auto, auto),
  table.header("name", "GPT-2 small", "GPT-2 medium", "GPT-2 large", "GPT-2 XL"),
  "embedding", [#calc_MiB(config_small.params.embedding)], [#calc_MiB(config_medium.params.embedding)], [#calc_MiB(config_large.params.embedding)], [#calc_MiB(config_xl.params.embedding)],
  "transformer block", [#calc_MiB(config_small.params.layer.total)], [#calc_MiB(config_medium.params.layer.total)], [#calc_MiB(config_large.params.layer.total)], [#calc_MiB(config_xl.params.layer.total)],
  "  (attn)", [#calc_MiB(config_small.params.layer.attn)], [#calc_MiB(config_medium.params.layer.attn)], [#calc_MiB(config_large.params.layer.attn)], [#calc_MiB(config_xl.params.layer.attn)],
  "  (ffn)", [#calc_MiB(config_small.params.layer.ffn)], [#calc_MiB(config_medium.params.layer.ffn)], [#calc_MiB(config_large.params.layer.ffn)], [#calc_MiB(config_xl.params.layer.ffn)],
  "  (rmsnorm)", [#calc_MiB(config_small.params.layer.rmsnorm)], [#calc_MiB(config_medium.params.layer.rmsnorm)], [#calc_MiB(config_large.params.layer.rmsnorm)], [#calc_MiB(config_xl.params.layer.rmsnorm)],
  "head embedding", [#calc_MiB(config_small.params.head_embedding.total)], [#calc_MiB(config_medium.params.head_embedding.total)], [#calc_MiB(config_large.params.head_embedding.total)], [#calc_MiB(config_xl.params.head_embedding.total)],
  "  (norm)", [#calc_MiB(config_small.params.head_embedding.norm)], [#calc_MiB(config_medium.params.head_embedding.norm)], [#calc_MiB(config_large.params.head_embedding.norm)], [#calc_MiB(config_xl.params.head_embedding.norm)],
  "  (proj)", [#calc_MiB(config_small.params.head_embedding.proj)], [#calc_MiB(config_medium.params.head_embedding.proj)], [#calc_MiB(config_large.params.head_embedding.proj)], [#calc_MiB(config_xl.params.head_embedding.proj)],
  "total (B)", [#calc_MiB(config_small.params.total, k: 3)], [#calc_MiB(config_medium.params.total, k: 3)], [#calc_MiB(config_large.params.total, k: 3)], [#calc_MiB(config_xl.params.total, k: 3)],
  "total (GiB)", [#calc_MiB(config_small.params.total_bytes, k: 3)], [#calc_MiB(config_medium.params.total_bytes, k: 3)], [#calc_MiB(config_large.params.total_bytes, k: 3)], [#calc_MiB(config_xl.params.total_bytes, k: 3)],

)

TFLOPs:
#table(
  columns: (auto, auto, auto, auto, auto),
  table.header("name", "GPT-2 small", "GPT-2 medium", "GPT-2 large", "GPT-2 XL"),
  "transformer block", [#calc_M(config_small.tflops.attn.total, k: 4)], [#calc_M(config_medium.tflops.attn.total, k: 4)], [#calc_M(config_large.tflops.attn.total, k: 4)], [#calc_M(config_xl.tflops.attn.total, k: 4)],
  "  (rope)", [#calc_M(config_small.tflops.attn.proj, k: 4)], [#calc_M(config_medium.tflops.attn.proj, k: 4)], [#calc_M(config_large.tflops.attn.proj, k: 4)], [#calc_M(config_xl.tflops.attn.proj, k: 4)],
  "  (proj x 4)", [#calc_M(config_small.tflops.attn.proj, k: 4)], [#calc_M(config_medium.tflops.attn.proj, k: 4)], [#calc_M(config_large.tflops.attn.proj, k: 4)], [#calc_M(config_xl.tflops.attn.proj, k: 4)],
  "  (ffn x 3)", [#calc_M(config_small.tflops.attn.ffn_linear, k: 4)], [#calc_M(config_medium.tflops.attn.ffn_linear, k: 4)], [#calc_M(config_large.tflops.attn.ffn_linear, k: 4)], [#calc_M(config_xl.tflops.attn.ffn_linear, k: 4)],
  "head embedding", [#calc_M(config_small.tflops.head_embedding, k: 4)], [#calc_M(config_medium.tflops.head_embedding, k: 4)], [#calc_M(config_large.tflops.head_embedding, k: 4)], [#calc_M(config_xl.tflops.head_embedding, k: 4)],
  "total (TFLOPs)", [#calc_M(config_small.tflops.total, k: 4)], [#calc_M(config_medium.tflops.total, k: 4)], [#calc_M(config_large.tflops.total, k: 4)], [#calc_M(config_xl.tflops.total, k: 4)],
)
FFN in Transformer block contributes the most FLOPs, since it has 3 matrix multiplies per layer, and there're lots of layers.

=== e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?
#let config_xl_4096 = calc_config((
  ..config_xl_base,
  context_length: 4096,
))
#let config_xl_16384 = calc_config((
  ..config_xl_base,
  context_length: 16384,
))
#let config_xl_100k = calc_config((
  ..config_xl_base,
  context_length: 102400,
))

#table(
  columns: (auto, auto, auto, auto, auto),
  table.header("name", "1024", "4096", "16384", "100k"),
  "transformer block", [#calc_M(config_xl.tflops.attn.total, k: 4)], [#calc_M(config_xl_4096.tflops.attn.total, k: 4)], [#calc_M(config_xl_16384.tflops.attn.total, k: 4)], [#calc_M(config_xl_100k.tflops.attn.total, k: 4)],
  "  (rope)", [#calc_M(config_xl.tflops.attn.proj, k: 4)], [#calc_M(config_xl_4096.tflops.attn.proj, k: 4)], [#calc_M(config_xl_16384.tflops.attn.proj, k: 4)], [#calc_M(config_xl_100k.tflops.attn.proj, k: 4)],
  "  (proj x 4)", [#calc_M(config_xl.tflops.attn.proj, k: 4)], [#calc_M(config_xl_4096.tflops.attn.proj, k: 4)], [#calc_M(config_xl_16384.tflops.attn.proj, k: 4)], [#calc_M(config_xl_100k.tflops.attn.proj, k: 4)],
  "  (ffn x 3)", [#calc_M(config_xl.tflops.attn.ffn_linear, k: 4)], [#calc_M(config_xl_4096.tflops.attn.ffn_linear, k: 4)], [#calc_M(config_xl_16384.tflops.attn.ffn_linear, k: 4)], [#calc_M(config_xl_100k.tflops.attn.ffn_linear, k: 4)],
  "head embedding", [#calc_M(config_xl.tflops.head_embedding, k: 4)], [#calc_M(config_xl_4096.tflops.head_embedding, k: 4)], [#calc_M(config_xl_16384.tflops.head_embedding, k: 4)], [#calc_M(config_xl_100k.tflops.head_embedding, k: 4)],
  "total (TFLOPs)", [#calc_M(config_xl.tflops.total, k: 4)], [#calc_M(config_xl_4096.tflops.total, k: 4)], [#calc_M(config_xl_16384.tflops.total, k: 4)], [#calc_M(config_xl_100k.tflops.total, k: 4)],
)
It is almost linear for FLOPs of every component, since the context length is only used in the matrix multiplication of the attention and feed-forward layers. The total FLOPs for one forward pass increases linearly with the context length.

= Training a Transformer LM
=== Problem (cross_entropy): Implement Cross entropy
Notes:
1. `x.gather(dim=dim, index=target.unsqueeze(dim)).squeeze(dim)`
2. The tests requires `mean` if batch is present.
=== Problem (learning_rate_tuning): Tuning the learning rate
- for `lr=1e1` the loss decreases slowly but steadily.
- for `lr=1e2` the loss decreases to a lower value faster.
- for `lr=1e3` the loss increases and diverges.
=== Problem (adamw): Implement AdamW
Notes:
1. `step_count` should start from 1 to avoid zero division, and stored in state
2. `alpha_t` would initially `~300` times of `alpha` and decay to `alpha` gradually (in 10000 steps).
=== Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup
=== Problem (gradient_clipping): Implement gradient clipping
Note: we are dealing with `param.grad` here.

=== Problem (adamwAccounting): Resource accounting for training with AdamW

= Training loop
=== Problem (data_loading): Implement data loading
why `torch.save` with `torch.load` would not work well with `dtype == torch.int`?
