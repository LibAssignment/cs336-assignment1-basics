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
