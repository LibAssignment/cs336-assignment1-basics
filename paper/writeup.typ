== Problem (unicode1)
=== a) What Unicode character does `chr(0)` return?
It would return `'\x00'`.

=== b) How does this character's string representation (`__repr__()`) differ from its printed representation?
The repr of `chr(0)` would be `'\\x00'`, while `print(chr(0))` would produce no visible output, as it is a non-printable character.

=== c) What happens when this character occurs in text?
When this character occurs in text, it is visibly represented as a space.
