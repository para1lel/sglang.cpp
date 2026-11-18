# Tokenizer Unicode data

The publisher is Qwen. The external `Qwen/Qwen3.8-27B-FP8` checkpoint supplies
`tokenizer.json` from revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`. The file is
12,809,320 bytes and has SHA-256
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
The native tokenizer verifies this digest before loading.
[Pinned tokenizer file][tokenizer] · [Checkpoint Apache-2.0 license][checkpoint-license]

See the [checkpoint authentication](../../docs/research/checkpoint.md#checkpoint-evidence)
and the [text completion verification procedure](../../docs/text-serving.md#verification).

The tokenizer uses Unicode 16.0 general categories, `White_Space` and simple
case folds in its pretokenizer. It uses Unicode 9.0 combining classes,
canonical decompositions and composition exclusions for NFC. These versions
match the tokenizer for the pinned Qwen3.8-27B-FP8 checkpoint. The generator
reads `UnicodeData.txt`, `PropList.txt`, `CaseFolding.txt` and
`DerivedNormalizationProps.txt` from the Unicode Character Database.

- [Unicode 16.0.0 UCD archive](https://www.unicode.org/Public/zipped/16.0.0/UCD.zip)
- [Unicode 9.0.0 UCD archive](https://www.unicode.org/Public/zipped/9.0.0/UCD.zip)
- [Unicode License v3](LICENSE)

Save the original archives as `UCD-16.0.0.zip` and `UCD-9.0.0.zip` in one
directory. The generator verifies their SHA-256 hashes before reading them.
From the project root, regenerate the checked-in tables with the Python
standard library.

```sh
python -B tools/unicode/generate.py /path/to/ucd src/tokenizer/unicode
```

The C++ build compiles the checked-in tables directly. The source archives and
Python are required only for regeneration. Keep the Unicode license with the
tables when distributing them.

[tokenizer]: https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/c3a462cc3d38cf351f3b26cbd763469eeb74ef81/tokenizer.json
[checkpoint-license]: https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/c3a462cc3d38cf351f3b26cbd763469eeb74ef81/LICENSE
