# Tokenizer Unicode data

The publisher is Qwen. The external `Qwen/Qwen3.8-27B-FP8` checkpoint supplies
`tokenizer.json` from revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`. The file is
12,809,320 bytes and has SHA-256
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
The native tokenizer verifies this digest before loading.
[Pinned tokenizer file][tokenizer] · [Checkpoint Apache-2.0 license][checkpoint-license]

See the [checkpoint authentication](../../docs/research/checkpoint.md#checkpoint-evidence)
and the [text completion verification procedure](../../docs/text-serving.md#verification).

The native build uses ICU 78.3, which ships Unicode 17.0 data. The tokenizer
limits general categories and `White_Space` to Unicode 16.0, NFC to the
Unicode 9.0 repertoire, and full case folding for decision labels to
Unicode 15.0. These limits preserve the pinned checkpoint and decision-label
behavior. The pretokenizer recognizes the Unicode 16.0 simple folds that can
form its Latin contractions.

In the managed Pixi environment, ICU headers are in `$CONDA_PREFIX/include`
and libraries are in `$CONDA_PREFIX/lib`. For a direct compile and link of
the tokenizer sources, add `-isystem "$CONDA_PREFIX/include"` to compilation
and put `-L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" -licuuc -licudata`
after the object files or tokenizer archive on the link command. The xmake
tokenizer target propagates the link dependency to its consumers.

- [ICU 78 release and license](https://github.com/unicode-org/icu/tree/release-78.3)
- [Unicode Character Database archives](https://www.unicode.org/ucd/)
- [Unicode License v3](LICENSE)

[tokenizer]: https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/c3a462cc3d38cf351f3b26cbd763469eeb74ef81/tokenizer.json
[checkpoint-license]: https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/c3a462cc3d38cf351f3b26cbd763469eeb74ef81/LICENSE
