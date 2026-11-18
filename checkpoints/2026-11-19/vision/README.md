# Offline vision reconstruction

The `vision` series adds the C++ preprocessor, its Python oracle, and the standalone CUDA encoder to the replayed tree. It also enables `Decoder::generate_multimodal` and adds the `vision-language-check` build target. The seven tool sources retain their frozen bytes in `patches/vision-tools.patch`.

## Regenerate the CPU inputs

The replay reads published W1 media bodies and case metadata at native commit `df9335cb5ce2f1f8e4f462c0b3c5d76a9c513250`. It generates six additional cases in memory. Set `CHECKPOINT` to the Qwen checkpoint and `VLLM_ENV` to the pinned vLLM environment. The environment must contain `venv/bin/python` and `source/` checked out at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. Set `SGLANG_REPO` to an upstream SGLang clone with commit `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. The replay extracts eight reference files to scratch and verifies their hashes against the replayed oracle.

The oracle expects Torch `2.14.0+cu132`, TorchVision `0.29.2`, Transformers `5.23.0`, vLLM `0.34.0rc1.dev787+g81a8cb063`, NumPy `2.3.5`, Pillow `13.0.0`, Tokenizers `0.23.3` and OpenCV headless `5.1.0.95`. It uses the recorded OpenCV FFmpeg backend. Compile the preprocessor using the default Pixi C++20 compiler or set `CXX` explicitly.

```sh
repo=$PWD
bundle="$repo/checkpoints/2026-11-19"
work=$(mktemp -d)
revision=f04d886bf79aa3d6237cbb12a3311838eb9f9bd8
git -C "$SGLANG_REPO" archive --format=tar.gz --prefix="sglang-$revision/" \
    "$revision" > "$work/sglang.tar.gz"
bash "$bundle/vision/replay-preprocess.sh" "$repo" "$CHECKPOINT" "$VLLM_ENV" \
    "$work/sglang.tar.gz" "$work/inputs"
```

The replay verifies the checkpoint config, safetensors header, BF16 payload, generated fixture bytes and reference source identities. It checks these SHA-256 values.

```text
f27c049c6c09f34371e92b86f7b08e6a071f95772ca8c094ea7018f7d4f87932  fixtures/results.json
36cdf21bf5719d6fb2c203d77ae856fe02986d45b43c1864b229d0229f6a4d1e  manifest/checkpoint_manifest.inc
```

## Build and run the native targets

In the default Pixi environment, build from the verified tree using the generated manifest.

```sh
tree="$work/inputs/trees/vision"
export VISION_NATIVE_SOURCE="$tree/vision/encoder"
export VISION_NATIVE_MANIFEST="$work/inputs/manifest"
export VISION_RUN="$VISION_NATIVE_MANIFEST"

(
    cd "$tree"
    xmake f -P . -m release -o "$work/build-vision" -y
    xmake -P . -j 4 vision-language-check
    xmake -P . -j 4 sglang-generate
)
(
    cd "$VISION_NATIVE_SOURCE"
    xmake f -P . -m release -o "$work/build-encoder" -y
    xmake -P . -j 4 vision-encoder
)
```

Expose one leased B200 to CUDA. Run the encoder on both fixtures and the decoder smoke test with four generated steps.

```sh
fixtures="$work/inputs/fixtures"
mkdir -p "$work/encoded/image" "$work/encoded/video"
"$VISION_RUN/build/bin/vision-encoder" "$CHECKPOINT/outside.safetensors" \
    "$fixtures/W1-image/native" "$work/encoded/image" \
    "$fixtures/W1-video/native" "$work/encoded/video"
"$work/build-vision/bin/vision-language-check" "$CHECKPOINT" "$fixtures" "$work/native" 4
```

The GPU smoke exercises native wiring. The historical numerical comparison used separate HF and vLLM references and recorded stage outputs. Those traces and their comparison harnesses are outside this source checkpoint.
