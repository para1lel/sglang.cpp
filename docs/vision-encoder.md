# Standalone vision encoder

Link `sglang_vision` and include `sglang/vision.h` to encode one preprocessed image or video on a B200.
The encoder produces BF16 features for the pinned Qwen3.8-27B-FP8 checkpoint.
Use [CPU media preprocessing](media-preprocessing.md) to prepare decoded RGB8 input.
Pass features to the [multimodal decoder](multimodal-decoder.md) for native generation. For requests carrying media bytes, use the [chat endpoint](chat-serving.md#image-and-video-inputs).
[Public API](../include/sglang/vision.h#L29)<!-- prefix: "class Vision" -->

## Load the encoder

Select the CUDA device before construction.
Create a `sglang::Checkpoint` from the local [pinned checkpoint](research/checkpoint.md).
Pass it to `sglang::VisionEncoder`.
The constructor validates the vision configuration and requires exactly 333 indexed `model.visual.*` tensors.
It checks the names, BF16 dtype, and shapes of the required tensors.
The pinned configuration has an empty DeepStack index list and produces zero DeepStack feature levels.
The constructor copies 921,460,192 bytes of weights to the current device and waits for the upload.
The encoder owns those weights. The caller can release the host checkpoint after construction.
[Configuration validation](../src/checkpoint.cpp#L340)<!-- prefix: "void Checkpo" --> · [Weight validation](../src/vision/weights.cu#L53)<!-- prefix: "const Bf16* " -->

## Encode patches

Use the preprocessor's raw `grid_thw` as `{T, H, W}` in `VisionGrid`.
`T` counts temporal pairs. `H` and `W` count patches before spatial merging.
Keep source frame indices and timestamps with the caller's metadata.
The grid requires `T >= 1`, positive even `H` and `W`, `H * W <= 4096`, and `T * H * W <= 16384`.
Call `vision_feature_rows(grid)` to validate those limits and get `T * H * W / 4`.
The CPU preprocessor has its own storage limits. Check the encoder grid before allocating device buffers.
[Grid validation](../src/vision/encoder.cu#L10)<!-- prefix: "Grid checked" -->

Upload the contiguous `MediaPatches::pixels` buffer as finite FP32 values.
Keep the [preprocessor's patch order](media-preprocessing.md#patch-layout-and-timestamps).
The input has `T * H * W` rows of 1,536 values. Allocate one output row of 5,120 BF16 values per spatial block.
Use aligned, disjoint device buffers with the exact element counts.
The caller guarantees that both allocations cover the full spans.

```cpp
const sglang::VisionGrid grid{media.grid_thw[0], media.grid_thw[1], media.grid_thw[2]};
const auto feature_elements = sglang::vision_feature_rows(grid) * sglang::vision_feature_width;
encoder.encode(grid, device_patches, media.pixels.size(), device_features, feature_elements, stream);
```

Use a stream on the encoder's device outside CUDA graph capture.
Order the patch upload before `encode` on that stream.
Each call owns its scratch buffers and synchronizes the stream before a successful return.
Keep both device buffers and the encoder live through the call.
An unsupported grid, device, stream, or span fails validation. Allocation and CUDA failures also throw.
Discard the output if the call throws.
[Stream and span checks](../src/vision/encoder.cu#L35)<!-- prefix: "void validat" --> · [Encode and completion](../src/vision/encoder.cu#L161)<!-- prefix: "void VisionE" -->

The forward path projects patches, adds learned positions, and runs 27 transformer blocks with split-half height and width RoPE.
Attention is bidirectional within each temporal pair.
The spatial merger maps each consecutive group of four patches into one feature row.
Output rows follow temporal pair and then spatial block order.
[Forward path](../src/vision/encoder.cu#L117)<!-- prefix: "void forward" -->

## Build and verify

Use the C++17 and CUDA toolkit in the project's default Pixi environment.
Run this command from the repository root.

```bash
host=/data1/shared/dongwu.chen
"$host/bin/pixi" run --manifest-path "$host/pixi.toml" --environment default --as-is \
    bash -c '
        set -e
        "$1" f -P . -o build/vision -p linux -a x86_64 -m release
        "$1" build -P . sglang_vision
    ' bash "$host/bin/xmake"
```

The static library uses native SM100 CUDA kernels. Its XMake target links `sglang_checkpoint` and CUDA Runtime.
The checkpoint target also links `sglang_json`. The runtime uses C++ and CUDA libraries.
[Vision target](../xmake.lua#L129)<!-- prefix: "target(\"sgla" --> · [Checkpoint target](../xmake.lua#L121)<!-- prefix: "target(\"sgla" -->

On a reserved B200, feed the same pinned FP32 patch fixtures to the public API and the independent reference.
Use FP64 affine, LayerNorm, and attention reductions with BF16 activation boundaries in the reference.
Check the output shape and finiteness. Compare raw BF16 output and record any difference by fixture.
For a video with two temporal pairs, change the second pair and confirm the first pair's features stay identical.
Test the accepted grid limits and confirm that invalid grids, spans, and checkpoint schemas fail.
Keep input and output digests and reference settings with the result.
