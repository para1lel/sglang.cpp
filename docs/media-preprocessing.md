# CPU media preprocessing

Link the C++17 static target `sglang_media_preprocess` and include `sglang/media_preprocess.h`.
To submit media bytes to the server, use the [chat attachment contract](chat-serving.md#image-and-video-inputs).
The library converts decoded RGB8 frames into normalized patches for the Qwen3.8-27B-FP8 vision encoder.
It uses the standard C++ runtime and runs on the CPU without loading checkpoint assets.
Pass the patches to the [standalone vision encoder](vision-encoder.md) to produce BF16 features.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Caller | Decode media to RGB8 and provide displayed dimensions and source video metadata. | [Input view](../include/sglang/media_preprocess.h#L10)<!-- prefix: "struct Decod" --> · [Video metadata](../include/sglang/media_preprocess.h#L20)<!-- prefix: "struct Video" --> |
| Preprocessor | Select frames, resize, normalize, and pack patches. | [Public functions](../include/sglang/media_preprocess.h#L35)<!-- prefix: "std::vector\u003c" --> · [Library target](../xmake.lua#L269)<!-- prefix: "target(\"sgla" --> |
| Consumer | Read the owned patch buffer, grid, and video metadata. | [Result](../include/sglang/media_preprocess.h#L26)<!-- prefix: "struct Media" --> |

## Runtime path

`DecodedRgb` borrows tightly packed `[frames, height, width, 3]` bytes with RGB in that order.
All frames have the same displayed size. Keep the buffer live until the call returns.
The caller handles orientation and color conversion before the call.

```cpp
#include "sglang/media_preprocess.h"

auto patches = sglang::preprocess_image({rgb.data(), rgb.size(), height, width, 1});
```

Use a byte buffer `rgb` with exactly `height * width * 3` bytes in this example.
`preprocess_image` accepts one frame and uses antialiased bicubic resize on RGB8.
The sizing rule aligns each axis to 32 and uses area targets of 65,536 and 16,777,216 pixels.
It puts the frame in both temporal planes and returns empty frame indices and timestamps.
[Image path](../src/media_preprocess.cpp#L192)<!-- prefix: "MediaPatches" -->

Set `VideoMetadata::source_frame_count` to the full source length.
Set `VideoMetadata::source_fps` to the source frame rate.
Call `preprocess_video` with `DecodedRgb`, `VideoMetadata`, and one of these profiles.

| Profile | Input frames | Processing |
| --- | --- | --- |
| `VideoPreprocess::SglangDefault` | Supply every source frame in order and leave `frame_indices` empty. Alternatively, supply only the frames returned by `sample_video_frames` with those exact indices. | Resize with the legacy factor-28 rule and antialiased float bilinear interpolation. Round to RGB8 with ties to even. Then apply the checkpoint video resize. |
| `VideoPreprocess::Checkpoint` | Supply decoded frames and one source index per frame. Use this profile for a caller-selected sampling policy. | Apply checkpoint video resize directly to the supplied frames. |

`sample_video_frames` targets two frames per second and at least four frames when available.
It caps the count at the source length or 768, whichever is smaller, and rounds down to an even count.
It spaces indices from the first to the last source frame and truncates interior positions to integers.
[Sampler](../src/media_preprocess.cpp#L177)<!-- prefix: "std::vector\u003c" -->

The checkpoint video sizing rule adjusts a short edge below 32 before it computes the final dimensions.
It aligns each axis to 32 and uses targets of 4,096 and 25,165,824 pixel-frames.
The video path resizes to those dimensions with antialiased bicubic interpolation on RGB8.
The sizing rules round to aligned dimensions, then adjust dimensions for the area targets.
Native storage checks use the resulting dimensions.
[Video sizing](../src/media_preprocess.cpp#L83)<!-- prefix: "Size checkpo" --> · [Video path](../src/media_preprocess.cpp#L202)<!-- prefix: "MediaPatches" -->

### Patch layout and timestamps

`MediaPatches::pixels` owns contiguous float32 values with shape `[T * H * W, 1536]`.
`grid_thw` is `{T, H, W}` in raw patch units. `T` counts pairs of frames.
`H` and `W` are the resized height and width divided by 16.
Rows advance by temporal pair, then by spatial 2×2 patch block in row-major order.
Within each block, rows visit its four patches in row-major order.
Each row contains `[RGB channel, temporal plane, pixel row, pixel column]` with dimensions `[3, 2, 16, 16]`.
The column index varies fastest. Each value is `(pixel - 127.5f) / 127.5f`.
The layout gives `T * H * W / 4` merged rows after spatial merging.
[Packing](../src/media_preprocess.cpp#L106)<!-- prefix: "void pack_pl" --> · [Grid assignment](../src/media_preprocess.cpp#L139)<!-- prefix: "result.grid_" -->

An odd video selection repeats the last frame in the final temporal pair.
`frame_indices` preserves the selected zero-based source indices before padding.
`timestamps_seconds` contains one center per temporal pair.
For source indices `a` and `b`, the center is `(a / source_fps + b / source_fps) / 2`.
The final odd pair uses the same source index twice.
These centers use source FPS and frame ordinals. A caller that needs container timestamps must track them separately.
[Timestamp centers](../src/media_preprocess.cpp#L165)<!-- prefix: "std::vector\u003c" -->

## Constraints and failure behavior

- Input height and width must be within 1 through 16,384. The aspect ratio limit is 200 at each sizing decision.
- Source video length must be within 2 through 1,000,000,000 frames. Source FPS must be finite and within 0.001 through 1,000,000.
- A preselected video must contain 2 through 768 frames. Its indices must be in source range and nondecreasing. Repeats are permitted by the checkpoint profile.
- The input bytes, float patch output, and each resize pixel buffer have separate 256 MiB limits.

Invalid inputs throw `std::invalid_argument`. Storage limit violations throw `std::length_error`.
Allocation can fail within the limits. Apply separate limits when decoding the input media.
[Input checks](../src/media_preprocess.cpp#L27)<!-- prefix: "void validat" --> · [Index checks](../src/media_preprocess.cpp#L155)<!-- prefix: "void validat" --> · [Output limit](../src/media_preprocess.cpp#L98)<!-- prefix: "std::size_t " -->

## Verification

On the project host, use XMake and the provisioned default Pixi toolchain.
Build the explicit CPU check target from the repository root. It builds the static library as a dependency.

```bash
host=/data1/shared/dongwu.chen
export CONDA_PREFIX="$host/.pixi/envs/default"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export PATH="$CONDA_PREFIX/bin:$PATH"
build_dir="$PWD/build/media-preprocess"

"$host/bin/xmake" f -o "$build_dir" -p linux -a x86_64 -m release --media-preprocess-check=y
"$host/bin/xmake" build media-preprocess-check
"$build_dir/bin/media-preprocess-check" sample 8 4
```

Expect sample indices `0 2 4 7`. To build only the library, select the target `sglang_media_preprocess`.
The CPU targets link the C++ runtime. The project uses the provisioned Pixi environment for XMake configuration.
[Check target](../xmake.lua#L283)<!-- prefix: "target(\"medi" --> · [Check driver](../tests/media/preprocess_check.cpp#L98)<!-- prefix: "void check(i" -->

For the numerical oracle, set `ORACLE_PYTHON` to an absolute interpreter path.
Use a reference environment with CPU-capable PyTorch, torchvision, NumPy, and Transformers 5.19.0.
Use SGLang revision `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`.
Set `SGLANG_MEDIA_SOURCE` to its local `python/sglang/srt/multimodal/qwen_vl_media_processing.py` file.
Use a local copy of the [pinned checkpoint](research/checkpoint.md) for its processor assets.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${ORACLE_PYTHON:?set the reference interpreter}" -B tests/media/verify_preprocess.py \
    "$build_dir/bin/media-preprocess-check" \
    "${SGLANG_MEDIA_SOURCE:?set the pinned SGLang source file}" \
    /data0/shared/Qwen3.8-27B-FP8 artifacts/media-preprocess-fixtures
```

The runner checks the SGLang source digest and runs its legacy sampler and resize on decoded RGB fixtures.
The checkpoint processor supplies the reference patch values, grid, and timestamp calculation.
Require equal grids, indices, timestamps, and float32 bytes. The runner also checks invalid inputs.
Keep the JSON report and raw fixtures for a failing case. A mismatch fails the run before it prints the report.
[Oracle](../tests/media/verify_preprocess.py#L116)<!-- prefix: "def oracle(c" --> · [Comparison](../tests/media/verify_preprocess.py#L158)<!-- prefix: "assert len(o" -->
