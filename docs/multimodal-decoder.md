# Native multimodal decoder

Link `sglang_decoder` and call `Decoder::generate_multimodal` to generate from text and completed visual features. For server requests, see [image and video inputs](chat-serving.md#image-and-video-inputs).

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Caller | Decode media, prepare and encode patches, render and tokenize the template, and retain feature storage. | [Media input](../include/sglang/decoder_media.h#L14)<!-- prefix: "struct Decod" --> |
| Prompt planner | Expand wrapped placeholders, assign feature spans, and compute three-axis rotary positions. | [Layout](../include/sglang/multimodal_prompt.h#L10)<!-- prefix: "struct Decod" --> · [Planner](../src/media/multimodal_prompt.cpp#L100)<!-- prefix: "MultimodalPr" --> |
| Decoder | Replace placeholder embeddings, run the language model, and deliver committed output IDs. | [Public entry](../src/media/decoder_media.cpp#L93)<!-- prefix: "DecoderResul" --> · [Embedding substitution](../src/media/decoder_media.cpp#L31)<!-- prefix: "void MediaPr" --> |

## Runtime path

### Prepare the prompt and features

Use [CPU media preprocessing](media-preprocessing.md) for decoded RGB8 frames. Pass the resulting patches and raw `{T, H, W}` grid to the [vision encoder](vision-encoder.md). `T` counts temporal pairs. `H` and `W` count patches before spatial merging. Use the preprocessor's video pair centers to format timestamps for the pinned chat template. Tokenize each formatted timestamp with the checkpoint tokenizer.

Render and tokenize the chat template before calling the decoder. Include one `<|vision_start|><|image_pad|><|vision_end|>` or `<|vision_start|><|video_pad|><|vision_end|>` per media item. Match each wrapped placeholder to a `DecoderMedia` in encounter order. Pass the same raw grid in `DecoderMedia::layout`. For an image, set `T` to one. Leave `video_timestamp_tokens` empty for images. For video, supply one nonempty timestamp token sequence per pair. [Placeholder expansion](../src/media/multimodal_prompt.cpp#L54)<!-- prefix: "void append_" -->

Pass a contiguous `[T * (H/2) * (W/2), 5120]` BF16 view for each item. Feature rows advance by temporal pair, then by merged spatial block in row-major order. Set `feature_elements` to the number of BF16 elements in the view. Use a shared owner that covers the entire view, including when the pointer is an alias into a larger allocation. Finish producer work on the decoder's CUDA device before the call. Keep features immutable until the decoder releases the shared owner. [Span checks](../src/media/decoder_media.cpp#L55)<!-- prefix: "std::shared_" --> · [Device checks](../src/media/decoder_media.cpp#L77)<!-- prefix: "void validat" -->

```cpp
#include "sglang/decoder.h"
#include "sglang/decoder_media.h"

sglang::DecoderMedia item{layout, finished_features, feature_elements};
sglang::DecoderOptions options;
options.max_new_tokens = 16;
auto result = decoder.generate_multimodal(template_token_ids, {item}, options, callbacks);
```

Here `finished_features` is the shared device owner and `callbacks` is a `DecoderCallbacks` value. Call `prepare_multimodal_prompt` with the template IDs and media layouts to obtain the expanded IDs, feature spans, and positions without using CUDA. Use the expanded ID count for prompt accounting. [Planner result](../include/sglang/multimodal_prompt.h#L31)<!-- prefix: "struct Multi" -->

### Expand, prefill, and continue

The planner preserves the outer template wrappers. It expands an image pad into one run of image pads. For video, it replaces the pad with one group per pair. Each group contains its timestamp tokens, a vision-start token, the video pad run, and a vision-end token. Each pad in a run maps to one visual feature row. The decoder substitutes those rows for the pad embeddings before language layer zero. This happens in both single-row and tiled prefill. The pinned checkpoint uses zero DeepStack levels. [Row path](../src/media/decoder_media.cpp#L23)<!-- prefix: "const Bf16* " --> · [Tiled path](../src/media/decoder_media.cpp#L31)<!-- prefix: "void MediaPr" -->

Physical token indices advance by one for every expanded ID. KV writes and causal attention use these indices. Text and wrapper tokens assign the same scalar to all three rotary axes. Within a visual plane, merged row `r` and column `c` use `(base, base + r, base + c)`. The next scalar advances by the larger merged spatial dimension. The language model rotates Q and K with interleaved temporal, height, and width sections of `[11, 11, 10]`. [Plane positions](../src/media/multimodal_prompt.cpp#L38)<!-- prefix: "void append_" --> · [Rotary axis selection](../src/text_attention_gqa.cu#L28)<!-- prefix: "__device__ i" -->

Prefill scores the first output. When the next forward pass consumes the output token at zero-based index `k`, its KV index is `expanded_prompt_length + k`. All three rotary axes then use `next_position + k`, where `next_position` is one above the largest prompt coordinate. The CPU planner returns `next_position`. [Continuation](../src/decoder.cpp#L501)<!-- prefix: "const int po" -->

## Constraints and failure behavior

Use `mtp_steps = 0`. Media prefill and decode run eager with private KV and GDN state. Call `Decoder::generate_multimodal` for serial generation or `DecoderBatch::admit_multimodal` for a batch lane. A media lane bypasses prefix reuse and retains its feature owners through normal or deferred CUDA cleanup. Text lanes can advance in the same batch. [Serial allocation](../src/decoder.cpp#L538)<!-- prefix: "const Reques" --> · [Batch admission](../src/decoder_batch.cpp#L453)<!-- prefix: "int DecoderB" -->

Media prefill uses tiles of up to 128 positions, including with the tensor-core policy. The target step cap can reduce the executed rows. A serial request can fall back to single-row prefill when workspace allocation fails. [Serial tiles](../src/decoder.cpp#L462)<!-- prefix: "const MediaP" --> · [Batch tiles](../src/decoder_batch.cpp#L766)<!-- prefix: "const auto t" -->

The decoder accepts the normal greedy, [sampling](text-serving.md#sampling), and [finite-choice](text-serving.md#finite-choice-constraints) options for K0. The usual token and cancellation callbacks apply. The decoder accepts salts of up to 256 bytes. The salt has no effect on media generation. `cached_prompt_tokens` stays zero. [Request validation](../src/media/decoder_media.cpp#L100)<!-- prefix: "DecoderResul" -->

- The grid requires a positive pair count and positive even spatial axes. Apply the encoder's separate patch limits when it produces the features.
- Expanded prompt length plus output budget must fit 262,144 tokens. Use IDs in `[0, 248320)`.
- Video timestamps may contain ordinary valid IDs. The planner rejects the two media pads and two vision wrappers in timestamps.
- The planner rejects missing, extra, or mismatched placeholders and malformed layouts. The decoder checks feature size, alignment, shared ownership, and CUDA device.
- The caller guarantees the full feature allocation and completion of producer work. A zero output budget or cancellation before device work skips device validation after host validation.

## Verification

Use the managed Pixi default environment. Run from the repository root. Build [`sglang_decoder`](../xmake.lua#L111)<!-- prefix: "target(\"sgla" --> before linking the fixture program. Call `prepare_multimodal_prompt` in the CPU planner check. Supply the same template IDs, grid, and timestamp IDs as the reference.

```bash
export CUDA_VISIBLE_DEVICES=''
xmake f -P . -m release -o build/media-decoder
xmake build -P . sglang_decoder
```

Compare expanded IDs, all three position arrays, continuation delta, and feature spans with the pinned planner oracle. Exercise multiple items and a span across the 128-token tile boundary through `prepare_multimodal_prompt`. Verify rejection of malformed placeholders, timestamps, and grids. Check the combined prompt and output budget at the generation entry point.

On a reserved B200, use the pinned checkpoint and matching BF16 features for each [W1 case](../benchmarks/media/README.md#frozen-inputs). Compare 16 teacher-forced top-1 winners and 16 free-running greedy IDs per case against the pinned Hugging Face FP8 oracle. During teacher forcing, feed the next reference token after each score. Compare native and reference vision features separately from decoder arithmetic.

Record the top-two logit gap by row to identify ties and small winner margins. Reject invalid feature views and require successful generation after rejection. Exercise seeded constraints and interleave media with warm text lanes. Verify rotary axis selection on BF16 Q and K. Trace media with the default options and with explicit eager execution. Verify zero media graph launches. Use graph-eligible text as the positive replay control.

Compute relative RMSE as the L2 norm of the difference divided by the L2 norm of the reference. Report the overall and worst-row values as diagnostics. The W1 cases each have 128 expanded prompt IDs. Separately check multi-item prompts, visual spans across tiles, and single-row fallback on the GPU.
