# Score-backed singleton GQA decode

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Batch scheduler | Selects eligible text decode steps. | [Batch options](../src/decoder_batch.cpp#L674)<!-- prefix: "const auto r" --> · [Lane selection](../src/text/decoder_graph_packed.cpp#L486)<!-- prefix: "first.narrow" --> |
| Batch execution owner | Allocates and releases the shared score buffer. | [Allocation](../src/decoder_resources.cpp#L574)<!-- prefix: "ExactGqaWork" --> · [Device release](../src/decoder_resources.cpp#L261)<!-- prefix: "CleanupError" --> |
| Exact attention | Computes FP32 scores and consumes them in token order. | [Score and replay launches](../include/sglang/text/gqa_ordered_launch.cuh#L4)<!-- prefix: "cudaError_t " --> |

## Runtime path

The batch selects this path for one ordinary text decode row with 32,768–36,864 visible tokens, including the current token. The step needs zero prefill rows, no MTP work, and no rotary position override. Both FP8 prefill policies support it. Independent full-prefix-hit scoring can share the step. The standalone serial decoder and the eager remainder of a packed group keep their existing attention paths. [Window](../include/sglang/decoder_internal.h#L472)<!-- prefix: "inline const" --> · [Packed remainder](../src/text/decoder_graph_packed.cpp#L573)<!-- prefix: "if (remainin" -->

With tensor-core prefill and no MTP work, a prefill step can reuse the buffer for an ordinary text lane's final one-row prompt grant. It can also reuse it for an ordinary decode row when all other work in the step consists of multi-row prefill or cache-hit scoring. These eager rows qualify above their inline score capacity and through 36,864 visible tokens. Each layer's score and replay launches run in stream order before the next row uses the buffer. [Batch selection](../src/decoder_batch.cpp#L671)<!-- prefix: "const auto e" --> · [Eager binding](../src/text/decoder_graph_packed.cpp#L421)<!-- prefix: "void prepare" -->

The batch stream owner allocates one buffer on first eligible use. It allocates before graph capture or eager submission. The bound is 3,538,944 device bytes (3.375 MiB) for 24 FP32 scores per position across 36,864 positions. All GQA layers and later requests share the buffer in stream order. It adds zero request-owned bytes. Request KV and existing graph scratch have separate allocations. [Query heads](../include/sglang/mxfp8_gqa.h#L11)<!-- prefix: "inline const" --> · [Resident admission](batch-graphs.md#resident-memory-admission)

The step rounds the live visible length up to a 128-token page for the per-head stride. It requires a page table and K/V payload and scale buffers with sufficient declared capacity in every attention layer. Eager decode passes the score descriptor by value. With graphs enabled, the first eligible width-one step warms up eagerly. Replay uploads the live position, page table, KV cache, and score descriptor. The same width-one executable can use short inline scores or the batch buffer. Both score-backed routes round the attention result to BF16. [Eligibility and stride](../src/text/decoder_graph_packed.cpp#L399)<!-- prefix: "ExactGqaWork" --> · [Eager read](../src/decoder_forward.cpp#L350)<!-- prefix: "if (singleto" --> · [Warmup](../src/text/decoder_graph_packed.cpp#L440)<!-- prefix: "auto& warmed" --> · [Replay binding](../src/text/decoder_graph_packed.cpp#L471)<!-- prefix: "scratch.bind" -->

Eager final-prompt replay stages packed V and its scales in two shared-memory buffers at 8,192 or more visible tokens when their bases are 16-byte and 8-byte aligned, respectively. Each CTA handles one query head and 64 value channels. Asynchronous copies follow the page table and zero-fill the invisible part of the last tile. The path reuses the score producer and preserves each channel's token-ordered FP32 recurrence. Eager decode, graph replay, shorter contexts, and misaligned buffers keep the existing replay. [Staged replay](../include/sglang/text/gqa_prefill_replay.cuh#L34)<!-- prefix: "__global__ v" -->

## Vectorized ordered replay

On B200, `tensor-core-mtp-target` uses vectorized V loads for score-backed text decode and verification. Each CTA stages 32 channels across a 32-token tile of decoded MXFP8 V. The kernel retains the score reduction, ascending FP32 recurrence, division, and BF16 rounding. When a tile's scores are at or below a finite running maximum, the history multiplier is one, so the numerator FMA reads the prior numerator directly. Each row reads only its visible positions, including after a partially accepted round. Prefill, draft attention, media, and GDN keep their existing dispatch.

## Experimental split-KV target attention

On B200, setting `SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA=1` with `tensor-core-mtp-target` selects split-KV target attention at 2,048–4,096 visible tokens. The effective server reports `experimental_target_split_gqa: true` in `/server_info`. This combination defines a separate numerical policy. Unset the environment variable or set it to `0` to use the published ordered arithmetic with the same FP8 flag. [Split target dispatch](../include/sglang/text/gqa_target_split.cuh#L5)<!-- prefix: "__global__ v" -->

Each verification row uses its own causal position. The kernel reads paged MXFP8 K/V, decodes it to BF16 for Tensor Core attention, and merges splits in FP32 before rounding to BF16. Ineligible heads in the window use the existing wide FP64 replay. The changed dot-product and softmax reductions can change target scores and greedy token IDs. Outside the window, rows use ordered attention on the current request state. Earlier split rows can affect that state.

Batch graph replay checks each live position on device. The standalone generator uses eager decode if the request can enter the window. Prefill, draft attention, media, and pointwise scoring use their existing paths. The experiment uses the existing checkpoint, KV capacity, and logits scratch.

## Constraints and failure behavior

Short singleton replay remains eligible through 5,120 visible tokens. Decode-only singletons at 5,121–32,767 use direct ordered eager attention. Above 36,864 tokens or when the page table or a layer's buffers lack the required capacity, the new eager binding stays empty and existing dispatch rules apply. The batch buffer stays fixed even if the request reserves more KV. The legacy width-two through width-four graph keeps its separate request-owned long scores. [Replay limit](../src/text/decoder_graph_packed.cpp#L436)<!-- prefix: "if (count ==" --> · [Direct eager read](../src/decoder_forward.cpp#L355)<!-- prefix: "return mxfp8" --> · [Context limit](../include/sglang/text_attention.h#L9)<!-- prefix: "inline const" --> · [Legacy scores](batch-graphs.md#constraints-and-failure-behavior)

After clean score-buffer allocation OOM, the batch remembers the failure for its lifetime and uses direct ordered eager attention for long singletons. Short replay remains eligible. Other allocation errors propagate. A recoverable graph rejection retires the graph for that width. The eager fallback can use the retained score buffer. The buffer survives request retirement and KV growth. Batch cleanup drains the stream and destroys graph handles before releasing the allocation. Cleanup retains device resources while stream or graph completion is uncertain. [Allocation fallback](../src/decoder_resources.cpp#L582)<!-- prefix: "if (error.st" --> · [Graph recovery](../src/decoder_resources.cpp#L643)<!-- prefix: "const auto f" --> · [Graph teardown](../src/decoder_resources.cpp#L227)<!-- prefix: "CleanupError" -->

## Verification

Run the same-input dispatch fixture on a reserved B200. Compare eager and replay BF16 attention bitwise with ordered eager. Exercise 32,767–32,769 and 36,864–36,865 visible tokens. Rebind short and long requests to the same executable. Include a remapped page table and a partial last page. Verify one batch buffer allocation. Verify that request-owned score bytes stay constant.

The [short-prefill checker](../tests/mxfp8_gqa_short_prefill_check.cu#L58)<!-- prefix: "void check_s" --> compares staged and ordered replay at the visibility threshold and on reversed pages, including misaligned V and scales.

Inject score-buffer OOM and a recoverable graph launch rejection in separate runs. Check direct ordered fallback after OOM. Check score-backed eager after graph rejection. Record warm dispatch timing and serving timing as separate measurements. See the [batch graph checks](batch-graphs.md#verification) for HTTP parity and failure recovery.
