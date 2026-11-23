# Score-backed singleton GQA decode

The score-backed singleton descriptions below cover the earlier packed
runtime. Current B200 text decode and verification are described under
[split-KV target attention](#split-kv-target-attention).

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Batch scheduler | Selects eligible text decode steps. | [Batch options](../src/decoder_batch.cpp#L592)<!-- prefix: "const auto r" --> · [Lane selection](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L486)<!-- prefix: "first.narrow" --> |
| Batch execution owner | Allocates and releases the shared score buffer. | [Allocation](../src/decoder_resources.cpp#L551)<!-- prefix: "ExactGqaWork" --> · [Device release](../src/decoder_resources.cpp#L259)<!-- prefix: "CleanupError" --> |
| Exact attention | Computes FP32 scores and consumes them in token order. | [Score and replay launches](../include/sglang/text/gqa_ordered_launch.cuh#L4)<!-- prefix: "cudaError_t " --> |

## Runtime path

The batch selects this path for one ordinary text decode row with 32,768–36,864 visible tokens, including the current token. The step needs zero prefill rows, no MTP work, and no rotary position override. Both FP8 prefill policies support it. Independent full-prefix-hit scoring can share the step. The standalone serial decoder and the eager remainder of a packed group keep their existing attention paths. [Window](../include/sglang/decoder_internal.h#L468)<!-- prefix: "inline const" --> · [Packed remainder](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L573)<!-- prefix: "if (remainin" -->

With tensor-core prefill and no MTP work, a prefill step can reuse the buffer for an ordinary text lane's final one-row prompt grant. It can also reuse it for an ordinary decode row when all other work in the step consists of multi-row prefill or cache-hit scoring. These eager rows qualify above their inline score capacity and through 36,864 visible tokens. Each layer's score and replay launches run in stream order before the next row uses the buffer. [Batch selection](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/decoder_batch.cpp#L675)<!-- prefix: "const auto e" --> · [Eager binding](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L421)<!-- prefix: "void prepare" -->

The batch stream owner allocates one buffer on first eligible use. It allocates before graph capture or eager submission. The bound is 3,538,944 device bytes (3.375 MiB) for 24 FP32 scores per position across 36,864 positions. All GQA layers and later requests share the buffer in stream order. It adds zero request-owned bytes. Request KV and existing graph scratch have separate allocations. [Query heads](../include/sglang/mxfp8_gqa.h#L11)<!-- prefix: "inline const" --> · [Resident admission](batch-graphs.md#resident-memory-admission)

The step rounds the live visible length up to a 128-token page for the per-head stride. It requires a page table and K/V payload and scale buffers with sufficient declared capacity in every attention layer. Eager decode passes the score descriptor by value. With graphs enabled, the first eligible width-one step warms up eagerly. Replay uploads the live position, page table, KV cache, and score descriptor. The same width-one executable can use short inline scores or the batch buffer. Both score-backed routes round the attention result to BF16. [Eligibility and stride](../src/text/decoder_graph.cpp#L9)<!-- prefix: "ExactGqaWork" --> · [Eager read](../src/decoder_forward.cpp#L418)<!-- prefix: "if (singleto" --> · [Warmup](../src/text/decoder_graph.cpp#L36)<!-- prefix: "auto& warmed" --> · [Replay binding](../src/text/decoder_graph.cpp#L66)<!-- prefix: "}" -->

Eager final-prompt replay stages packed V and its scales in two shared-memory buffers at 8,192 or more visible tokens when their bases are 16-byte and 8-byte aligned, respectively. Each CTA handles one query head and 64 value channels. Asynchronous copies follow the page table and zero-fill the invisible part of the last tile. The path reuses the score producer and preserves each channel's token-ordered FP32 recurrence. Eager decode, graph replay, shorter contexts, and misaligned buffers keep the existing replay. [Staged replay](../include/sglang/text/gqa_prefill_replay.cuh#L34)<!-- prefix: "__global__ v" -->

## Historical vectorized ordered replay

On B200, `tensor-core-mtp-target` uses vectorized V loads for score-backed text decode and verification. Each CTA stages 32 channels across a 32-token tile of decoded MXFP8 V. The kernel retains the score reduction, ascending FP32 recurrence, division, and BF16 rounding. When a tile's scores are at or below a finite running maximum, the history multiplier is one, so the numerator FMA reads the prior numerator directly. Each row reads only its visible positions, including after a partially accepted round. Prefill, draft attention, media, and GDN keep their existing dispatch.

## Split-KV target attention

The B200 text target selects split-KV attention for eligible decode and verification rows from 2,048 through 262,144 visible tokens. [Split target dispatch](../include/sglang/text/gqa_target_split.cuh#L6)<!-- prefix: "__global__ v" -->

Each verification row uses its own causal position. The kernel reads paged MXFP8 K/V, decodes it to BF16 for Tensor Core attention, and merges splits in FP32 before rounding to BF16. Ineligible heads in the window use the existing wide FP64 replay. The changed dot-product and softmax reductions can change target scores and greedy token IDs. Outside the window, rows use ordered attention on the current request state. Earlier split rows can affect that state.

Batch graph replay checks each live position on device. The standalone generator uses eager decode if the request can enter the window. Prefill, draft attention, media, and pointwise scoring use their existing paths. The split operator uses the existing checkpoint, KV capacity, and logits scratch.

Eager verification groups 10–20 rows when every row's Tensor Core inputs are aligned and its work
averages at most 10 chunks per split. Smaller groups, unaligned inputs, higher
split work, and bound graph replay use groups of at most five. Each row keeps
its own causal position and scratch. The split count is capped at 16 per row,
so scratch size is independent of context length. FP64 head fallback still
applies when inputs fall outside the split kernel's numerical range.

The earlier bounded and long profiles used explicit launch settings. Their
[historical paired results](serving/paired-split-attention-results.md) record
the measured configuration.

## Constraints and failure behavior

Short singleton replay remains eligible through 5,120 visible tokens. Decode-only singletons at 5,121–32,767 use direct ordered eager attention. Above 36,864 tokens or when the page table or a layer's buffers lack the required capacity, the new eager binding stays empty and existing dispatch rules apply. The batch buffer stays fixed even if the request reserves more KV. The legacy width-two through width-four graph keeps its separate request-owned long scores. [Replay limit](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L436)<!-- prefix: "if (count ==" --> · [Direct eager read](../src/decoder_forward.cpp#L423)<!-- prefix: "return mxfp8" --> · [Context limit](../include/sglang/text_attention.h#L9)<!-- prefix: "inline const" --> · [Legacy scores](batch-graphs.md#constraints-and-failure-behavior)

After clean score-buffer allocation OOM, the batch remembers the failure for its lifetime and uses direct ordered eager attention for long singletons. Short replay remains eligible. Other allocation errors propagate. A recoverable graph rejection retires the graph for that width. The eager fallback can use the retained score buffer. The buffer survives request retirement and KV growth. Batch cleanup drains the stream and destroys graph handles before releasing the allocation. Cleanup retains device resources while stream or graph completion is uncertain. [Allocation fallback](../src/decoder_resources.cpp#L559)<!-- prefix: "if (error.st" --> · [Graph recovery](../src/decoder_resources.cpp#L610)<!-- prefix: "const auto f" --> · [Graph teardown](../src/decoder_resources.cpp#L225)<!-- prefix: "CleanupError" -->

## Verification

Run the same-input dispatch fixture on a reserved B200. Compare eager and replay BF16 attention bitwise with ordered eager. Exercise 32,767–32,769 and 36,864–36,865 visible tokens. Rebind short and long requests to the same executable. Include a remapped page table and a partial last page. Verify one batch buffer allocation. Verify that request-owned score bytes stay constant.

The [short-prefill checker](../tests/mxfp8_gqa_short_prefill_check.cu#L119)<!-- prefix: "void check_s" --> compares staged and ordered replay at the visibility threshold and on reversed pages, including misaligned V and scales.

Inject score-buffer OOM and a recoverable graph launch rejection in separate runs. Check direct ordered fallback after OOM. Check score-backed eager after graph rejection. Record warm dispatch timing and serving timing as separate measurements. See the [batch graph checks](batch-graphs.md#verification) for HTTP parity and failure recovery.
