# Native prefill workspace

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder prefill | Own the arena and lend two scratch spans during GQA. | [Tile layout](../src/decoder_prefill.cpp#L173)<!-- prefix: "TileBuffers(" --> · [Scratch view](../src/decoder_prefill.cpp#L195)<!-- prefix: "PairedPrefil" --> |
| Native Flash dispatcher | Try staged attention within the default paired route. | [Selection](../src/mxfp8_gqa_prefill.cu#L871)<!-- prefix: "const bool d" --> |
| Paired launcher | Check bounds and capture state, then submit predecode and attention on the caller's stream. | [Staged launch](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L389)<!-- prefix: "std::optiona" --> |

## Runtime path

Under `tensor-core`, the native dispatcher can stage a 1,024-row tile at starts 8,192 through 31,744 inclusive. The start must be a multiple of 1,024 or have a nonzero offset within a 128-token page. This uses the default paired selection with `SGLANG_EXPERIMENTAL_TMEM_PREFILL` unset or set to `ragged`. The [SM100 page-128 opt-ins](sm100-page128-prefill.md) run before native dispatch. [Decoder call](../src/decoder_prefill.cpp#L323)<!-- prefix: "const auto s" -->

The decoder lends two disjoint spans of its existing `PrefillWorkspace` allocation. The first address is inclusive and the end address is exclusive. These capacities apply to a 1,024-row arena.

| Scratch use | Physical span | Capacity | Largest contents at start 31,744 |
| --- | --- | ---: | ---: |
| BF16 K | `mlp_gate` through the end of `mlp_up`, bounded by `q_gate` | 68 MiB | 64 MiB |
| BF16 V and invalid-token indices | `gdn_qkv`, `gdn_z`, `gdn_b`, `gdn_a`, and GDN scratch, bounded by `activation.values` | 64.1875 MiB | 64 MiB + 4 KiB |

Each span has room for up to 127 bytes of padding for 128-byte alignment. The launcher checks both spans separately. Workspace allocation size, memory pool setup, and KV cache capacity stay the same. The staged route adds zero device allocation bytes. [Arena allocation](../src/decoder_prefill.cpp#L501)<!-- prefix: "PrefillWorks" --> · [Bounds](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L398)<!-- prefix: "const int fu" -->

The MLP projection writes its gate and up buffers after GQA. GDN layers write their projection and scratch inputs before use. Predecode, paired attention, later layer work, and workspace release use the workspace's execution stream. Stream ordering puts the last attention read before the next write to either span. Keep both spans exclusive to that stream until the queued work completes. [Layer loop](../src/decoder_prefill.cpp#L622)<!-- prefix: "execute_laye" --> · [Release](../src/decoder_prefill.cpp#L556)<!-- prefix: "cudaError_t " -->

Predecode follows the page table and expands each complete MXFP8 page once. Separate K and V TMA producers load alternating shared-memory stages. A partial page uses the native masked conversion. The paired kernel retains the ordered BF16 operands, FP32 softmax and accumulation, and protected repair. [Predecode](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L77)<!-- prefix: "__global__ v" --> · [Producers](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L146)<!-- prefix: "__device__ v" -->

The [CUDA 13.4 stream ordering guide](https://docs.nvidia.com/cuda/archive/13.4.3/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-stream-ordering) defines the execution order. The [CUTLASS SM100 MLA change](https://github.com/NVIDIA/cutlass/pull/2472) illustrates a two-CTA asynchronous attention pipeline. The native route uses its own MXFP8 layout and arithmetic.

## Constraints and failure behavior

Missing or undersized scratch, capture, and a known staging setup rejection select raw paired attention before staged work enters the stream. Setup rejection includes unavailable kernel resources, supported predecode unavailability statuses, and tensor-map rejection or descriptor out-of-memory. The raw path retains its [existing fallback rules](flash-prefill.md#constraints-and-failure-behavior). [Capture check](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L415)<!-- prefix: "const auto c" --> · [Setup and submission](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L435)<!-- prefix: "const auto e" -->

Unexpected CUDA failures return to the caller. After predecode submission succeeds, the attention launch status also returns directly. A paired graph capture records the original raw kernel with maps by value and borrows neither span. Keep the captured cache, page table, queries, and output valid through replay. [Raw launch](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L361)<!-- prefix: "std::optiona" --> · [CUDA graph capture](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/04-special-topics/cuda-graphs.html#stream-capture)

## Multi-request text prefill

On B200, a tensor-core prefill-only step with a positive budget of at most 8,192 rows can group consecutive text grants in slot order. Each grant must fit one of its original tiles and contribute at least 256 rows in multiples of 128. A group has at least two grants and at most 1,023 total rows. Normalization, quantization, dense projections, and the MLP run across the concatenated rows. GDN and GQA keep each grant's original positions, recurrent state, and KV pages. Each request receives its own final hidden and residual row. [Grouping](../src/decoder_prefill.cpp#L398)<!-- prefix: "PrefillGroup" --> · [Layer loop](../src/decoder_prefill.cpp#L412)<!-- prefix: "void prefill" -->

The step reuses one stream-ordered arena. Its row capacity covers the largest group or original scalar tile. If the larger allocation is unavailable, the step retries the original capacity before starting any grouped work. Mixed decode, tensor-prefix caching, MTP, media, and ineligible grants retain their existing paths. Use `--trace-batches` to observe grouped rows alongside the original row grants. [Step guard](../src/decoder_batch.cpp#L757)<!-- prefix: "const bool l" --> · [Allocation fallback](../src/decoder_prefill.cpp#L660)<!-- prefix: "if (!workspa" -->

## Verification

On one reserved B200, replay identical checkpoint Q, MXFP8 payloads and scales, and remapped page tables. Exercise both bounds and nonzero page offsets. Compare full BF16 output with raw paired and archived wide Flash. Include a short tail, causal masks, invalid full- and partial-page V, and repair.

Check each scratch span's bounds and canaries, including a span one byte shorter than the aligned requirement. Check descriptor rejection, empty cluster occupancy, and unavailable predecode. Require raw output parity in those cases. Inject a launch failure after predecode and require the error with no second attention launch.

Check a cold capture with scratch supplied. Require one raw kernel node and compare replay after a selector change. Check explicit `paired` and the default dispatch boundaries. Count allocation calls on each arm.

Time eager predecode and attention together with resident inputs in base, candidate, candidate, base order. Nine matched checkpoint captures gave byte-identical BF16 and 13.51–21.93% lower operator latency. This evidence covers operator output, dispatch, and allocation calls. Decoder layer lifetime and request performance require separate evidence.
