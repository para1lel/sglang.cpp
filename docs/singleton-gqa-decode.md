# Score-backed singleton GQA decode

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Batch scheduler | Selects eligible text decode steps. | [Batch options](../src/decoder_batch.cpp#L691)<!-- prefix: "const auto r" --> · [Lane selection](../src/text/decoder_graph_packed.cpp#L472)<!-- prefix: "first.narrow" --> |
| Batch execution owner | Allocates and releases the shared score buffer. | [Allocation](../src/decoder_resources.cpp#L455)<!-- prefix: "ExactGqaWork" --> · [Device release](../src/decoder_resources.cpp#L263)<!-- prefix: "CleanupError" --> |
| Exact attention | Computes FP32 scores and consumes them in token order. | [Score and replay launches](../src/mxfp8_gqa.cu#L810)<!-- prefix: "cudaError_t " --> |

## Runtime path

The batch selects this path for one ordinary text decode row with 32,768–36,864 visible tokens, including the current token. The step needs zero prefill rows, no MTP work, and no rotary position override. Both FP8 prefill policies support it. Independent full-prefix-hit scoring can share the step. The standalone serial decoder and the eager remainder of a packed group keep their existing attention paths. [Window](../include/sglang/decoder_internal.h#L406)<!-- prefix: "inline const" --> · [Packed remainder](../src/text/decoder_graph_packed.cpp#L558)<!-- prefix: "if (remainin" -->

The batch stream owner allocates one buffer on first eligible use. It allocates before graph capture or eager submission. The bound is 3,538,944 device bytes (3.375 MiB) for 24 FP32 scores per position across 36,864 positions. All GQA layers and later requests share the buffer in stream order. It adds zero request-owned bytes. Request KV and existing graph scratch have separate allocations. [Query heads](../include/sglang/mxfp8_gqa.h#L11)<!-- prefix: "inline const" --> · [Resident admission](batch-graphs.md#resident-memory-admission)

The step rounds the live visible length up to a 128-token page for the per-head stride. It requires a page table and K/V payload and scale buffers with sufficient declared capacity in every attention layer. Eager decode passes the score descriptor by value. With graphs enabled, the first eligible width-one step warms up eagerly. Replay uploads the live position, page table, KV cache, and score descriptor. The same width-one executable can use short inline scores or the batch buffer. Both score-backed routes round the attention result to BF16. [Eligibility and stride](../src/text/decoder_graph_packed.cpp#L399)<!-- prefix: "ExactGqaWork" --> · [Eager read](../src/decoder_forward.cpp#L202)<!-- prefix: "if (singleto" --> · [Warmup](../src/text/decoder_graph_packed.cpp#L426)<!-- prefix: "auto& warmed" --> · [Replay binding](../src/text/decoder_graph_packed.cpp#L457)<!-- prefix: "scratch.bind" -->

## Constraints and failure behavior

Short singleton replay remains eligible through 5,120 visible tokens. At 5,121–32,767 and above 36,864, singletons use direct ordered eager attention through the model's 262,144-token limit. The batch buffer stays fixed even if the request reserves more KV. The legacy width-two through width-four graph keeps its separate request-owned long scores. [Replay limit](../src/text/decoder_graph_packed.cpp#L422)<!-- prefix: "if (count ==" --> · [Direct eager read](../src/decoder_forward.cpp#L207)<!-- prefix: "return mxfp8" --> · [Context limit](../include/sglang/text_attention.h#L9)<!-- prefix: "inline const" --> · [Legacy scores](batch-graphs.md#constraints-and-failure-behavior)

After clean score-buffer allocation OOM, the batch remembers the failure for its lifetime and uses direct ordered eager attention for long singletons. Short replay remains eligible. Other allocation errors propagate. A recoverable graph rejection retires the graph for that width. The eager fallback can use the retained score buffer. The buffer survives request retirement and KV growth. Batch cleanup drains the stream and destroys graph handles before releasing the allocation. Cleanup retains device resources while stream or graph completion is uncertain. [Allocation fallback](../src/decoder_resources.cpp#L463)<!-- prefix: "if (error.st" --> · [Graph recovery](../src/decoder_resources.cpp#L524)<!-- prefix: "const auto f" --> · [Graph teardown](../src/decoder_resources.cpp#L229)<!-- prefix: "CleanupError" -->

## Verification

Run the same-input dispatch fixture on a reserved B200. Compare eager and replay BF16 attention bitwise with ordered eager. Exercise 32,767–32,769 and 36,864–36,865 visible tokens. Rebind short and long requests to the same executable. Include a remapped page table and a partial last page. Verify one batch buffer allocation. Verify that request-owned score bytes stay constant.

Inject score-buffer OOM and a recoverable graph launch rejection in separate runs. Check direct ordered fallback after OOM. Check score-backed eager after graph rejection. Record warm dispatch timing and serving timing as separate measurements. See the [batch graph checks](batch-graphs.md#verification) for HTTP parity and failure recovery.
