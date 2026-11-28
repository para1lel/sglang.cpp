# GQA decode and target verification

## Responsibilities and sources of truth

The backend picks B200 split-KV or generic SM100. The batch executor owns replay and scratch. Target attention owns KV append, causal reads, and output gating. [Device policy](../src/runtime/backend/cuda_backend_policy.cpp#L17)<!-- prefix: "const bool b" --> · [Batch selection](../src/decoder_batch.cpp#L499)<!-- prefix: "run = run_de" --> · [Graph dispatch](../src/text/decoder_graph.cpp#L169)<!-- prefix: "DecodeRun ru" --> · [Score allocation](../src/runtime/backend/cuda_resources.cpp#L491)<!-- prefix: "ExactGqaWork" --> · [Eager attention](../src/runtime/execution/decoder_forward_attention.cpp#L556)<!-- prefix: "if (batch.ta" --> · [Bound singleton](../src/runtime/execution/decoder_forward_attention.cpp#L322)<!-- prefix: "for (int ind" --> · [Bound verification](../src/runtime/execution/decoder_forward_attention.cpp#L410)<!-- prefix: "void attend_" -->

## Runtime path

B200 batch replay binds live singleton state after eager warmup. In vector attention, the bound shape switches from direct to split-KV at zero-based position 2,047 and can replay through the configured split end. The inline score capacity still guards the legacy direct route where no replacement score span applies. Positive MTP uses the [verification graph policy](mtp.md#target-verification-graph). Serial generation stays eager when its request can enter the split window. [Replay policy](../src/runtime/execution/decoder_execution_policy.cpp#L21)<!-- prefix: "bool Decoder" --> · [Shape and score guard](../src/text/decoder_graph.cpp#L47)<!-- prefix: "const bool s" --> · [Serial selection](../src/decoder.cpp#L735)<!-- prefix: "bool Decoder" -->

### Generic SM100 singleton

| Generic one-lane route | Guard and workspace | Sources |
| --- | --- | --- |
| Page-parallel narrow | 1,536–2,560 visible tokens with sufficient workspace. | [Narrow selection](../src/text/decoder_graph.cpp#L178)<!-- prefix: "first.narrow" --> · [Narrow window](../src/mxfp8_gqa.cu#L366)<!-- prefix: "constexpr in" --> |
| Ordered scores and replay | Other rows with inline scores. Beyond that capacity, eager uses the scratch-free reader unless shared scores qualify. | [Inline scores](../src/runtime/model/qwen_request.cpp#L257)<!-- prefix: "ExactGqaWork" --> · [Direct reader](../src/runtime/execution/decoder_forward_attention.cpp#L207)<!-- prefix: "return mxfp8" --> |
| Shared scores, eager or replay | 32,768–36,864 visible tokens. Every GQA cache and page table must cover live rows. First use allocates 3,538,944 bytes (24 FP32 scores per position); each bind rounds stride to 128 tokens. Layers and requests reuse it in stream order. | [Eligibility and stride](../src/text/decoder_graph.cpp#L9)<!-- prefix: "ExactGqaWork" --> · [Replay binding](../src/text/decoder_graph.cpp#L102)<!-- prefix: "if (lane.sin" --> · [Score and replay](../src/mxfp8_gqa.cu#L709)<!-- prefix: "cudaError_t " --> |

<a id="scratch-free-ordered-attention"></a>

### Ordered single-row attention

On B200, eager vector prefill can use the same chunked ordered reader as draft attention. The visible span must exceed inline score capacity and fit the model context. Existing score scratch must hold one 4,096-token tile plus the FP32 carry. Each tile scores first and then replays in token order. The last tile writes the BF16 output. Invalid inputs or CUDA errors after selection propagate. [Eager route](../src/runtime/execution/decoder_forward_attention.cpp#L200)<!-- prefix: "if (mode == " --> · [Eligibility](../src/text/gqa/draft_attention.cu#L253)<!-- prefix: "decoder::try" --> · [Score and replay order](../src/text/gqa/draft_attention.cu#L167)<!-- prefix: "cudaError_t " -->

When that route declines, eager prefill can use the scratch-free reader. It uses the tiny kernel below eight visible tokens, then stages 32 keys for two query heads with causal masking and token-order accumulation. Serial graphs with reserved KV beyond inline capacity use the tiled reader and reload device position on replay. [Tiled dispatch](../src/mxfp8_gqa.cu#L801)<!-- prefix: "cudaError_t " --> · [Serial graph dispatch](../src/runtime/execution/decoder_forward_attention.cpp#L215)<!-- prefix: "if (capacity" --> · [Ordered kernel](../src/text/gqa/ordered_prefill.cu#L12)<!-- prefix: "__global__ v" --> · [Tile geometry](../include/sglang/text/mxfp8_gqa_prefill_common.cuh#L40-L41)<!-- prefix: "inline const" -->

### Draft singleton reroute

Scored draft singletons use the batch entry. Unscored seed skips attention. On B200, eligible lanes use grouped online split GQA at 2,048 or more visible tokens; other lanes stay per row. A geometry mismatch rejects the batch after possible earlier submission. Target scores choose winners and verified rows repair draft state, while approximate probabilities can change draft proposals. [Scored singleton](../src/decoder_mtp.cpp#L533)<!-- prefix: "const DraftA" --> · [Batch entry](../src/text/gqa/draft_attention.cu#L298)<!-- prefix: "cudaError_t " --> · [Draft attention](native-model-info.md#draft-attention)

### Grouped long draft reads

On B200, a scored draft wave with exactly 16 rows can launch one grouped online split read and combine per rank outside CUDA capture. Public TP2 submits the same wave to both draft shards. Each rank uses its two local KV heads, while an unsharded draft uses four. Every row must have a zero-based position of at least 32,768, a visible length of at most 262,144, readable pages, aligned buffers, and enough existing scratch. Rows in a group must agree on KV width. Shorter waves, capture, and clean declines use four-row packets or smaller tails. CUDA errors propagate. [Ranked proposal wave](../src/decoder_tp.cpp#L287)<!-- prefix: "MtpDraftRequ" --> · [Per-rank draft read](../src/text/draft_projection.cpp#L438)<!-- prefix: "work.read_at" --> · [Draft dispatch](../src/text/draft_projection.cpp#L262)<!-- prefix: "void DraftPr" --> · [Grouped eligibility](../src/text/gqa/draft_attention.cu#L225)<!-- prefix: "std::optiona" --> · [Multistage split](../src/text/gqa_decode_split.cu#L872)<!-- prefix: "cudaError_t " -->

The scoped TP2 width-16 probe exercises ranked drafting with mixed prompt lengths from 127 to 32,771 tokens, including three lanes at or above 32,768. The direct short-prefill fixture exercises scalar draft attention. To verify the grouped long route, compare proposals, committed IDs, and restored state against four-row execution on 16 eligible long B200 TP2 lanes and confirm one grouped read on each rank. Also cover partial waves and clean decline. [Scalar draft fixture](../tests/mxfp8_gqa_short_prefill_check.cu#L91)<!-- prefix: "check(decode" -->

## Split-KV target attention

B200 text target rows use split-KV for 2,048–262,144 visible tokens, including the current token, with ordered attention outside that range. Bound replay tests live position on device. Split-KV expands paged MXFP8 to BF16, merges FP32 partials, rounds to BF16, and replays flagged heads in FP64. The reductions can change scores and greedy IDs. [Lower bound](../include/sglang/decoder_cuda.cuh#L327)<!-- prefix: "inline const" --> · [B200 upper bound](../src/runtime/backend/cuda_backend_policy.cpp#L27)<!-- prefix: "backend.targ" --> · [Bound route and fallback](../src/text/gqa/target_split.cu#L145)<!-- prefix: "cudaError_t " --> · [Combine and FP64 replay](../src/text/gqa/target_split.cu#L68)<!-- prefix: "__global__ v" -->

### Batched eager verification

| Eager verify route | Eligibility, grouping, and fallback |
| --- | --- |
| Fused preprocessing through gating | Vector attention, full-context policy, 2–20 rows in the split window. Each row retains workspace, pages and position. Other groups do Q/K norm and RoPE, append and gating per row but can share reads. Bound replay dispatches separately. [Eligibility and launch order](../src/text/gqa/verify_batch.cpp#L13)<!-- prefix: "bool try_gqa" --> · [Eager fallback](../src/runtime/execution/decoder_forward_attention.cpp#L663)<!-- prefix: "if (!try_gqa" --> |
| Short exact scores | TP2 vector lanes below the split range use one score/replay kernel pair. A side-effect-free eligibility check leaves constrained or non-vector groups per row. [Short verify](../src/text/gqa/draft_attention.cu#L634)<!-- prefix: "cudaError_t " --> · [Eligibility](../src/text/gqa/draft_attention.cu#L614)<!-- prefix: "bool decoder" --> |
| Ordinary split launches | With full-context policy, alignment, and at most ten 64-token chunks per split on average per row, group 10–20 rows in one stage. QK transposes 16 keys across eight slots (six active) and restores token order before masking and softmax. Other eligible groups use up to five rows. Narrow, singleton, and bound launches use two stages. Scratch fits 16 splits per row. [Grouping](../src/text/gqa/target_split.cu#L231)<!-- prefix: "gqa_decode::" --> · [Launch geometry](../src/text/gqa_decode_split.cu#L865)<!-- prefix: "cudaError_t " --> · [Transposed scores](../src/text/gqa_decode_split.cu#L291)<!-- prefix: "__device__ v" --> |

The [MTP verifier](mtp.md#runtime-path) owns acceptance and the committed KV prefix.

### TP1 short verification

The TP1 verifier shares decoded keys and values across two to five ordered vector attention rows from the same request. Each selected row has a zero-based position at most 2,046, or 2,047 visible tokens. The verifier selects rows within each verification packet and can dispatch the other rows separately. [Row selection](../src/runtime/execution/decoder_forward_attention.cpp#L360)<!-- prefix: "ShortVerifyR" -->

Bound replay selects direct rows. Eager execution requires readable pages and sufficient score scratch, and excludes constrained rows. If the eager group declines, the existing per-row reader handles its rows. [Bound selection](../src/runtime/execution/decoder_forward_attention.cpp#L433)<!-- prefix: "const bool s" --> · [Eager selection](../src/runtime/execution/decoder_forward_attention.cpp#L633)<!-- prefix: "ShortVerifyR" --> · [Scratch guard](../src/text/gqa/draft_attention.cu#L614)<!-- prefix: "bool decoder" -->

Each row keeps its causal mask and existing score storage. The reader preserves the chronological FP32 recurrence before BF16 rounding. TP2 uses the short exact-score route in the table above. [Shared scores](../src/text/gqa/draft_attention.cu#L459)<!-- prefix: "__global__ v" --> · [Shared values](../src/text/gqa/draft_attention.cu#L547)<!-- prefix: "__global__ v" -->


### Shared KV reads for long K4 packets

Under the full-context policy, a 20-row eager packet can share reads across four distinct caches and page tables, with five consecutive positions in one 64-token chunk per request. Every zero-based position is 32,768–40,959, with readable KV, alignment and valid scratch. Each request/KV-head/split CTA shares staged K/V and scales across five rows, with independent masks and partials. Other packets use ordinary split. [Packet dispatch](../src/text/gqa/target_split.cu#L219)<!-- prefix: "if (count ==" --> · [Eligibility](../src/text/gqa_decode_split.cu#L740)<!-- prefix: "bool make_re" --> · [Causal operands](../src/text/gqa_decode_split.cu#L711)<!-- prefix: "if (chunk \u003c " -->

### Tensor Core reads across long K4 packets

The long verifier needs eager vector attention and the full-context policy. Selected rows must be unconstrained, unbound, aligned, and have readable KV and valid buffers. Duplicate page tables prevent selection. Positions below are zero-based.

| Shape | Selection | Splits and scratch |
| --- | --- | --- |
| TP2, 8×5, 12 Q and 2 KV heads per rank | Distinct page tables. First position 8,191–10,235. The fifth must average ≤10 64-token chunks per split by the 20-row packing rule, so it reaches at most 10,239 (10,240 visible). Starts can differ and cross chunk/page boundaries. Other shapes can try the suffix route. Once this route is selected, any ineligible row returns the round to ordinary packets. [Shape guard](../src/text/gqa/verify_batch.cpp#L143)<!-- prefix: "bool is_rank" --> · [Split packing rule](../src/text/gqa/geometry.cpp#L24)<!-- prefix: "bool support" --> | Up to 16 splits. |
| TP2, 16×5, 12 Q and 2 KV heads per rank | Every first position 32,768–36,863 (32,769–36,864 visible). [Wide dispatch](../src/text/gqa_decode_long.cu#L688)<!-- prefix: "bool rank_wi" --> · [Scratch bound](../include/sglang/text/gqa/gqa_decode_long.cuh#L17)<!-- prefix: "static_asser" --> | 9 splits. Partials use 112,320 B per row. |
| One selected TP2 request, 12 Q and 2 KV heads per rank | The selected suffix has one to five rows and starts at position 32,768 or later. Other rows may run through ordinary attention. [Reader selection](../src/text/gqa_decode_long.cu#L703)<!-- prefix: "if (requests" --> · [Combine](../src/text/gqa/target_split.cu#L254)<!-- prefix: "if (splits =" --> | 32 splits. Partials use 399,360 B of the existing 496,640 B score buffer per row. [Scratch bound](../include/sglang/text/gqa/gqa_decode_long.cuh#L17)<!-- prefix: "static_asser" --> · [Buffer reuse](../src/text/gqa/verify_batch.cpp#L8)<!-- prefix: "PagedDecodeB" --> |
| Other long waves | Contiguous suffixes of one to five rows from position 32,768, including across 64-token chunks and 128-token pages. Ordinary packets run earlier and ineligible rows first. Append selected KV before one reader and combine per GQA layer. [Sequence collection](../src/text/gqa/verify_batch.cpp#L110)<!-- prefix: "int collect_" --> · [Guard and suffix selection](../src/text/gqa/verify_batch.cpp#L202)<!-- prefix: "bool try_gqa" --> | 16 splits, including TP1 single requests. |

Each request/KV-head/split block stages 64 tokens by TMA. Splits follow the first position, with the final split through the newest row and each row masking later keys. BF16 Tensor Cores form FP32 scores. PV accumulates the rounded BF16 weight and its rounded BF16 residual, which can change scores. The reader uses one BF16 K and two alternating V slots, refilled after their last consumer. Combine uses the reader’s split count and replays flagged heads in FP64. [Paged read](../src/text/gqa_decode_long.cu#L269)<!-- prefix: "__device__ v" --> · [Split bounds](../src/text/gqa_decode_long.cu#L556-L558)<!-- prefix: "const int to" --> · [Probability split](../src/text/gqa_decode_long.cu#L448)<!-- prefix: "const auto h" --> · [Operand slots](../src/text/gqa_decode_long.cu#L175)<!-- prefix: "struct align" --> · [Refill barriers](../src/text/gqa_decode_long.cu#L345)<!-- prefix: "constexpr in" --> · [Split combine](../src/text/gqa/target_split.cu#L264)<!-- prefix: "cudaError_t " -->

The reader remaps K/V producers for 16×5 TP2. Deep 8×5 also remaps with two KV heads per rank and every first position ≥131,072. The XOR changes thread assignment while preserving scales, visibility, layout, split boundaries and FP32 accumulation order. Other shapes retain the original mapping. [Producer assignment](../src/text/gqa_decode_long.cu#L309)<!-- prefix: "if constexpr" --> · [Shape selection](../src/text/gqa_decode_long.cu#L689)<!-- prefix: "bool deep_ra" -->

## Constraints and failure behavior

| Failure | Result |
| --- | --- |
| Clean shared-score OOM | Remember the failure. Long singleton steps use direct eager attention, and short replay stays eligible. Other allocation errors reach recovery. [Allocation fallback](../src/runtime/backend/cuda_resources.cpp#L491)<!-- prefix: "ExactGqaWork" --> |
| Recoverable graph rejection | Drain, destroy graph handles, disable replay for that width. Eager may use the retained scores. Teardown drains and destroys graphs before freeing the buffer. [Graph recovery](../src/runtime/backend/cuda_resources.cpp#L572)<!-- prefix: "const auto f" --> · [Teardown](../src/runtime/backend/cuda_resources.cpp#L234)<!-- prefix: "CleanupError" --> |
| 20-row shared reader decline | Clean decline or recognized optional setup/launch rejection falls back for this packet. Later packets may retry. A pending or distinct recorded CUDA error blocks fallback and stays recorded. Clear only a matching recorded status. The returned status wins on hard failure. [Rejection handling](../src/text/gqa_decode_split.cu#L780)<!-- prefix: "std::optiona" --> |
| Long-reader decline | Clean device, shared-memory or tensor-map setup decline uses packet split on already appended KV, before gating. Pending errors and failed setup, reader or combine calls propagate without retry. A recorded terminal status wins over a nonterminal returned status. [Long setup](../src/text/gqa_decode_long.cu#L755)<!-- prefix: "std::optiona" --> · [Fallback after append](../src/text/gqa/verify_batch.cpp#L241)<!-- prefix: "if (const au" --> · [Status precedence](../src/text/gqa_decode_long.cu#L680)<!-- prefix: "cudaError_t " --> |

## Verification

- Follow the [batch CUDA Graph runbook](../tests/batch/README.md) on a reserved B200. The singleton mode compares IDs and full BF16 scores against eager at slot 31. It exercises request replacement and width-local launch rejection. It requires replay at 5,120 and 5,121 visible tokens in the vector route. The `mxfp8-gqa-check` target compares score-backed and direct BF16 outputs on reversed pages. It also checks invalid arguments and bound replay. [Singleton fixture](../tests/batch/batch_graph_lanes.cpp#L226)<!-- prefix: "void singlet" --> · [Exact operator check](../tests/mxfp8_gqa_check.cu#L501)<!-- prefix: "void check_e" -->
- Follow the [long decode runbook](../tests/decoder/README.md#wide-long-context-decode). It covers split and combine geometry, shuffled pages, alignment, full context, FP64 recovery, and decode dispatch. Enable `--long-decode-check=y` and build `gqa-split-fp64-check`. Run it on the same reserved GPU for independent FP64 attention checks. For a changed reduction, use the [operator qualification policy](performance.md#operator-qualification). [FP64 check](../tests/decoder/gqa_split_fp64_check.cpp#L181)<!-- prefix: "int main() t" -->
- Run the [K4 crossing fixture](../tests/decoder/README.md#k4-boundary-crossings) and [mixed-request fixture](../tests/decoder/README.md#mixed-k4-verification) for long attention. Check reader and combine counts, output IDs, ordinary remainder, clean decline, and error propagation. Follow the [MTP verification runbook](mtp.md#verification) for acceptance cuts and continuation.

### TP2 8×5 shared-KV operator probe

On B200, check both first-position bounds, adjacent declines, and ineligible waves. The crossing case starts one request at 32,767. Its suffix starts at 32,768 and requires a 32-split reader and matching combine, with 36 rows on ordinary attention. Compare BF16 output and independent FP64 at fixed Q and KV. Time reader plus combine separately from enqueue. [Route cases](../tests/decoder/long_decode_routes.cpp#L352)<!-- prefix: "void check_m" --> · [Crossing assertion](../tests/decoder/long_decode_routes.cpp#L379)<!-- prefix: "expect(full," -->

### TP2 single-request split probe

On a B200, the isolated 32-split probe covered 15 cases and 595,968 output elements against independent FP64. Maximum absolute error was 0.000489123 and maximum normalized RMSE was 0.00168134. Scratch guards passed. At first position 203,776 with five rows, reader plus combine measured 353.551 µs at 16 splits and 188.096 µs at 32.

The ranked narrow and wide route checks passed. The production reader and combine passed 16 FP64 boundary and scratch guard cases on B200 covering 340,992 output values. Maximum absolute error was 0.00049132 and maximum normalized RMSE was 0.00167659.

### TP2 16×5 nine-split operator probe

On a B200, check launch order and geometry at the first-position bounds for 16×5. Require 16 splits for 79 rows and for sequences outside either bound, plus one ordinary row below the band. Cover nonconsecutive positions, graph bindings and constraints, and rerun the 8×5 coverage. Compare 9 and 18 splits against FP64 with guarded scratch. At first position 34,768, isolated reader plus combine took 263.0933 µs at 18 splits and 239.9573 µs at 9. The split count changes rounding. [Wide route cases](../tests/decoder/long_decode_routes.cpp#L324)<!-- prefix: "void check_m" --> · [Launch checks](../tests/decoder/long_decode_routes.cpp#L744)<!-- prefix: "static void " -->

### K/V producer remap probe

On two B200s, compare winners, acceptance, both rank states, and scoring-rank carry for 16×5 at positions 32,768–36,863 and 8×5 at positions starting at 131,072 or later. Check that narrower or shallower controls and deep singletons skip remapping.

### Ordered single-row attention checks

Build both binaries with the [long decode runbook](../tests/decoder/README.md#wide-long-context-decode). Set `serial_ids` to a whitespace-delimited file of at least 9,216 valid checkpoint token IDs. Keep the reserved B200 and the `out` and `checkpoint` settings from that runbook. For the checkpoint fixture, start an isolated [LMCache daemon](serving/lmcache.md#start-the-daemon) on the same GPU with namespace `sglang`.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/bin/long-decode-check" serial
CUDA_VISIBLE_DEVICES="$GPU_UUID" LMCACHE_SOCKET="$LMCACHE_SOCKET" \
    "$out/bin/long-decode-routes" "$checkpoint" --serial-prefill "$serial_ids"
```

The operator fixture compares direct tiled output against a single captured node while changing the device position. The checkpoint fixture primes 9,216 tokens and restores the exact suffix checkpoint at 8,192 (`9,216 − 1,024`). It checks a one-row tail and forces arena allocation failure for a three-row tail. Each attention row must run three ordered score/replay pairs and no tiled reader. A subsequent unforced request must recover. [Operator fixture](../tests/decoder/long_decode_check.cu#L470)<!-- prefix: "void check_s" --> · [Checkpoint fixture](../tests/decoder/long_decode_routes.cpp#L605)<!-- prefix: "void serial_" --> · [Launch counts](../tests/decoder/long_decode_routes.cpp#L74)<!-- prefix: "void check_s" -->

On a B200 with fixed Q and remapped KV pages, the chunked reader matched the full tiled BF16 row bitwise in both rank geometries. The operator probe covered five visible spans from 8,193 to 196,609 tokens per geometry. Sampled FP64 error was at most 0.000083808.
