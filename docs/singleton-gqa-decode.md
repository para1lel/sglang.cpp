# GQA decode and target verification

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| CUDA backend policy | Chooses the B200 split-KV policy or the generic SM100 path. | [Device policy](../src/runtime/backend/cuda_backend_policy.cpp#L17)<!-- prefix: "const bool b" --> |
| Batch executor | Selects direct decode lanes, binds scores and request state, and owns graph and scratch lifetimes. | [Batch selection](../src/decoder_batch.cpp#L373)<!-- prefix: "run = run_de" --> · [Graph dispatch](../src/text/decoder_graph.cpp#L83)<!-- prefix: "DecodeRun ru" --> · [Score allocation](../src/runtime/backend/cuda_resources.cpp#L498)<!-- prefix: "ExactGqaWork" --> |
| Target attention | Appends KV, reads each row's causal prefix, and gates the attention output before projection. | [Eager attention](../src/runtime/execution/decoder_forward_attention.cpp#L397)<!-- prefix: "if (batch.ta" --> · [Bound singleton](../src/runtime/execution/decoder_forward_attention.cpp#L273)<!-- prefix: "if (batch.ta" --> · [Bound verification](../src/runtime/execution/decoder_forward_attention.cpp#L293)<!-- prefix: "void attend_" --> |

## Runtime path

Regular B200 batch replay supports one eligible direct decode lane. The first eligible step runs eagerly. The graph binds the live position, page table, KV, and score workspace for the selected request. At width one it requires inline score capacity through 5,120 visible tokens unless a shared score buffer is available. B200 does not use that buffer. Positive MTP uses a [separate verification graph policy](mtp.md#target-verification-graph). The standalone serial generator uses eager decode if its request can enter the split window. [Replay policy](../src/runtime/execution/decoder_execution_policy.cpp#L21)<!-- prefix: "bool Decoder" --> · [Warmup and score check](../src/text/decoder_graph.cpp#L30)<!-- prefix: "bool replay_" --> · [Serial selection](../src/decoder.cpp#L715)<!-- prefix: "bool Decoder" -->

### Generic SM100 singleton

The generic policy enables narrow attention for one direct batch decode lane. With sufficient workspace the page-parallel reader covers 1,536–2,560 visible tokens. Other rows with inline scores use ordered score and replay kernels. Eager rows beyond inline capacity use the scratch-free reader unless they qualify for shared scores. [Narrow selection](../src/text/decoder_graph.cpp#L92)<!-- prefix: "first.narrow" --> · [Narrow window](../src/mxfp8_gqa.cu#L366)<!-- prefix: "constexpr in" --> · [Inline scores](../src/runtime/model/qwen_request.cpp#L198)<!-- prefix: "ExactGqaWork" --> · [Direct reader](../src/runtime/execution/decoder_forward_attention.cpp#L204)<!-- prefix: "return mxfp8" -->

From 32,768 through 36,864 visible tokens, the generic batch can bind a single shared score buffer for eager or graph decode. Every GQA cache and the page table must cover the live rows. The batch allocates 3,538,944 bytes on first use for 24 FP32 scores per position. Each bind rounds its stride to a 128-token page. GQA layers and later requests reuse the buffer in stream order. [Eligibility and stride](../src/text/decoder_graph.cpp#L9)<!-- prefix: "ExactGqaWork" --> · [Replay binding](../src/text/decoder_graph.cpp#L57)<!-- prefix: "if (lane.sin" --> · [Score and replay](../src/mxfp8_gqa.cu#L710)<!-- prefix: "cudaError_t " -->

### Scratch-free ordered attention

Restored singleton prefill and allocation-limited prefill can reach the same reader as eager decode without score scratch. The public attention call keeps its tiny reader when the query end is below eight tokens. At eight or more it stages 32 keys for two query heads at a time. It applies the causal mask and accumulates values in token order. A standalone serial graph whose reserved KV exceeds inline score capacity uses the same tiled reader with a device position. Replay loads that position on each launch. [Eager dispatch](../src/mxfp8_gqa.cu#L805)<!-- prefix: "cudaError_t " --> · [Serial graph dispatch](../src/runtime/execution/decoder_forward_attention.cpp#L212)<!-- prefix: "if (capacity" --> · [Ordered kernel](../src/text/gqa/ordered_prefill.cu#L11)<!-- prefix: "__global__ v" --> · [Tile geometry](../include/sglang/text/mxfp8_gqa_prefill_common.cuh#L40-L41)<!-- prefix: "inline const" -->

### Draft singleton reroute

Single scored lanes route through the batch entry; unscored draft seeding skips attention. On B200 a single lane now uses grouped online split GQA above 2,047 visible tokens; ineligible lanes keep the per-row routes. A geometry mismatch rejects the whole batch, and earlier rows may have already submitted. Target verification still selects winners from target scores and repairs draft state from verified rows. Approximate probabilities can change draft proposals. A paid W2 reroute measured 1,623.189917 to 1,454.878128 milliseconds against the pinned 1,420.734951; both native arms returned identical token streams, with one measured cohort each. The makespan delta includes changed verify-group composition. [Scored singleton](../src/decoder_mtp.cpp#L421)<!-- prefix: "const DraftA" --> · [Batch entry](../src/text/gqa/draft_attention.cu#L277)<!-- prefix: "cudaError_t " --> · [Draft attention](native-model-info.md#draft-attention)

### Grouped long draft reads

On B200, rank 0 in TP2 can group exactly 16 eager online draft rows into one split launch and one combine. Each row must start at zero-based position 32,768 or later and fit the 262,144-token context. The route requires full draft geometry, readable paged KV, aligned operands, and sufficient existing per-row partial scratch. It uses the same multistage arithmetic and combine as the four-row packets without allocating device buffers. Any eligibility decline or capture uses the four-row path. CUDA errors propagate. [Draft dispatch](../src/text/draft_projection.cpp#L179)<!-- prefix: "void DraftPr" --> · [Grouped eligibility](../src/text/gqa/draft_attention.cu#L221)<!-- prefix: "std::optiona" --> · [Multistage split](../src/text/gqa_decode_split.cu#L831)<!-- prefix: "cudaError_t " -->

A bounded two-B200 W4 replay matched all 3,200 IDs across cold and restored stages, with 16 restores per rank at 32,871 cached tokens. Excluding the first clean frame, the median CUDA span for four proposal steps fell from 2.671 to 2.494 ms. Use the full [serving campaign](performance.md#matched-k4-evidence) for W4 makespan. [Frozen bounded replay and receipt reference](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/singleton-gqa-decode.md#grouped-long-draft-reads)

## Split-KV target attention

B200 selects split-KV attention for eligible eager and bound text target rows from 2,048 through 262,144 visible tokens, including the current token. Bound kernels check the live position on device. Inside the window the reader decodes paged MXFP8 K/V to BF16 for Tensor Core attention, merges FP32 split partials, and rounds to BF16. Flagged heads use FP64 replay. Outside the window the target uses ordered attention. The split reductions can change scores and greedy IDs. [Window](../include/sglang/decoder_cuda.cuh#L320)<!-- prefix: "inline const" --> · [Bound route and fallback](../src/text/gqa/target_split.cu#L152)<!-- prefix: "cudaError_t " --> · [Combine and FP64 replay](../src/text/gqa/target_split.cu#L68)<!-- prefix: "__global__ v" -->

### Batched eager verification

The batched path requires vector attention, the full-context policy, and 2–20 rows inside the split window. It runs Q/K normalization and RoPE, KV append, split reads, and sigmoid gating in stream order. Each row keeps its workspace, KV page table, and causal position. Other groups run preprocessing, append, and gating per row. They can still group eligible reads. Bound replay uses its own dispatch. The [MTP verifier](mtp.md#runtime-path) owns acceptance and the committed KV prefix. [Eligibility and launch order](../src/text/gqa/verify_batch.cpp#L13)<!-- prefix: "bool try_gqa" --> · [Eager fallback](../src/runtime/execution/decoder_forward_attention.cpp#L441)<!-- prefix: "if (!try_gqa" -->

TP2 vector-backend lanes below the split range batch their exact-scores and replay passes into one eager kernel pair. A side-effect-free eligibility check gates the route, and constrained or non-vector batches keep the per-row passes. [Short verify](../src/text/gqa/draft_attention.cu#L411)<!-- prefix: "cudaError_t " --> · [Eligibility](../src/text/gqa/draft_attention.cu#L391)<!-- prefix: "bool decoder" -->

Under the full-context policy, ordinary eager launches can group 10–20 rows. All inputs must satisfy Tensor Core alignment, and each row must average at most ten 64-token chunks per split. This route uses one shared-memory copy stage. It transposes QK across 16 keys and eight head slots (six active and two padded), then restores token order before masking and softmax. Other valid eager groups use at most five rows per launch. Narrow, singleton, and bound launches use two stages. Each row reserves scratch for up to 16 splits. [Grouping](../src/text/gqa/target_split.cu#L128)<!-- prefix: "int eager_ta" --> · [Launch geometry](../src/text/gqa_decode_split.cu#L824)<!-- prefix: "cudaError_t " --> · [Transposed scores](../src/text/gqa_decode_split.cu#L290)<!-- prefix: "__device__ v" -->

### Shared KV reads for long K4 packets

A full 20-row eager packet can share KV reads across four distinct cache and page-table identities. Each request contributes five consecutive zero-based positions within one 64-token chunk. Positions must be at least 32,768 and below 40,960. Each row needs readable KV, valid scratch, and Tensor Core alignment. One CTA per request, KV head, and split stages K/V and scales for its five rows. Each row masks its own tail and writes independent split partials. Other packets use ordinary split attention. [Packet dispatch](../src/text/gqa/target_split.cu#L190)<!-- prefix: "if (count ==" --> · [Eligibility](../src/text/gqa_decode_split.cu#L728)<!-- prefix: "bool make_re" --> · [Causal operands](../src/text/gqa_decode_split.cu#L699)<!-- prefix: "if (chunk \u003c " -->

### Tensor Core reads across long K4 packets

The proposed W3 route selects eight complete five-row groups with distinct page tables and 12 query heads and two KV heads per TP2 rank. Each group's first zero-based position must be at least 8,191. Its fifth row must average at most ten 64-token chunks per split under the existing 20-row split packing rule. With at most 16 splits, the first position can reach 10,235 and the fifth can reach 10,239 (10,240 visible tokens). Groups can start at different positions and cross chunk or page boundaries. The route requires eager vector attention and the full-context split policy. Every row must be unconstrained, have no graph binding, and pass KV readability, alignment, and buffer checks. A wave that misses the W3 guard uses the existing selection rules below. [W3 guard](../src/text/gqa/verify_batch.cpp#L138)<!-- prefix: "bool is_rank" --> · [Split packing rule](../src/text/gqa/geometry.cpp#L24)<!-- prefix: "bool support" -->

For other shapes, the full-context eager Tensor Core verifier selects contiguous suffixes of one to five rows from positions 32,768 onward. Selected suffixes can cross a 64-token chunk or 128-token page boundary. Duplicate page tables in the wave prevent selection. Every selected row must be unconstrained and have readable KV and valid, aligned buffers. Earlier rows and ineligible requests run first in their original packets. The verifier then appends selected KV and submits one long reader and combine per GQA layer. [Sequence collection](../src/text/gqa/verify_batch.cpp#L105)<!-- prefix: "int collect_" --> · [Guard and suffix selection](../src/text/gqa/verify_batch.cpp#L155)<!-- prefix: "bool try_gqa" -->

One block per request, KV head, and split stages each 64-token chunk by TMA for up to five rows. Splits follow the first selected position and the final split extends through the newest row. Each row masks later keys. BF16 Tensor Core multiplies form FP32 scores. The value multiply accumulates the rounded BF16 weight and its rounded BF16 residual. This arithmetic can change scores relative to the ordinary split reader. [Paged read](../src/text/gqa_decode_long.cu#L273)<!-- prefix: "__device__ v" --> · [Split bounds](../src/text/gqa_decode_long.cu#L563-L565)<!-- prefix: "const int to" --> · [Probability split](../src/text/gqa_decode_long.cu#L452)<!-- prefix: "const auto h" -->

The reader uses one expanded BF16 K slot and alternates between two expanded BF16 V slots. It refills a slot after the last QK or PV consumer of that slot finishes. The combine uses the reader's split count. Flagged heads still replay in FP64. [Operand slots](../src/text/gqa_decode_long.cu#L179)<!-- prefix: "struct align" --> · [Refill barriers](../src/text/gqa_decode_long.cu#L349)<!-- prefix: "constexpr in" --> · [Split combine](../src/text/gqa/target_split.cu#L217)<!-- prefix: "cudaError_t " -->

On TP2 the W4 verifier uses 18 splits for 16 complete five-row sequences with 12 query heads and two KV heads per rank. The first row of every sequence must have 32,769–36,864 visible tokens (zero-based positions 32,768–36,863). Other eligible long groups use 16 splits. At 18 splits, the partials occupy 224,640 bytes per row and leave 174,720 bytes in the existing scratch allocation. [W4 dispatch](../src/text/gqa_decode_long.cu#L669)<!-- prefix: "bool rank_wi" --> · [Scratch bound](../include/sglang/text/gqa/gqa_decode_long.cuh#L14)<!-- prefix: "static_asser" -->

The native reader remaps K/V producer rows for that W4 shape. It also remaps eight complete five-row sequences with two KV heads per rank when every first zero-based position is at least 131,072. The producer XORs its starting row with the low bit of the channel group. This changes which thread reads each row while preserving scales, visibility, destination layout, split boundaries, and FP32 accumulation order. Other shapes keep the original producer mapping. [Producer assignment](../src/text/gqa_decode_long.cu#L313)<!-- prefix: "if constexpr" --> · [Shape selection](../src/text/gqa_decode_long.cu#L670)<!-- prefix: "bool deep_ra" -->

## Constraints and failure behavior

A clean shared-score OOM makes the generic batch remember the allocation failure. Long singleton steps use direct eager attention, while short replay remains eligible. Other allocation errors reach batch recovery. A recoverable graph rejection drains the stream, destroys its graph handles, and disables replay for that width. Eager decode can still use the retained score buffer. Batch cleanup drains the stream and destroys graphs before freeing it. [Allocation fallback](../src/runtime/backend/cuda_resources.cpp#L498)<!-- prefix: "ExactGqaWork" --> · [Graph recovery](../src/runtime/backend/cuda_resources.cpp#L557)<!-- prefix: "const auto f" --> · [Teardown](../src/runtime/backend/cuda_resources.cpp#L246)<!-- prefix: "CleanupError" -->

For the 20-row shared reader, a clean decline or a recognized optional setup or launch rejection returns that packet to the ordinary split route. A later packet can try again. A preexisting or distinct recorded CUDA error prevents fallback and remains recorded. The code clears only a recorded status matching the failed call and gives the returned status precedence on hard failure. [Rejection handling](../src/text/gqa_decode_split.cu#L767)<!-- prefix: "std::optiona" -->

A clean device, shared-memory, or tensor-map setup decline sends the selected rows to the existing packet split dispatch. That dispatch reads their appended KV before gating. Pending errors and failed setup, reader, or combine calls propagate without retry. A recorded terminal CUDA error takes precedence over a nonterminal returned status. [Long setup](../src/text/gqa_decode_long.cu#L666)<!-- prefix: "std::optiona" --> · [Fallback after append](../src/text/gqa/verify_batch.cpp#L227)<!-- prefix: "if (const au" --> · [Status precedence](../src/text/gqa_decode_long.cu#L659)<!-- prefix: "cudaError_t " -->

## Verification

- Follow the [batch CUDA Graph runbook](../tests/batch/README.md) on a reserved B200. The singleton mode compares IDs and full BF16 scores against eager at slot 31. It exercises request replacement and width-local launch rejection. It requires replay at 5,120 visible tokens and eager execution at 5,121. The `mxfp8-gqa-check` operator target compares score-backed and direct BF16 outputs on reversed pages. It also checks invalid argument guards and bound replay. [Singleton fixture](../tests/batch/batch_graph_lanes.cpp#L214)<!-- prefix: "void singlet" --> · [Exact operator check](../tests/mxfp8_gqa_check.cu#L541)<!-- prefix: "void check_e" -->
- Follow the [long decode runbook](../tests/decoder/README.md#wide-long-context-decode). It covers split and combine geometry, shuffled pages, alignment, full context, FP64 recovery, and decode dispatch. Enable `--long-decode-check=y` and build `gqa-split-fp64-check`. Run it on the same reserved GPU for independent FP64 attention checks. For a changed reduction, use the [operator qualification policy](performance.md#operator-qualification). [FP64 check](../tests/decoder/gqa_split_fp64_check.cpp#L181)<!-- prefix: "int main() t" -->
- Run the [K4 crossing fixture](../tests/decoder/README.md#k4-boundary-crossings) and [mixed-request fixture](../tests/decoder/README.md#mixed-k4-verification) for long attention. Check reader and combine counts, output IDs, ordinary remainder, clean decline, and error propagation. Follow the [MTP verification runbook](mtp.md#verification) for acceptance cuts and continuation.

### W3 shared-KV operator probe

Check both position bounds, adjacent declines, ineligible waves, and the 32,768-token suffix route. Compare BF16 output and independent FP64 at fixed Q and KV. Time reader plus combine separately from enqueue. [Route cases](../tests/decoder/long_decode_routes.cpp#L332)<!-- prefix: "void check_m" --> · [Frozen B200 shapes, errors, and timings](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/singleton-gqa-decode.md#w3-shared-kv-operator-probe)

### W4 18-split operator probe

The focused B200 route test checked reader and combine order and launch geometry at both W4 bounds. It verified 16-split selection for 79 rows and for a sequence just below or above the band, with one ordinary row below the band. It also checked nonconsecutive positions, graph binding, and a constrained sequence. The W3 route test passed on the same build. [W4 route cases](../tests/decoder/long_decode_routes.cpp#L304)<!-- prefix: "void check_m" --> · [Launch checks](../tests/decoder/long_decode_routes.cpp#L707)<!-- prefix: "static void " -->

Compare 18- and 16-split readers against FP64 at 32,769, 34,817, and 36,864 visible tokens. Require every row to write the 18-split tail and preserve scratch guards. [Frozen numerical and reader-plus-combine timing receipts](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/singleton-gqa-decode.md#w4-18-split-operator-probe)

### W4 and W5 producer remap probe

Compare W4 M80 and W5 M40 winners, acceptance, both rank states, and scoring-rank carry. Require zero remapped calls for W1–W3 and deep singletons. [Frozen two-B200 paired verification waves](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/singleton-gqa-decode.md#w4-and-w5-producer-remap-probe)

### Scratch-free attention checks

Build both binaries with the [long decode runbook](../tests/decoder/README.md#wide-long-context-decode). Set `serial_ids` to a whitespace-delimited file of at least 8,195 valid checkpoint token IDs. Keep the reserved B200 and the `out` and `checkpoint` settings from that runbook. For the checkpoint fixture, start an isolated [LMCache daemon](serving/lmcache.md#start-the-daemon) on the same GPU with namespace `sglang`.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/bin/long-decode-check" serial
CUDA_VISIBLE_DEVICES="$GPU_UUID" LMCACHE_SOCKET="$LMCACHE_SOCKET" \
    "$out/bin/long-decode-routes" "$checkpoint" --serial-prefill "$serial_ids"
```

The operator fixture compares eager output against a single captured node while changing the device position. The checkpoint fixture checks restored singleton prefill and a three-row tail after a real arena allocation failure. It counts tiled reader launches and checks that a subsequent unforced request recovers. [Operator fixture](../tests/decoder/long_decode_check.cu#L470)<!-- prefix: "void check_s" --> · [Checkpoint fixture](../tests/decoder/long_decode_routes.cpp#L582)<!-- prefix: "void serial_" -->

The earlier packed-runtime notes and historical probes remain in the [archived guide](https://github.com/para1lel/sglang.cpp/blob/2676f3320cbfbe0f09f5d1193f67ec5abcf67639/docs/singleton-gqa-decode.md).
