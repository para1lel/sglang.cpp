# GQA decode and target verification

## Responsibilities and sources of truth

The backend picks B200 split-KV or generic SM100. The batch executor owns replay and scratch. Target attention owns KV append, causal reads, and output gating. [Device policy](../src/runtime/backend/cuda_backend_policy.cpp#L17)<!-- prefix: "const bool b" --> · [Batch selection](../src/decoder_batch.cpp#L488)<!-- prefix: "run = run_de" --> · [Graph dispatch](../src/text/decoder_graph.cpp#L85)<!-- prefix: "DecodeRun ru" --> · [Score allocation](../src/runtime/backend/cuda_resources.cpp#L491)<!-- prefix: "ExactGqaWork" --> · [Eager attention](../src/runtime/execution/decoder_forward_attention.cpp#L336)<!-- prefix: "if (batch.ta" --> · [Bound singleton](../src/runtime/execution/decoder_forward_attention.cpp#L212)<!-- prefix: "if (batch.ta" --> · [Bound verification](../src/runtime/execution/decoder_forward_attention.cpp#L232)<!-- prefix: "void attend_" -->

## Runtime path

B200 batch replay binds one direct lane’s live position, page table, KV, and inline scores after an eager warmup. It covers up to 5,120 visible tokens. B200 skips shared scores. Positive MTP uses the [separate verification graph policy](mtp.md#target-verification-graph). Serial generation stays eager when its request can enter the split window. [Replay policy](../src/runtime/execution/decoder_execution_policy.cpp#L21)<!-- prefix: "bool Decoder" --> · [Warmup and score check](../src/text/decoder_graph.cpp#L30)<!-- prefix: "bool replay_" --> · [Serial selection](../src/decoder.cpp#L714)<!-- prefix: "bool Decoder" -->

### Generic SM100 singleton

| Generic one-lane route | Guard and workspace | Sources |
| --- | --- | --- |
| Page-parallel narrow | 1,536–2,560 visible tokens with sufficient workspace. | [Narrow selection](../src/text/decoder_graph.cpp#L94)<!-- prefix: "first.narrow" --> · [Narrow window](../src/mxfp8_gqa.cu#L366)<!-- prefix: "constexpr in" --> |
| Ordered scores and replay | Other rows with inline scores. Beyond that capacity, eager uses the scratch-free reader unless shared scores qualify. | [Inline scores](../src/runtime/model/qwen_request.cpp#L256)<!-- prefix: "ExactGqaWork" --> · [Direct reader](../src/runtime/execution/decoder_forward_attention.cpp#L143)<!-- prefix: "return mxfp8" --> |
| Shared scores, eager or replay | 32,768–36,864 visible tokens. Every GQA cache and page table must cover live rows. First use allocates 3,538,944 bytes (24 FP32 scores per position); each bind rounds stride to 128 tokens. Layers and requests reuse it in stream order. | [Eligibility and stride](../src/text/decoder_graph.cpp#L9)<!-- prefix: "ExactGqaWork" --> · [Replay binding](../src/text/decoder_graph.cpp#L57)<!-- prefix: "if (lane.sin" --> · [Score and replay](../src/mxfp8_gqa.cu#L710)<!-- prefix: "cudaError_t " --> |

<a id="scratch-free-ordered-attention"></a>

### Ordered single-row attention

On B200, eager vector prefill can use the same chunked ordered reader as draft attention. The visible span must exceed inline score capacity and fit the model context. Existing score scratch must hold one 4,096-token tile plus the FP32 carry. Each tile scores first and then replays in token order. The last tile writes the BF16 output. Invalid inputs or CUDA errors after selection propagate. [Eager route](../src/runtime/execution/decoder_forward_attention.cpp#L136)<!-- prefix: "if (mode == " --> · [Eligibility](../src/text/gqa/draft_attention.cu#L249)<!-- prefix: "decoder::try" --> · [Score and replay order](../src/text/gqa/draft_attention.cu#L166)<!-- prefix: "cudaError_t " -->

When that route declines, eager prefill can use the scratch-free reader. It uses the tiny kernel below eight visible tokens, then stages 32 keys for two query heads with causal masking and token-order accumulation. Serial graphs with reserved KV beyond inline capacity use the tiled reader and reload device position on replay. [Tiled dispatch](../src/mxfp8_gqa.cu#L805)<!-- prefix: "cudaError_t " --> · [Serial graph dispatch](../src/runtime/execution/decoder_forward_attention.cpp#L151)<!-- prefix: "if (capacity" --> · [Ordered kernel](../src/text/gqa/ordered_prefill.cu#L12)<!-- prefix: "__global__ v" --> · [Tile geometry](../include/sglang/text/mxfp8_gqa_prefill_common.cuh#L40-L41)<!-- prefix: "inline const" -->

### Draft singleton reroute

Scored draft singletons use the batch entry. Unscored seed skips attention. On B200, eligible lanes use grouped online split GQA at 2,048 or more visible tokens; other lanes stay per row. A geometry mismatch rejects the batch after possible earlier submission. Target scores choose winners and verified rows repair draft state, while approximate probabilities can change draft proposals. [Scored singleton](../src/decoder_mtp.cpp#L443)<!-- prefix: "const DraftA" --> · [Batch entry](../src/text/gqa/draft_attention.cu#L294)<!-- prefix: "cudaError_t " --> · [Draft attention](native-model-info.md#draft-attention) · [Frozen W2 reroute measurement](https://github.com/para1lel/sglang.cpp/blob/57ac5fbbdf20a5a80c5bc079b49073a44c06680f/docs/singleton-gqa-decode.md#draft-singleton-reroute)

### Grouped long draft reads

On B200 TP2, rank 0 groups exactly 16 eager online draft rows into one split launch and combine. All zero-based positions are at least 32,768 and within the 262,144-token context. It needs full draft geometry, readable pages, alignment, and existing per-row scratch. Capture or a clean decline uses the four-row path with the same multistage arithmetic and no new device buffers. CUDA errors propagate. [Draft dispatch](../src/text/draft_projection.cpp#L179)<!-- prefix: "void DraftPr" --> · [Grouped eligibility](../src/text/gqa/draft_attention.cu#L223)<!-- prefix: "std::optiona" --> · [Multistage split](../src/text/gqa_decode_split.cu#L831)<!-- prefix: "cudaError_t " -->

A bounded W4 replay matched all 3,200 IDs and made 16 restores per rank at 32,871 cached tokens. The median CUDA span for four proposal steps was 2.671→2.494 ms after excluding the first clean frame. See [serving campaign](performance.md#matched-k4-evidence) for makespan. [Frozen bounded replay and receipt reference](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/singleton-gqa-decode.md#grouped-long-draft-reads)

## Split-KV target attention

B200 text target rows use split-KV for 2,048–262,144 visible tokens, including the current token, with ordered attention outside that range. Bound replay tests live position on device. Split-KV expands paged MXFP8 to BF16, merges FP32 partials, rounds to BF16, and replays flagged heads in FP64. The reductions can change scores and greedy IDs. [Lower bound](../include/sglang/decoder_cuda.cuh#L325)<!-- prefix: "inline const" --> · [B200 upper bound](../src/runtime/backend/cuda_backend_policy.cpp#L27)<!-- prefix: "backend.targ" --> · [Bound route and fallback](../src/text/gqa/target_split.cu#L152)<!-- prefix: "cudaError_t " --> · [Combine and FP64 replay](../src/text/gqa/target_split.cu#L68)<!-- prefix: "__global__ v" -->

### Batched eager verification

| Eager verify route | Eligibility, grouping, and fallback |
| --- | --- |
| Fused preprocessing through gating | Vector attention, full-context policy, 2–20 rows in the split window. Each row retains workspace, pages and position. Other groups do Q/K norm and RoPE, append and gating per row but can share reads. Bound replay dispatches separately. [Eligibility and launch order](../src/text/gqa/verify_batch.cpp#L13)<!-- prefix: "bool try_gqa" --> · [Eager fallback](../src/runtime/execution/decoder_forward_attention.cpp#L380)<!-- prefix: "if (!try_gqa" --> |
| Short exact scores | TP2 vector lanes below the split range use one score/replay kernel pair. A side-effect-free eligibility check leaves constrained or non-vector groups per row. [Short verify](../src/text/gqa/draft_attention.cu#L428)<!-- prefix: "cudaError_t " --> · [Eligibility](../src/text/gqa/draft_attention.cu#L408)<!-- prefix: "bool decoder" --> |
| Ordinary split launches | With full-context policy, alignment, and at most ten 64-token chunks per split on average per row, group 10–20 rows in one stage. QK transposes 16 keys across eight slots (six active) and restores token order before masking and softmax. Other eligible groups use up to five rows. Narrow, singleton, and bound launches use two stages. Scratch fits 16 splits per row. [Grouping](../src/text/gqa/target_split.cu#L128)<!-- prefix: "int eager_ta" --> · [Launch geometry](../src/text/gqa_decode_split.cu#L824)<!-- prefix: "cudaError_t " --> · [Transposed scores](../src/text/gqa_decode_split.cu#L290)<!-- prefix: "__device__ v" --> |

The [MTP verifier](mtp.md#runtime-path) owns acceptance and the committed KV prefix.

### Shared KV reads for long K4 packets

Under the full-context policy, a 20-row eager packet can share reads across four distinct caches and page tables, with five consecutive positions in one 64-token chunk per request. Every zero-based position is 32,768–40,959, with readable KV, alignment and valid scratch. Each request/KV-head/split CTA shares staged K/V and scales across five rows, with independent masks and partials. Other packets use ordinary split. [Packet dispatch](../src/text/gqa/target_split.cu#L190)<!-- prefix: "if (count ==" --> · [Eligibility](../src/text/gqa_decode_split.cu#L728)<!-- prefix: "bool make_re" --> · [Causal operands](../src/text/gqa_decode_split.cu#L699)<!-- prefix: "if (chunk \u003c " -->

### Tensor Core reads across long K4 packets

The long verifier needs eager vector attention and the full-context policy. Selected rows must be unconstrained, unbound, aligned, and have readable KV and valid buffers. Duplicate page tables prevent selection. Positions below are zero-based.

| Shape | Selection | Splits and scratch |
| --- | --- | --- |
| W3 TP2, 8×5, 12 Q and 2 KV heads per rank | Distinct page tables. First position 8,191–10,235. The fifth must average ≤10 64-token chunks per split by the 20-row packing rule, so it reaches at most 10,239 (10,240 visible). Starts can differ and cross chunk/page boundaries. Nonmatching shapes can try the suffix route. Once W3 is selected, any ineligible row returns the round to ordinary packets. [W3 guard](../src/text/gqa/verify_batch.cpp#L138)<!-- prefix: "bool is_rank" --> · [Split packing rule](../src/text/gqa/geometry.cpp#L24)<!-- prefix: "bool support" --> | Up to 16 splits. |
| W4 TP2, 16×5, 12 Q and 2 KV heads per rank | Every first position 32,768–36,863 (32,769–36,864 visible). [W4 dispatch](../src/text/gqa_decode_long.cu#L669)<!-- prefix: "bool rank_wi" --> · [Scratch bound](../include/sglang/text/gqa/gqa_decode_long.cuh#L14)<!-- prefix: "static_asser" --> | 18 splits. Partials use 224,640 B per row, leaving 174,720 B. |
| Other long waves | Contiguous suffixes of 1–5 rows from position 32,768, including across 64-token chunks and 128-token pages. Ordinary packets run earlier and ineligible rows first. Append selected KV before one reader and combine per GQA layer. [Sequence collection](../src/text/gqa/verify_batch.cpp#L105)<!-- prefix: "int collect_" --> · [Guard and suffix selection](../src/text/gqa/verify_batch.cpp#L155)<!-- prefix: "bool try_gqa" --> | 16 splits. |

Each request/KV-head/split block stages 64 tokens by TMA. Splits follow the first position, with the final split through the newest row and each row masking later keys. BF16 Tensor Cores form FP32 scores. PV accumulates the rounded BF16 weight and its rounded BF16 residual, which can change scores. The reader uses one BF16 K and two alternating V slots, refilled after their last consumer. Combine uses the reader’s split count and replays flagged heads in FP64. [Paged read](../src/text/gqa_decode_long.cu#L273)<!-- prefix: "__device__ v" --> · [Split bounds](../src/text/gqa_decode_long.cu#L563-L565)<!-- prefix: "const int to" --> · [Probability split](../src/text/gqa_decode_long.cu#L452)<!-- prefix: "const auto h" --> · [Operand slots](../src/text/gqa_decode_long.cu#L179)<!-- prefix: "struct align" --> · [Refill barriers](../src/text/gqa_decode_long.cu#L349)<!-- prefix: "constexpr in" --> · [Split combine](../src/text/gqa/target_split.cu#L217)<!-- prefix: "cudaError_t " -->

The reader remaps K/V producers for W4 16×5. Deep 8×5 also remaps with two KV heads per rank and every first position ≥131,072. The XOR changes thread assignment while preserving scales, visibility, layout, split boundaries and FP32 accumulation order. Other shapes retain the original mapping. [Producer assignment](../src/text/gqa_decode_long.cu#L313)<!-- prefix: "if constexpr" --> · [Shape selection](../src/text/gqa_decode_long.cu#L670)<!-- prefix: "bool deep_ra" -->

## Constraints and failure behavior

| Failure | Result |
| --- | --- |
| Clean shared-score OOM | Remember the failure. Long singleton steps use direct eager attention, and short replay stays eligible. Other allocation errors reach recovery. [Allocation fallback](../src/runtime/backend/cuda_resources.cpp#L491)<!-- prefix: "ExactGqaWork" --> |
| Recoverable graph rejection | Drain, destroy graph handles, disable replay for that width. Eager may use the retained scores. Teardown drains and destroys graphs before freeing the buffer. [Graph recovery](../src/runtime/backend/cuda_resources.cpp#L550)<!-- prefix: "const auto f" --> · [Teardown](../src/runtime/backend/cuda_resources.cpp#L234)<!-- prefix: "CleanupError" --> |
| 20-row shared reader decline | Clean decline or recognized optional setup/launch rejection falls back for this packet. Later packets may retry. A pending or distinct recorded CUDA error blocks fallback and stays recorded. Clear only a matching recorded status. The returned status wins on hard failure. [Rejection handling](../src/text/gqa_decode_split.cu#L767)<!-- prefix: "std::optiona" --> |
| Long-reader decline | Clean device, shared-memory or tensor-map setup decline uses packet split on already appended KV, before gating. Pending errors and failed setup, reader or combine calls propagate without retry. A recorded terminal status wins over a nonterminal returned status. [Long setup](../src/text/gqa_decode_long.cu#L666)<!-- prefix: "std::optiona" --> · [Fallback after append](../src/text/gqa/verify_batch.cpp#L227)<!-- prefix: "if (const au" --> · [Status precedence](../src/text/gqa_decode_long.cu#L659)<!-- prefix: "cudaError_t " --> |

## Verification

- Follow the [batch CUDA Graph runbook](../tests/batch/README.md) on a reserved B200. The singleton mode compares IDs and full BF16 scores against eager at slot 31. It exercises request replacement and width-local launch rejection. It requires replay at 5,120 visible tokens and eager execution at 5,121. The `mxfp8-gqa-check` operator target compares score-backed and direct BF16 outputs on reversed pages. It also checks invalid argument guards and bound replay. [Singleton fixture](../tests/batch/batch_graph_lanes.cpp#L214)<!-- prefix: "void singlet" --> · [Exact operator check](../tests/mxfp8_gqa_check.cu#L540)<!-- prefix: "void check_e" -->
- Follow the [long decode runbook](../tests/decoder/README.md#wide-long-context-decode). It covers split and combine geometry, shuffled pages, alignment, full context, FP64 recovery, and decode dispatch. Enable `--long-decode-check=y` and build `gqa-split-fp64-check`. Run it on the same reserved GPU for independent FP64 attention checks. For a changed reduction, use the [operator qualification policy](performance.md#operator-qualification). [FP64 check](../tests/decoder/gqa_split_fp64_check.cpp#L181)<!-- prefix: "int main() t" -->
- Run the [K4 crossing fixture](../tests/decoder/README.md#k4-boundary-crossings) and [mixed-request fixture](../tests/decoder/README.md#mixed-k4-verification) for long attention. Check reader and combine counts, output IDs, ordinary remainder, clean decline, and error propagation. Follow the [MTP verification runbook](mtp.md#verification) for acceptance cuts and continuation.

### W3 shared-KV operator probe

Check both position bounds, adjacent declines, ineligible waves, and the 32,768-token suffix route. Compare BF16 output and independent FP64 at fixed Q and KV. Time reader plus combine separately from enqueue. [Route cases](../tests/decoder/long_decode_routes.cpp#L352)<!-- prefix: "void check_m" --> · [Frozen B200 shapes, errors, and timings](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/singleton-gqa-decode.md#w3-shared-kv-operator-probe)

### W4 18-split operator probe

On a B200, check reader/combine order and geometry at both W4 bounds. Require 16 splits for 79 rows and sequences outside either bound, plus one ordinary row below the band. Cover nonconsecutive positions, graph bindings and constraints, and rerun W3. Compare 18 and 16 splits against FP64 at 32,769, 34,817 and 36,864 visible tokens. Require every row to write the 18-split tail with intact scratch guards. [W4 route cases](../tests/decoder/long_decode_routes.cpp#L324)<!-- prefix: "void check_m" --> · [Launch checks](../tests/decoder/long_decode_routes.cpp#L743)<!-- prefix: "static void " --> · [Frozen numerical and reader-plus-combine timing receipts](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/singleton-gqa-decode.md#w4-18-split-operator-probe)

### W4 and W5 producer remap probe

Compare W4 M80 and W5 M40 winners, acceptance, both rank states, and scoring-rank carry. Require zero remapped calls for W1–W3 and deep singletons. [Frozen two-B200 paired verification waves](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/singleton-gqa-decode.md#w4-and-w5-producer-remap-probe)

### Ordered single-row attention checks

Build both binaries with the [long decode runbook](../tests/decoder/README.md#wide-long-context-decode). Set `serial_ids` to a whitespace-delimited file of at least 9,216 valid checkpoint token IDs. Keep the reserved B200 and the `out` and `checkpoint` settings from that runbook. For the checkpoint fixture, start an isolated [LMCache daemon](serving/lmcache.md#start-the-daemon) on the same GPU with namespace `sglang`.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/bin/long-decode-check" serial
CUDA_VISIBLE_DEVICES="$GPU_UUID" LMCACHE_SOCKET="$LMCACHE_SOCKET" \
    "$out/bin/long-decode-routes" "$checkpoint" --serial-prefill "$serial_ids"
```

The operator fixture compares direct tiled output against a single captured node while changing the device position. The checkpoint fixture primes 9,216 tokens and restores the exact suffix checkpoint at 8,192 (`9,216 − 1,024`). It checks a one-row tail and forces arena allocation failure for a three-row tail. Each attention row must run three ordered score/replay pairs and no tiled reader. A subsequent unforced request must recover. [Operator fixture](../tests/decoder/long_decode_check.cu#L470)<!-- prefix: "void check_s" --> · [Checkpoint fixture](../tests/decoder/long_decode_routes.cpp#L604)<!-- prefix: "void serial_" --> · [Launch counts](../tests/decoder/long_decode_routes.cpp#L74)<!-- prefix: "void check_s" -->

With fixed Q and remapped KV pages, the chunked reader matched the full tiled BF16 row bitwise in both rank geometries. The operator probe covered five visible spans from 8,193 to 196,609 tokens per geometry. Sampled FP64 error was at most 0.000083808. See `artifacts/pool-w6/hotpool/singleton.2C7iOVnI/results.log`.

The earlier packed-runtime notes and historical probes remain in the [archived guide](https://github.com/para1lel/sglang.cpp/blob/2676f3320cbfbe0f09f5d1193f67ec5abcf67639/docs/singleton-gqa-decode.md).
