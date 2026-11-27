# Native MXFP8 flash prefill

## Responsibilities and sources of truth

The decoder selects tiles and appends uncovered KV. The native dispatcher chooses Flash, ordered or protected attention. [Policy](../src/decoder_prefill.cpp#L613)<!-- prefix: "const bool t" --> · [Attention call](../src/runtime/execution/decoder_prefill_tiles.cpp#L224)<!-- prefix: "const auto s" --> · [Dispatch](../src/text/gqa/ordered_prefill.cu#L292)<!-- prefix: "cudaError_t " --> · [Paired attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L472)<!-- prefix: "__global__ v" --> · [Single-CTA attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L366)<!-- prefix: "__global__ v" --> · [Split merge](../src/text/mxfp8_gqa_prefill_tmem.cu#L569)<!-- prefix: "__global__ v" --> · [Serial prefill](text-serving.md#serial-prefill) · [Batch planner](batch-graphs.md#runtime-path)

## Runtime path

Eligible B200 prefill uses the native Flash dispatcher. The full-tile path checks the zero-based start, buffer alignment, and device resources. Restored prompts, media, and scored requests use the same planner. Each lane retains its input and scoring rules.

| Automatic single-lane TMEM route | Start position for 1,024 query rows |
| --- | --- |
| [Raw paired ordered](../src/text/gqa/ordered_prefill.cu#L312)<!-- prefix: "if (rows == " --> | 0 |
| [Staged or raw paired ordered](../src/text/gqa/tmem_staged.cu#L562)<!-- prefix: "if (position" --> | 1,024–7,168, at 1,024-token boundaries |
| Staged or raw paired ordered | 8,192–31,744, at 1,024-token boundaries or inside a 128-token page |
| [Raw paired ordered](../src/text/gqa/ordered_prefill.cu#L324)<!-- prefix: "if (default_" --> | 32,768–130,048, at 1,024-token boundaries |

| Other automatic path | Selection |
| --- | --- |
| Ordered partial TMEM | 148-SM SM100 before regular Flash, both scale bases aligned to 8 B. Try 144–255 rows at 8,192–32,768 visible or 256–512 rows with start ≥8,192 and visibility ≤32,768. Try [grouped attention](ordered-gqa-prefill.md#grouped-partial-grants) before per-lane dispatch. See [long serial TP2 tiles](ordered-gqa-prefill.md#long-serial-tp2-tiles) above that range. [Partial TMEM selection](../src/text/gqa/ordered_prefill.cu#L302)<!-- prefix: "if ((ragged " --> |
| Matched whole tiles | [whole-tile MTP projection group](native-prefill-workspace.md#projection-pairs-for-mtp) with four 1,024-row tiles at matching aligned starts 1,024–130,048. Stage four lanes through 32,768 visible, two through 65,536, one through 131,072. One predecode and reader per subgroup. |
| Mixed whole tiles | Pair distinct text owners with the next eligible matching aligned start at 8,192–64,512. Unpaired tiles can stage at 32,768–130,048. A declined pair tries singleton, then lane-local. Lanes append uncovered KV and retain their own pages, masks and slices. Submitted errors propagate. [Owner and position checks](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L333)<!-- prefix: "std::optiona" --> · [Paired read](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L375)<!-- prefix: "if (const au" --> · [Lane fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L62)<!-- prefix: "read_paged_s" --> |
| Regular Flash | Causal 64-key tiles, BF16 products, FP32 accumulators and online normalization. Overlap KV loads and mask partial blocks. At start 1,024 a full tile tries 88 rows/CTA. Aligned partials starting ≥4,096 use 32/48 rows per CTA if one SM wave fits, or 88 if it fits and 64 needs two waves. At 148 SMs the respective query-row bands are 144–384, 385–576 and 769–1,023. [Regular Flash](../src/mxfp8_gqa_prefill.cu#L403)<!-- prefix: "__global__ v" --> · [Partial dispatch](../src/mxfp8_gqa_prefill.cu#L499)<!-- prefix: "if (aligned_" --> |

### Long-tile TMEM pipeline

TMEM needs `sm_100a` and regular Flash uses `sm_100`. TMEM uses TMA, FP32 value accumulators and BF16 probabilities. [TMEM build target](../xmake.lua#L177)<!-- prefix: "target(\"sgla" -->

| TMEM reader | Geometry and selection |
| --- | --- |
| Single-CTA ordered | 32-query tile for 144–384 rows or 64-query for eligible 385–512 rows and grouped ragged shapes. Four-lane staging tries the 64-query reader before paired on clean setup decline. [Pipeline selection](../src/text/mxfp8_gqa_prefill_tmem.cu#L759)<!-- prefix: "const bool b" --> · [Setup and fallback](../src/text/gqa/tmem_staged.cu#L399)<!-- prefix: "if constexpr" --> |
| Paired | 128 queries per head pair in 64-key order. Synchronize before output and repair. Capture is one raw node with maps by value and no global scratch. Operands stay live through replay. [Arguments](../src/text/mxfp8_gqa_prefill_tmem.cu#L472)<!-- prefix: "__global__ v" --> |
| Split | Three partitions and 76,087,296 B of stream-ordered scratch for FP32 partials and merge. [Scratch allocation](../src/text/mxfp8_gqa_prefill_tmem.cu#L799)<!-- prefix: "auto status " --> |
| Page-wide | Eligible four-lane partial groups and one-/two-lane long groups. One CTA covers 128 keys, 64 queries, two heads, with ordered 64-key chunks and lane-local masking/repair. Multi-lane range validation and long-page setup require 16 B output alignment. A clean long-subgroup decline can try paired. Grouped partial uses its grouped fallback. Long-page submission errors propagate. Grouped partial can decline on an allowlisted launch rejection. Unexpected CUDA errors propagate. [Page kernel](../src/text/gqa/tmem_staged_kernels.cu#L522)<!-- prefix: "__global__ _" --> · [Long-page setup](../src/text/gqa/tmem_staged.cu#L334)<!-- prefix: "std::optiona" --> · [Submission checks](../src/text/gqa/tmem_staged.cu#L455)<!-- prefix: "if constexpr" --> · [Pipeline synchronization and layout](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/flash-prefill.md#long-tile-tmem-pipeline) |

### Same-request causal GQA

B200 [causal projection path](native-prefill-workspace.md#causal-segments-within-one-grant) stages four consecutive 1,024-row text tiles with the first start aligned to 1,024, including zero. Preserve native Q/K norm and RoPE and append only uncovered KV. One predecode and one reader cover 4,096 queries. The 128-key page reader covers 8,192–131,072 visible tokens for TP2’s 12 Q and 2 KV heads per rank and 32,769–131,072 for other layouts. [Decoder guard](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L122)<!-- prefix: "bool prefill" --> · [Launcher](../src/text/gqa/tmem_staged.cu#L570)<!-- prefix: "std::optiona" --> · [Combined preparation](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L140)<!-- prefix: "auto combine" -->

TP2 issues next-page scores into a separate slot while the current page waits for its first V and probability operands. Softmax, BF16 rounding, value order, causal cutoffs and FP64 repair are unchanged. [Score issue](../src/text/gqa/tmem_staged_kernels.cu#L469)<!-- prefix: "__device__ v" -->

Through 32,768 visible tokens, clean page setup decline (including descriptor OOM) tries the single-CTA staged reader. Its allowlisted predecode/attention launch rejection can decline to per-segment reads of appended KV on the same stream. Page submission and unexpected CUDA errors propagate without relaunch. [Descriptor setup](../src/text/gqa/tmem_staged.cu#L359)<!-- prefix: "if (const au" --> · [Reader fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L151)<!-- prefix: "for (int lan" --> · [Page submission](../src/text/gqa/tmem_staged.cu#L455)<!-- prefix: "if constexpr" -->

Hold grants and projection policy fixed. Use the segment-by-segment path as the control. Compare full BF16 vocabulary rows, sampled token scores, output IDs, saved hidden and residual rows, target and draft KV, and GDN state. Check start zero, the 8,192 and 32,768 transitions, and the 131,072 limit. Include unequal ready grants and a clean setup decline. Check that unexpected launch failures propagate. See the [B200 operator replay](native-prefill-workspace.md#tp2-causal-gqa-operator-replay).

## Constraints and failure behavior

| Route | Eligibility or dynamic shared bytes, plus static allocation |
| --- | --- |
| Regular Flash | 144–1,024 rows, 16-byte-aligned payloads, visibility = start + rows ≤131,072. One row uses ordinary forward. [Eligibility](../src/text/gqa/ordered_prefill.cu#L309)<!-- prefix: "if (rows >= " --> · [Decoder](../src/decoder_prefill.cpp#L582)<!-- prefix: "if (rows == " --> |
| Regular variants | 32/48/88 rows use 135,440/152,464/195,024 B. The 64-row async/sync variants use 169,488/135,696 B. Async needs both scale bases aligned to 8 B. Capacity or alignment can select another fitting variant. |
| Ordered or split TMEM | Capability 10.0, signed 32-bit TMA token coordinates, 214,016 B for single-CTA ordered and split. Split needs CUDA pools. [Resource checks](../src/text/mxfp8_gqa_prefill_tmem.cu#L719)<!-- prefix: "std::optiona" --> |
| Raw paired TMEM | Four 16-byte-aligned KV bases, 416 threads, a resident two-CTA cluster and 218,112 B. [Paired checks](../src/text/mxfp8_gqa_prefill_tmem.cu#L819)<!-- prefix: "std::optiona" --> · [Device and occupancy](../src/text/mxfp8_gqa_prefill_tmem.cu#L676)<!-- prefix: "std::optiona" --> |

Unsupported raw paired setup falls through to regular Flash before submission. Valid shapes with unaligned payloads or a known regular Flash shared-memory shortfall use [ordered fallback dispatch](ordered-gqa-prefill.md#constraints-and-failure-behavior). Missing buffers and insufficient pages return `cudaErrorInvalidValue`. Tensor-map encoding failure returns this status for explicit Flash. See [staged scratch and fallback](native-prefill-workspace.md#constraints-and-failure-behavior) for staging. [Encoder](../src/text/gqa/tmem_staged.cu#L24)<!-- prefix: "std::optiona" --> · [Submission](../src/text/mxfp8_gqa_prefill_tmem.cu#L843)<!-- prefix: "return cudaL" -->

A clean 32-query setup decline returns to ordinary dispatch for 144–255 rows and can try stock 64-query TMEM for existing ragged shapes. A tensor-map decline returns to ordinary dispatch. Pending and unexpected CUDA discovery, setup and launch errors propagate. After pipeline submission a recorded terminal error wins over a nonterminal launch failure. On launch success return recorded status. After allocating scratch, split always attempts release. Return the first launch error or, after successful launches, the release status. [Setup fallback](../src/text/mxfp8_gqa_prefill_tmem.cu#L769)<!-- prefix: "if (pipeline" --> · [Submission status](../src/text/gqa/tmem_staged.cu#L15)<!-- prefix: "cudaError_t " -->

Out-of-range Q or visible KV triggers protected repair for affected rows and head pairs. Ordered TMEM repairs after release and split TMEM at merge. Invalid KV affects only rows that can see it. Protected attention retains selective FP64 recovery. [Single-CTA ordered repair](../src/text/gqa/tmem_ops.cu#L476)<!-- prefix: "__device__ _" -->

### Reference implementations

See the [upstream attention and operand-precision references](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/flash-prefill.md#reference-implementations) for SGLang, FlashInfer, CUTLASS, FlashAttention-4, and DeepGEMM.

## Verification

Check BF16 results against independent FP64 and compare logits and IDs. Test causality, remaps, guards, repair, graph replay, and dispatch fallbacks.

Run the [Flash regression](../tests/mxfp8_gqa_flash_check.cu#L880)<!-- prefix: "void check_f" --> on a reserved B200 using the [checker commands](development.md#p2-gqa-verification). Against FP64, require sampled maximum absolute error below 0.08 and normalized RMSE below 0.005. [FP64 limits](../tests/mxfp8_gqa_flash_check.cu#L397)<!-- prefix: "if (!(max_ab" -->

The [partial boundary cases](../tests/mxfp8_gqa_flash_check.cu#L807)<!-- prefix: "void check_p" --> cover the 143/144 and 191/192/193 row boundaries and unaligned K/V scale parity. [Tail recovery](../tests/mxfp8_gqa_flash_check.cu#L840)<!-- prefix: "void check_t" --> confines nonfinite outputs to affected channels on the final active row.

The [long-context cases](../tests/mxfp8_gqa_flash_check.cu#L748)<!-- prefix: "void check_l" --> check 32,768/32,769-token partial tiles, the 131,072-token cap, causality, and overflow recovery. The aligned 1,023-row case checks geometry by SM count and shared memory. At 32,769, compare wide and synchronous Flash.

Check ordered and protected fallback bitwise against the ordered FP32 reference for finite scores and quotients. Check FP64 recovery independently. Use the [exact prefill runbook](../tests/prefill/README.md) for the separate `exact` contract.

If a partial Flash change preserves arithmetic, match BF16 outputs bitwise on identical checkpoint Q and KV. In that comparison, match full-vocabulary BF16 logits and greedy IDs across prefill and decode with identical grants and teacher tokens. For reordered arithmetic, follow the [operator qualification procedure](performance.md#operator-qualification). Cover unaligned starts, remaps, unwritten tails, and unchanged shapes. Weight eager timings by recorded tile counts.

The [lower-edge cases](../tests/mxfp8_gqa_flash_check.cu#L727)<!-- prefix: "void check_p" --> check dispatch and alignment at 4,096. [Graph checks](../tests/mxfp8_gqa_flash_check.cu#L335)<!-- prefix: "void check_d" --> require one node for the automatic path. The [historical split checks](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/mxfp8_gqa_flash_check.cu#L352)<!-- prefix: "void check_t" --> verified graph dependencies and scratch lifetime.

### Paired ordered TMEM verification

The [aligned regression](../tests/mxfp8_gqa_flash_check.cu#L624)<!-- prefix: "void check_p" --> checks dispatch, replay, causality, and 1,023-row parity with synchronous 64-row Flash using unaligned V scales. The [unaligned regression](../tests/mxfp8_gqa_flash_check.cu#L706)<!-- prefix: "void check_u" --> checks page offsets 1, 63, 64, and 127, replay, short page counts, and causal repair.

Check complete BF16 parity and exactly one paired graph node on remapped aligned and unaligned 1,024-row captures. Mutate visible V on full and partial pages, check unaffected rows and heads, then run memcheck, initcheck and synccheck on repair with unwritten tails. Record racecheck coverage. Test raw at zero, singleton staging at 31,744/32,768, group starts 1,024/7,168/8,192/130,048, both visibility transitions, off-grid starts, mixed grant order, and short-page or misaligned inputs. Compare staged one- and two-lane subgroups against paired for full BF16 scores, IDs, GDN, hidden/residual state and target KV, and draft continuation. Check causality with an independent FP64 sample. Require parity on clean decline and propagate submission errors. Time raw paired graphs A/B/B/A, weighted by launch count. Use [partial-grant checks](ordered-gqa-prefill.md#verification) for 256/512-row grants and [serving gates](performance.md#scheduler-and-reporting-policy) for release. [Full paired replay and timing matrix](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/flash-prefill.md#paired-ordered-tmem-verification)

### Ordered TMEM operator verification

Repeat remaps and causal boundaries with visible KV NaNs and query infinity. Compare BF16 against wide Flash. Measure operator, checkpoint, and serving separately.

### Split TMEM operator verification

Verify graph replay, output guards, and causal invariance at 65,536, 98,304, and 131,072 visible tokens. Compare with FP64 on identical packed KV. Inject unsupported device or resource settings and split or merge failures to check fallback, error propagation, and scratch release. The archived probe measured lower operator latency including the merge.

### TMEM checkpoint verification

The archived split TMEM checkpoint probe changed greedy first tokens at 65,536 and 98,304 tokens. The retired page-128 study changed first tokens at 16,384 and 32,768 tokens. [Frozen TMEM verification and reproduction](https://github.com/para1lel/sglang.cpp/blob/f6c74b50d57dff48137d2e4a19c0a9c99a15bd20/docs/flash-prefill.md#tmem-checkpoint-verification) · [Page-128 study](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/sm100-page128-prefill.md) · [Serving gate](performance.md#measurement)
