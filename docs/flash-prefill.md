# Native MXFP8 flash prefill

## Responsibilities and sources of truth

The decoder selects tiles and appends uncovered KV. The native dispatcher chooses Flash, ordered or protected attention. [Policy](../src/decoder_prefill.cpp#L636)<!-- prefix: "const bool t" --> · [Attention call](../src/runtime/execution/decoder_prefill_tiles.cpp#L363)<!-- prefix: "void Chunk::" --> · [Dispatch](../src/text/gqa/ordered_prefill.cu#L354)<!-- prefix: "cudaError_t " --> · [Paired attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L430)<!-- prefix: "__global__ v" --> · [Single-CTA attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L325)<!-- prefix: "__global__ v" --> · [Serial prefill](text-serving.md#serial-prefill) · [Batch planner](batch-graphs.md#runtime-path)

## Runtime path

Eligible B200 prefill uses the native Flash dispatcher. The full-tile path checks the zero-based start, buffer alignment, and device resources. Restored prompts, media, and scored requests use the same planner. Each lane retains its input and scoring rules.

| Automatic single-lane TMEM route | Start position for 1,024 query rows |
| --- | --- |
| [Staged paired ordered](../src/text/gqa/tmem_staged.cu#L547)<!-- prefix: "if (position" --> | With eligible workspace, use 1,024-token boundaries from 1,024 through 31,744. Also try starts inside a 128-token page from 8,192 through 31,744. |
| [Raw paired ordered](../src/text/gqa/ordered_prefill.cu#L297)<!-- prefix: "return launc" --> | Any start from 0 through 261,120. Try after staging declines or has no eligible workspace. |

| Other automatic path | Selection |
| --- | --- |
| Ordered partial TMEM | 148-SM SM100 tries the stock 64-query reader for 1–1,023 rows at 32,769–262,144 visible tokens. Through 32,768, retain tuned ragged selection and scale alignment checks. Try [grouped attention](ordered-gqa-prefill.md#grouped-partial-grants) and eligible [long serial TP2 staging](ordered-gqa-prefill.md#long-serial-tp2-tiles) before per-lane dispatch. [Partial TMEM selection](../src/text/gqa/ordered_prefill.cu#L301)<!-- prefix: "std::optiona" --> |
| Matched whole tiles | [whole-tile MTP projection group](native-prefill-workspace.md#projection-pairs-for-mtp) with four 1,024-row tiles at matching aligned starts 1,024–130,048. Stage four lanes through 32,768 visible, two through 65,536, one through 131,072. One predecode and reader per subgroup. |
| Mixed whole tiles | Pair distinct text owners with the next eligible matching aligned start at 8,192–64,512. Unpaired tiles can stage at 32,768–130,048. A declined pair tries singleton, then lane-local. Lanes append uncovered KV and retain their own pages, masks and slices. Submitted errors propagate. [Owner and position checks](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L361)<!-- prefix: "std::optiona" --> · [Paired read](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L403)<!-- prefix: "if (const au" --> · [Lane fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L63)<!-- prefix: "read_paged_s" --> |
| Regular Flash | Causal 64-key tiles, BF16 products, FP32 accumulators and online normalization. Overlap KV loads and mask partial blocks. At start 1,024 a full tile tries 88 rows/CTA. Aligned partials starting ≥4,096 use 32/48 rows per CTA if one SM wave fits, or 88 if it fits and 64 needs two waves. At 148 SMs the respective query-row bands are 144–384, 385–576 and 769–1,023. [Regular Flash](../src/mxfp8_gqa_prefill.cu#L398)<!-- prefix: "__global__ v" --> · [Partial dispatch](../src/mxfp8_gqa_prefill.cu#L494)<!-- prefix: "if (aligned_" --> |

### Long-tile TMEM pipeline

TMEM needs `sm_100a` and regular Flash uses `sm_100`. TMEM uses TMA, FP32 value accumulators and BF16 probabilities. [TMEM build target](../xmake.lua#L176)<!-- prefix: "target(\"sgla" -->

| TMEM reader | Geometry and selection |
| --- | --- |
| Single-CTA ordered | Tuned 32-query tile for 144–384 rows or 64-query for eligible 385–512 rows and grouped ragged shapes. Long partial tiles use the stock 64-query reader. Four-lane staging tries the 64-query reader before paired on clean setup decline. [Pipeline selection](../src/text/mxfp8_gqa_prefill_tmem.cu#L668)<!-- prefix: "const bool t" --> · [Setup and fallback](../src/text/gqa/tmem_staged.cu#L377)<!-- prefix: "if constexpr" --> |
| Paired | 128 queries per head pair in 64-key order. Synchronize before output and repair. Capture is one raw node with maps by value and no global scratch. Operands stay live through replay. [Arguments](../src/text/mxfp8_gqa_prefill_tmem.cu#L430)<!-- prefix: "__global__ v" --> |
| Page-wide | Eligible four-lane partial groups and one-/two-lane long groups. One CTA covers 128 keys, 64 queries, two heads, with ordered 64-key chunks and lane-local masking/repair. Multi-lane range validation and long-page setup require 16 B output alignment. A clean long-subgroup decline can try paired. Grouped partial uses its grouped fallback. Long-page submission errors propagate. Grouped partial can decline on an allowlisted launch rejection. Unexpected CUDA errors propagate. [Page kernel](../src/text/gqa/tmem_staged_kernels.cu#L522)<!-- prefix: "__global__ _" --> · [Long-page setup](../src/text/gqa/tmem_staged.cu#L312)<!-- prefix: "std::optiona" --> · [Submission checks](../src/text/gqa/tmem_staged.cu#L433)<!-- prefix: "if constexpr" --> · [Pipeline synchronization and layout](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/flash-prefill.md#long-tile-tmem-pipeline) |

### Same-request causal GQA

B200 [causal projection path](native-prefill-workspace.md#causal-segments-within-one-grant) stages four consecutive 1,024-row text tiles with the first start aligned to 1,024, including zero. Preserve native Q/K norm and RoPE and append only uncovered KV. One predecode and one reader cover 4,096 queries. The 128-key page reader covers 8,192–131,072 visible tokens for TP2’s 12 Q and 2 KV heads per rank and 32,769–131,072 for other layouts. [Decoder guard](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L123)<!-- prefix: "bool prefill" --> · [Launcher](../src/text/gqa/tmem_staged.cu#L555)<!-- prefix: "std::optiona" --> · [Combined preparation](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L148)<!-- prefix: "auto combine" -->

TP2 issues next-page scores into a separate slot while the current page waits for its first V and probability operands. Softmax, BF16 rounding, value order, causal cutoffs and FP64 repair are unchanged. [Score issue](../src/text/gqa/tmem_staged_kernels.cu#L469)<!-- prefix: "__device__ v" -->

Through 32,768 visible tokens, clean page setup decline (including descriptor OOM) tries the single-CTA staged reader. Its allowlisted predecode/attention launch rejection can decline to per-segment reads of appended KV on the same stream. Page submission and unexpected CUDA errors propagate without relaunch. [Descriptor setup](../src/text/gqa/tmem_staged.cu#L337)<!-- prefix: "if (const au" --> · [Reader fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L163)<!-- prefix: "for (int lan" --> · [Page submission](../src/text/gqa/tmem_staged.cu#L433)<!-- prefix: "if constexpr" -->

Hold grants and projection policy fixed. Use the segment-by-segment path as the control. Compare full BF16 vocabulary rows, sampled token scores, output IDs, saved hidden and residual rows, target and draft KV, and GDN state. Check start zero, the 8,192 and 32,768 transitions, and the 131,072 limit. Include unequal ready grants and a clean setup decline. Check that unexpected launch failures propagate. See the [B200 operator replay](native-prefill-workspace.md#tp2-causal-gqa-operator-replay).

### Warm cached GQA through the full context

Eager Tensor Core TP1 and TP2 can stage one text grant with at least 32,768 cached GQA tokens. Its start must be at least 32,768. The planner requires 1,024-row tiles, at least 2,048 grant rows, and an end at or below 262,144. It requests scratch through the grant end. TP2 requires staging on both ranks. [Warm grant selection](../src/runtime/execution/decoder_prefill_plan.cpp#L47)<!-- prefix: "int resident" --> · [Both-rank staging](../src/runtime/execution/decoder_prefill_tiles.cpp#L324)<!-- prefix: "PrefillGqaSt" --> · [Scratch layout](native-prefill-workspace.md#dedicated-long-context-staging)

The resident path can combine three or four consecutive full tiles from the same request, including unaligned starts. It appends uncovered KV, predecodes the visible pages, and runs the page-wide reader with absolute causal limits. Eligible ordinary tiles can use the same staging planes. A clean staged-group decline reads the prepared tiles individually. A clean tile decline enters ordinary TMEM and Flash dispatch. [Causal group](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L130)<!-- prefix: "const bool r" --> · [Tile reader](../src/runtime/execution/decoder_prefill_tiles.cpp#L348)<!-- prefix: "staged = lon" --> · [Warm launcher](../src/text/gqa/tmem_staged.cu#L614)<!-- prefix: "decoder::try" -->

Check staged and ordinary readers with fixed grants and source pages. Cover unaligned starts, three- and four-tile groups, eligible tail tiles, visible KV repair, and visibility through 262,144. Require clean decline when scratch is short and propagation on unexpected CUDA errors.

The isolated reader probe on B200 compared 198,180,864 output elements across 24 TP1 and TP2 cases against FP64. Maximum absolute error was 0.0086611 and maximum normalized RMSE was 0.0023062. The probe also checked clean decline for missing, short or overlapping scratch and graph capture.

## Constraints and failure behavior

| Route | Eligibility or dynamic shared bytes, plus static allocation |
| --- | --- |
| Regular Flash | 144–1,024 rows, 16-byte-aligned payloads and up to 262,144 visible tokens. Visibility is start plus rows. Model and public multirow calls share dispatch. A model one-row tile uses ordinary forward. [Eligibility](../src/text/gqa/ordered_prefill.cu#L376)<!-- prefix: "if (flash_el" --> · [Public entry](../src/text/gqa/ordered_prefill.cu#L386)<!-- prefix: "cudaError_t " --> · [Decoder](../src/decoder_prefill.cpp#L594)<!-- prefix: "if (rows == " --> |
| Regular variants | 32/48/88 rows use 135,440/152,464/195,024 B. The 64-row async/sync variants use 169,488/135,696 B. Async needs both scale bases aligned to 8 B. Capacity or alignment can select another fitting variant. |
| Ordered TMEM | Capability 10.0, signed 32-bit TMA token coordinates and 214,016 B for single-CTA ordered. The raw launcher uses one stock or tuned pipeline kernel with no device scratch allocation. [Resource checks](../src/text/mxfp8_gqa_prefill_tmem.cu#L634)<!-- prefix: "std::optiona" --> |
| Raw paired TMEM | Four 16-byte-aligned KV bases, 416 threads, a resident two-CTA cluster and 218,112 B. [Paired checks](../src/text/mxfp8_gqa_prefill_tmem.cu#L706)<!-- prefix: "std::optiona" --> · [Device and occupancy](../src/text/mxfp8_gqa_prefill_tmem.cu#L591)<!-- prefix: "std::optiona" --> |

Clean raw paired or partial TMEM decline tries regular Flash when eligible. Valid shapes with unaligned payloads or a known regular Flash shared-memory shortfall use [ordered fallback dispatch](ordered-gqa-prefill.md#constraints-and-failure-behavior). Missing buffers and insufficient pages return `cudaErrorInvalidValue`. Tensor-map encoding failure returns this status for explicit Flash. See [staged scratch and fallback](native-prefill-workspace.md#constraints-and-failure-behavior) for staging. [Encoder](../src/text/gqa/tmem_staged.cu#L25)<!-- prefix: "std::optiona" --> · [Submission](../src/text/mxfp8_gqa_prefill_tmem.cu#L730)<!-- prefix: "return cudaL" -->

A clean 32-query setup decline returns to ordinary dispatch for 144–255 rows and can try stock 64-query TMEM for existing ragged shapes. A tensor-map decline returns to ordinary dispatch. Pending and unexpected CUDA discovery, setup and launch errors propagate. After pipeline submission, a recorded terminal error wins over a nonterminal launch failure. On launch success return the recorded status. [Setup fallback](../src/text/mxfp8_gqa_prefill_tmem.cu#L681)<!-- prefix: "if (pipeline" --> · [Submission status](../src/text/gqa/tmem_staged.cu#L16)<!-- prefix: "cudaError_t " -->

Out-of-range Q or visible KV triggers protected repair for affected rows and head pairs. Ordered TMEM repairs after release. Invalid KV affects only rows that can see it. Protected attention retains selective FP64 recovery. [Single-CTA ordered repair](../src/text/gqa/tmem_ops.cu#L473)<!-- prefix: "__device__ _" -->

### Reference implementations

See the [upstream attention and operand-precision references](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/flash-prefill.md#reference-implementations) for SGLang, FlashInfer, CUTLASS, FlashAttention-4, and DeepGEMM.

## Verification

Check BF16 results against independent FP64 and compare logits and IDs. Test causality, remaps, guards, repair, graph replay, and dispatch fallbacks.

Run the [Flash regression](../tests/mxfp8_gqa_flash_check.cu#L936)<!-- prefix: "void check_f" --> on a reserved B200 using the [checker commands](development.md#gqa-verification). Against FP64, require sampled maximum absolute error below 0.08 and normalized RMSE below 0.005. [FP64 limits](../tests/mxfp8_gqa_flash_check.cu#L400)<!-- prefix: "if (!(max_ab" -->

The [partial boundary cases](../tests/mxfp8_gqa_flash_check.cu#L814)<!-- prefix: "void check_p" --> cover the 143/144 and 191/192/193 row boundaries and unaligned K/V scale parity. [Tail recovery](../tests/mxfp8_gqa_flash_check.cu#L852)<!-- prefix: "void check_t" --> confines nonfinite outputs to affected channels on the final active row.

The [long-context cases](../tests/mxfp8_gqa_flash_check.cu#L752)<!-- prefix: "void check_l" --> cover the 32,768/32,769 partial-tile transition, 131,072/131,073 full tiles, causality, and overflow recovery. At 32,769, compare wide and synchronous Flash. Check aligned and unaligned starts through the full 262,144-token context in operator replay.

Check ordered and protected fallback bitwise against the ordered FP32 reference for finite scores and quotients. Check FP64 recovery independently. Use the [exact prefill runbook](../tests/prefill/README.md) for the separate `exact` contract.

If a partial Flash change preserves arithmetic, match BF16 outputs bitwise on identical checkpoint Q and KV. In that comparison, match full-vocabulary BF16 logits and greedy IDs across prefill and decode with identical grants and teacher tokens. For reordered arithmetic, follow the [operator qualification procedure](performance.md#operator-qualification). Cover unaligned starts, remaps, unwritten tails, and unchanged shapes.

The [lower-edge cases](../tests/mxfp8_gqa_flash_check.cu#L731)<!-- prefix: "void check_p" --> check dispatch and alignment at 4,096. [Graph checks](../tests/mxfp8_gqa_flash_check.cu#L338)<!-- prefix: "void check_d" --> require one node for the automatic path. The [historical split checks](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/mxfp8_gqa_flash_check.cu#L352)<!-- prefix: "void check_t" --> verified graph dependencies and scratch lifetime.

### Paired ordered TMEM verification

The [aligned regression](../tests/mxfp8_gqa_flash_check.cu#L628)<!-- prefix: "void check_p" --> checks dispatch, replay, causality, and 1,023-row parity with synchronous 64-row Flash using unaligned V scales. The [unaligned regression](../tests/mxfp8_gqa_flash_check.cu#L710)<!-- prefix: "void check_u" --> checks page offsets 1, 63, 64, and 127, replay, short page counts, and causal repair.

Check complete BF16 parity and exactly one paired graph node on remapped aligned and unaligned 1,024-row captures. Mutate visible V on full and partial pages, check unaffected rows and heads, then run memcheck, initcheck and synccheck on repair with unwritten tails. Record racecheck coverage. Test raw at zero, singleton staging at 31,744/32,768, group starts 1,024/7,168/8,192/130,048, both visibility transitions, off-grid starts, mixed grant order, and short-page or misaligned inputs. Compare staged one- and two-lane subgroups against paired for full BF16 scores, IDs, GDN, hidden/residual state and target KV, and draft continuation. Check causality with an independent FP64 sample. Require parity on clean decline and propagate submission errors. Time isolated raw paired graphs A/B/B/A on a B200 and record the grant shape and visible span. Use the [partial-grant checks](ordered-gqa-prefill.md#verification) for 256/512-row grants.

### Ordered TMEM operator verification

Repeat remaps and causal boundaries with visible KV NaNs and query infinity. Compare BF16 against wide Flash. For isolated timing, report the B200 rank configuration, row count and visible span.

### Split TMEM operator verification

Verify graph replay, output guards, and causal invariance at 65,536, 98,304, and 131,072 visible tokens. Compare with FP64 on identical packed KV. Inject unsupported device or resource settings and split or merge failures to check fallback, error propagation, and scratch release. The archived probe measured lower operator latency including the merge.

### TMEM checkpoint verification

The split TMEM checkpoint probe changed greedy first tokens at 65,536 and 98,304 tokens. The retired page-128 study changed first tokens at 16,384 and 32,768 tokens. [Page-128 study](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/sm100-page128-prefill.md)
