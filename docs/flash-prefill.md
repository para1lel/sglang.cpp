# Native MXFP8 flash prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select target prefill and append uncovered KV before attention. | [Policy](../src/decoder_prefill.cpp#L613)<!-- prefix: "const bool t" --> · [Attention call](../src/runtime/execution/decoder_prefill_tiles.cpp#L224)<!-- prefix: "const auto s" --> |
| Native CUDA operator | Select TMEM Flash, regular Flash, staged or direct ordered tiles, or protected single-row attention. | [Dispatch](../src/text/gqa/ordered_prefill.cu#L292)<!-- prefix: "cudaError_t " --> |
| Long-tile TMEM | Scan the visible prefix in order or merge three partitions. | [Paired attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L472)<!-- prefix: "__global__ v" --> · [Single-CTA attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L366)<!-- prefix: "__global__ v" --> · [Split merge](../src/text/mxfp8_gqa_prefill_tmem.cu#L569)<!-- prefix: "__global__ v" --> |
| Serving | Select actual tile rows before dispatch. | [Serial prefill](text-serving.md#serial-prefill) · [Batch planner](batch-graphs.md#runtime-path) |

## Runtime path

Eligible B200 prefill uses the native Flash dispatcher. The full-tile path checks the zero-based start, buffer alignment, and device resources. Restored prompts, media, and scored requests use the same planner. Each lane retains its input and scoring rules.

| Automatic single-lane TMEM route | Start position for 1,024 query rows |
| --- | --- |
| [Raw paired ordered](../src/text/gqa/ordered_prefill.cu#L312)<!-- prefix: "if (rows == " --> | 0 |
| [Staged or raw paired ordered](../src/text/gqa/tmem_staged.cu#L562)<!-- prefix: "if (position" --> | 1,024–7,168, at 1,024-token boundaries |
| Staged or raw paired ordered | 8,192–31,744, at 1,024-token boundaries or inside a 128-token page |
| [Raw paired ordered](../src/text/gqa/ordered_prefill.cu#L324)<!-- prefix: "if (default_" --> | 32,768–130,048, at 1,024-token boundaries |

On 148-SM SM100 devices, automatic lane-local dispatch tries ordered TMEM before regular Flash. Both scale bases need eight-byte alignment. It tries 144–255 rows at 8,192–32,768 visible tokens. It tries 256–512 rows from start positions of at least 8,192, up to 32,768 visible tokens. Eligible partial-grant groups try [grouped attention](ordered-gqa-prefill.md#grouped-partial-grants) before per-lane dispatch. See [long serial TP2 tiles](ordered-gqa-prefill.md#long-serial-tp2-tiles) for eager staged reads above this range. [Partial TMEM selection](../src/text/gqa/ordered_prefill.cu#L302)<!-- prefix: "if ((ragged " -->

Within an eligible [whole-tile MTP projection group](native-prefill-workspace.md#projection-pairs-for-mtp), four 1,024-row tiles can stage together at matching aligned starts from 1,024 to 130,048. Use four lanes through 32,768 visible tokens, two through 65,536, then one through 131,072. Each subgroup issues one predecode and one attention launch. For mixed starts, distinct text owners pair with the next eligible matching aligned start at 8,192–64,512. Unpaired tiles can stage alone at 32,768–130,048. Each lane appends uncovered KV and retains its own page table, masks, and slices. A declined pair uses singleton dispatch, which can decline to lane-local. Submitted CUDA errors propagate. [Owner and position checks](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L333)<!-- prefix: "std::optiona" --> · [Paired read](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L375)<!-- prefix: "if (const au" --> · [Lane fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L62)<!-- prefix: "read_paged_s" -->

Flash scans causal 64-key tiles with BF16 products, FP32 accumulation, and online normalization. Regular Flash overlaps KV loads and masks partial blocks. A full tile at start 1,024 tries 88 rows per CTA. Aligned partial tiles starting at 4,096 or later select 32/48 rows when they fit one SM wave, or 88 when it fits and 64 needs two. At 148 SMs these cover 144–384, 385–576, and 769–1,023 query rows. [Regular Flash](../src/mxfp8_gqa_prefill.cu#L403)<!-- prefix: "__global__ v" --> · [Partial dispatch](../src/mxfp8_gqa_prefill.cu#L499)<!-- prefix: "if (aligned_" -->

### Long-tile TMEM pipeline

TMEM requires the Blackwell `sm_100a` image. Regular Flash uses `sm_100`. TMEM stages raw KV through TMA, keeps FP32 value accumulators, and uses BF16 probabilities. Single-CTA ordered TMEM can use 32-query tiles at 144–384 rows or 64-query tiles for eligible 385–512-row and grouped ragged shapes. Four-lane whole-tile staging first tries the 64-query single-CTA reader and can fall back to paired on clean setup decline. [TMEM build target](../xmake.lua#L177)<!-- prefix: "target(\"sgla" --> · [Pipeline selection](../src/text/mxfp8_gqa_prefill_tmem.cu#L759)<!-- prefix: "const bool b" --> · [Setup and fallback](../src/text/gqa/tmem_staged.cu#L399)<!-- prefix: "if constexpr" -->

Paired clusters handle 128 queries per head pair, preserve 64-key order, and synchronize before output and repair. Capture records one node with maps by value and no global scratch. Keep all operands live through replay. Split TMEM uses three partitions and 76,087,296 bytes of stream-ordered scratch for FP32 partials and merge. [Arguments](../src/text/mxfp8_gqa_prefill_tmem.cu#L472)<!-- prefix: "__global__ v" --> · [Scratch allocation](../src/text/mxfp8_gqa_prefill_tmem.cu#L799)<!-- prefix: "auto status " -->

The page-wide reader handles eligible four-lane partial groups and one- and two-lane long subgroups. One CTA reads 128 keys for 64 queries across two heads. Softmax and BF16 PV preserve the two 64-key chunks' order and lane-local causality and repair. Long-page output needs 16-byte alignment. Clean setup decline tries paired before submission. Setup, predecode, and launch errors propagate. [Page kernel](../src/text/gqa/tmem_staged_kernels.cu#L522)<!-- prefix: "__global__ _" --> · [Long-page setup](../src/text/gqa/tmem_staged.cu#L334)<!-- prefix: "std::optiona" --> · [Submission checks](../src/text/gqa/tmem_staged.cu#L455)<!-- prefix: "if constexpr" --> · [Pipeline synchronization and layout](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/flash-prefill.md#long-tile-tmem-pipeline)

### Same-request causal GQA

On B200, the [causal projection path](native-prefill-workspace.md#causal-segments-within-one-grant) can stage four consecutive 1,024-row text tiles from one request. The first tile starts on a 1,024-row boundary, including zero. The decoder preserves native Q/K normalization and RoPE across the group and appends only uncovered KV. One KV predecode and one reader launch cover all 4,096 queries. For TP2 ranks with 12 query heads and two KV heads, the native 128-key page reader starts at 8,192 visible tokens. Other head layouts use it above 32,768 visible tokens. The page reader supports up to 131,072 visible tokens. [Decoder guard](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L122)<!-- prefix: "bool prefill" --> · [Launcher](../src/text/gqa/tmem_staged.cu#L570)<!-- prefix: "std::optiona" --> · [Combined preparation](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L140)<!-- prefix: "auto combine" -->

The TP2 group issues the next page’s scores into a separate slot before waiting for the current page’s first V and probability operands. Softmax, BF16 rounding, value update order, causal cutoffs, and FP64 repair stay the same. [Score issue](../src/text/gqa/tmem_staged_kernels.cu#L469)<!-- prefix: "__device__ v" -->

Through 32,768 visible tokens, the launcher retries the single-CTA staged reader if page setup declines. A tensor-map descriptor allocation failure can cause this decline. The single-CTA path can also decline on an allowlisted predecode or attention launch rejection. If staging declines, the per-segment reader reuses appended KV on the same stream. Page submission errors and unexpected CUDA failures propagate without another attention launch. [Descriptor setup](../src/text/gqa/tmem_staged.cu#L359)<!-- prefix: "if (const au" --> · [Reader fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L151)<!-- prefix: "for (int lan" --> · [Page submission](../src/text/gqa/tmem_staged.cu#L455)<!-- prefix: "if constexpr" -->

Hold grants and projection policy fixed. Use the segment-by-segment path as the control. Compare full BF16 vocabulary rows, sampled token scores, output IDs, saved hidden and residual rows, target and draft KV, and GDN state. Check start zero, the 8,192 and 32,768 transitions, and the 131,072 limit. Include unequal ready grants and a clean setup decline. Check that unexpected launch failures propagate. See the [B200 operator replay](native-prefill-workspace.md#tp2-causal-gqa-operator-replay).

## Constraints and failure behavior

Regular Flash needs 16-byte-aligned payloads and 144–1,024 rows. Visibility caps at 131,072. Visibility equals start plus rows. One row uses ordinary forward execution. [Eligibility](../src/text/gqa/ordered_prefill.cu#L309)<!-- prefix: "if (rows >= " --> · [Decoder](../src/decoder_prefill.cpp#L582)<!-- prefix: "if (rows == " -->

The 32-, 48-, and 88-row variants need 135,440, 152,464, and 195,024 dynamic shared-memory bytes respectively. The 64-row variants need 169,488 bytes asynchronously or 135,696 synchronously. Each needs static allocation as well. Asynchronous staging needs 8-byte alignment for both scale bases. Capacity or alignment can select another variant when it fits.

TMEM requires compute capability 10.0 and signed 32-bit TMA token coordinates. Single-CTA ordered and split TMEM need 214,016 dynamic shared-memory bytes plus static allocation. Split TMEM needs CUDA memory pools. [Resource checks](../src/text/mxfp8_gqa_prefill_tmem.cu#L719)<!-- prefix: "std::optiona" -->

Raw paired TMEM requires four 16-byte-aligned KV bases, a 416-thread kernel, a resident two-CTA cluster, and 218,112 dynamic shared-memory bytes plus static allocation. [Paired checks](../src/text/mxfp8_gqa_prefill_tmem.cu#L819)<!-- prefix: "std::optiona" --> · [Device and occupancy](../src/text/mxfp8_gqa_prefill_tmem.cu#L676)<!-- prefix: "std::optiona" -->

Unsupported raw paired clusters, kernel configurations, encoders, or tensor maps fall through to regular Flash before submission. Missing buffers or insufficient pages return `cudaErrorInvalidValue`. Propagate unexpected CUDA setup errors and the launch status. See [staged scratch and fallback](native-prefill-workspace.md#constraints-and-failure-behavior) for staged submission rules. [Encoder](../src/text/gqa/tmem_staged.cu#L24)<!-- prefix: "std::optiona" --> · [Submission](../src/text/mxfp8_gqa_prefill_tmem.cu#L843)<!-- prefix: "return cudaL" -->

Other valid shapes, unaligned payloads, and a known regular Flash shared memory shortfall use the [ordered fallback dispatch](ordered-gqa-prefill.md#constraints-and-failure-behavior). CUDA discovery, configuration, and launch errors propagate.

When the 32-query kernel setup declines, 144–255-row grants return to ordinary dispatch. Existing ragged shapes can try stock 64-query TMEM. A clean tensor-map decline returns to ordinary dispatch. Missing buffers or pages and explicit encoding failures return `cudaErrorInvalidValue`. Pending and unexpected CUDA errors propagate. After pipeline submission, a recorded terminal error takes precedence over a nonterminal launch error. A successful launch returns the recorded CUDA status. If split attention allocates scratch, the launcher attempts release and returns the first launch error or the release status. [Setup fallback](../src/text/mxfp8_gqa_prefill_tmem.cu#L769)<!-- prefix: "if (pipeline" --> · [Submission status](../src/text/gqa/tmem_staged.cu#L15)<!-- prefix: "cudaError_t " -->

Protected attention repairs affected rows and head pairs when a query or visible KV value falls outside Flash's numerical range. Ordered TMEM repairs after release. Split TMEM marks partitions for repair at merge. Invalid KV affects only queries that can see it. Protected attention retains selective FP64 recovery. [Single-CTA ordered repair](../src/text/gqa/tmem_ops.cu#L476)<!-- prefix: "__device__ _" -->

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
