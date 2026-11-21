# Native MXFP8 flash prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the target prefill policy and append KV before attention. | [Policy](../src/decoder_prefill.cpp#L486)<!-- prefix: "const bool t" --> · [Attention call](../src/decoder_prefill.cpp#L293)<!-- prefix: "const auto s" --> |
| Native CUDA operator | Select TMEM Flash, regular Flash, ordered multirow, or protected single-row attention. | [Dispatch](../src/mxfp8_gqa_prefill.cu#L540)<!-- prefix: "cudaError_t " --> |
| Long-tile pipeline | Partition the visible prefix and merge the partial softmax results. | [TMEM attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L321)<!-- prefix: "__global__ v" --> · [Merge](../src/text/mxfp8_gqa_prefill_tmem.cu#L421)<!-- prefix: "__global__ v" --> |
| Serving | Select actual tile rows before dispatch. | [Serial prefill](text-serving.md#serial-prefill) · [Batch planner](batch-graphs.md#runtime-path) |

## Runtime path

The CLI and C++ default policy is `exact`. Its original MXFP8 attention path retains the exact prefill contract. With `tensor-core`, the decoder selects regular Flash for eligible multirow tiles on the SM100 build. Set `SGLANG_EXPERIMENTAL_TMEM_PREFILL=1` before starting the process to enable experimental TMEM for eligible full 1,024-row tiles. Unset and other values keep regular Flash. The 128-row media, MTP, and tensor-core prefix-cache tile limits keep those routes outside TMEM.

On SM100, `SGLANG_PREFILL_ATTN=sm100` makes the model's tensor-core prefill try experimental split attention before the portable dispatcher. The launcher accepts up to 8,192 query rows and 262,144 visible tokens. It requires at least 144 rows through 32,768 visible tokens, 16-byte alignment for Q and output, and 32-byte alignment for K and V payloads. On B200, the batch can use 2,048-row tiles for text prompts of at most 32,768 tokens when the tensor-core prefix-cache tile planner is inactive and the request has no MTP draft. The kernel rescales the online softmax whenever its maximum increases and waits for PV completion before accumulator reuse. [Model dispatch](../include/sglang/text/prefill_attn_sm100.h#L22)<!-- prefix: "inline cudaE" --> · [SM100 launcher](../src/text/prefill_attn_sm100.cu#L717)<!-- prefix: "std::optiona" --> · [Batch tile cap](../src/decoder_batch.cpp#L699)<!-- prefix: "const bool w" -->

For ordinary multisplit results, each half of the finish block handles one query head. Each thread rescales positive-mass splits to their largest score, accumulates in FP32 in split order, and writes two adjacent BF16 channels. A combined mass that fails `> 0` yields two positive zeros. The producer writes ordinary single-split results. Protected repair uses the whole finish block. [Merge](../src/text/prefill_attn_sm100.cu#L549)<!-- prefix: "__device__ v" --> · [Finish](../src/text/prefill_attn_sm100.cu#L585)<!-- prefix: "prefill_attn" --> · [Producer output](../src/text/prefill_attn_sm100.cu#L410)<!-- prefix: "__device__ v" -->

Both Flash paths follow the logical page table. Each work group covers up to 64 query rows and two query heads that share a KV head. It processes 64-key tiles and masks each query at its own position. QK and PV use BF16 Tensor Core products with FP32 accumulators. Online softmax keeps FP32 normalization and rounds the probabilities used by PV to BF16.

When a regular Flash block has fewer than 64 rows, it stages zeros for inactive queries. It masks their scores and skips their output stores and repair work. Multiples of 64 use the specialization for complete blocks. [Regular Flash](../src/mxfp8_gqa_prefill.cu#L450)<!-- prefix: "__global__ v" -->

### Long-tile TMEM pipeline

For each query block and head pair, three CTAs handle contiguous partitions of the visible prefix, with boundaries at 64-key tiles. The last tile uses a causal mask for each query. The Blackwell `sm_100a` target uses `tcgen05` and tensor memory (TMEM) for scores and FP32 value accumulators. Regular Flash stays in the `sm_100` target. [TMEM build target](../xmake.lua#L67)<!-- prefix: "target(\"sgla" -->

Tensor Memory Accelerator (TMA) copies raw K and V from the physical pages into two alternating shared-memory buffers. Producer warps convert scaled MXFP8 to BF16. Packed conversion handles finite payloads and scale exponents 15–148. Other values use the scalar conversion and its range checks.

Four warps consume QK scores and compute softmax while eight producer warps convert V. The next raw tile can load during current-tile work. PV overlaps conversion of the next K tile. Barrier phases and proxy fences order the transfers, conversions, and tensor operations before buffer reuse. [Pipeline](../src/text/mxfp8_gqa_prefill_tmem.cu#L321)<!-- prefix: "__global__ v" --> · [Conversion](../src/text/mxfp8_gqa_prefill_tmem.cu#L99)<!-- prefix: "__device__ _" -->

Each launch uses 76,087,296 bytes of stream-ordered scratch. Each of the three partitions stores a 256-channel FP32 numerator, a local maximum, and a normalization mass for each of 1,024 rows and 24 query heads. The merge rescales to the global maximum, combines the FP32 masses and numerators, divides, and rounds to BF16. The launch queues allocation, attention, merge, and release on the same stream. [Scratch ownership](../src/text/mxfp8_gqa_prefill_tmem.cu#L463)<!-- prefix: "std::optiona" -->

## Constraints and failure behavior

Regular Flash accepts 144–1,024 query rows through 32,768 visible tokens. From 32,769 through 131,072 visible tokens, it accepts only full 1,024-row tiles. Visible tokens include the final query and equal the zero-based start position plus the row count. Both K and V payload bases must be 16-byte aligned. Regular Flash requires 154,128 dynamic shared-memory bytes plus static allocation. The decoder handles a one-row prompt tile through its ordinary forward path. [Eligibility](../src/mxfp8_gqa_prefill.cu#L554)<!-- prefix: "if (rows \u003e= " --> · [Decoder](../src/decoder_prefill.cpp#L466)<!-- prefix: "if (rows == " -->

With the opt-in, TMEM requires exactly 1,024 query rows and at most 131,072 visible tokens. The zero-based start must be at least 32,768 and divisible by 64. TMEM also requires compute capability 10.0, CUDA memory pools, and a K/V pool addressable by signed 32-bit TMA token coordinates. The launch needs opt-in shared memory for 214,016 dynamic bytes plus static allocation. A known resource or device mismatch, or a successful driver lookup without a tensor-map encoder, selects regular Flash when eligible. [TMEM eligibility](../src/mxfp8_gqa_prefill.cu#L557)<!-- prefix: "if (rows == " -->

Other valid shapes, unaligned payloads, and a known regular Flash shared memory shortfall use the [ordered fallback dispatch](development.md#ordered-multirow-gqa-prefill). Above 32,768 visible tokens, that dispatch uses protected single-row attention. CUDA discovery, configuration, and launch errors propagate.

The TMEM launcher checks the required page counts and pointers, then encodes separate K and V tensor maps before allocating scratch. Discovery or encoding errors and allocation failures propagate. After allocation, the launcher attempts release even if attention or merge fails. It returns the first launch failure, or the release status if both launches succeed.

If a query or visible KV value falls outside Flash's numerical range, protected attention repairs the affected row and query-head pair. TMEM propagates the invalid state through the partition statistics to the merge. A bad KV position beyond a query's causal limit leaves that query unaffected. Protected attention retains selective FP64 recovery.

## Verification

For the explicit tensor-core policy, prioritize measured efficiency over bitwise parity with ordered attention. Flash can change BF16 results, logits, and generated IDs. Compare flash with an independent FP64 reference over the same represented inputs. Check causality, page remaps, output guards, repaired rows, and eager versus CUDA Graph replay. Trace dispatch at the eligibility bounds and through alignment and capacity fallback.

Run [`mxfp8-gqa-check`](development.md#p2-gqa-verification) on a reserved B200 with the TMEM setting unset. Repeat with `SGLANG_EXPERIMENTAL_TMEM_PREFILL=1`. Require both runs to pass. The [Flash regression](../tests/mxfp8_gqa_flash_check.cu#L673)<!-- prefix: "void check_f" --> checks capture and replay, exact dispatch, and unaligned K and V separately. Require sampled maximum absolute error below 0.08 and normalized RMSE below 0.005 against FP64. [FP64 limits](../tests/mxfp8_gqa_flash_check.cu#L435)<!-- prefix: "if (!(max_ab" -->

The [partial boundary cases](../tests/mxfp8_gqa_flash_check.cu#L611)<!-- prefix: "void check_p" --> check the 143/144-row floor and the 191/192/193-row block boundary. The [partial recovery case](../tests/mxfp8_gqa_flash_check.cu#L630)<!-- prefix: "void check_p" --> checks score overflow and limits nonfinite outputs to the affected channels of the final active row.

The [long-context cases](../tests/mxfp8_gqa_flash_check.cu#L562)<!-- prefix: "void check_l" --> require regular Flash for aligned 1,023- and 1,024-row tiles at 32,768 visible tokens. At 32,769, the full tile uses Flash and the partial tile uses protected attention. At 65,536, 98,304, and 131,072 visible tokens, full tiles require regular Flash by default and TMEM with the opt-in. At 131,073 and 131,136, full tiles require protected attention in both modes. The 131,072-token cases also check the exact route, causality, and score overflow recovery.

Build `mxfp8-gqa-sm100-check` with `--mxfp8-gqa-check=y` in the SM100 Release build and run it on the reserved B200. The check enables the experimental launcher itself. The rising-max control requires maximum absolute error below 0.00002. The Candidate B rerun measured 2.12333e-06 in that control. See `artifacts/coding-fix-a/final-mxfp8-gqa-sm100.log`. [Build target](../xmake.lua#L202)<!-- prefix: "target(\"mxfp" --> · [Rising-max assertion](../tests/prefill/mxfp8_gqa_sm100_check.cpp#L144)<!-- prefix: "require(maxi" -->

Independent B200 replay matched the previous merge's BF16 bytes across remapped pages, poisoned output, a 193-row tail, and exceptional inputs. The direct four-split finish probe passed FP64 comparison and memcheck.

Paired median CUDA Graph replay time fell 2.26–6.48% for three 1,024-row, three-split shapes. Across six pairs per shape, reductions were 2.13–6.73%. The four single-split medians rose 0.11–1.54%. Timing includes scratch allocation, flag initialization, both kernels, and release. Checkpoint quality and full serving performance require separate qualification. See `artifacts/resume-merge-accept/handoff.md`.

Check ordered and protected fallback bitwise against the ordered FP32 reference for finite scores and quotients. Check FP64 recovery independently. Use the [exact prefill runbook](../tests/prefill/README.md) for the separate `exact` contract.

The [lower-edge cases](../tests/mxfp8_gqa_flash_check.cu#L543)<!-- prefix: "void check_t" --> cover the aligned start below 32,768, positions 32,767 through 32,769, and unaligned K and V separately. With the opt-in, the first eligible tile also repeats graph replay. The [graph check](../tests/mxfp8_gqa_flash_check.cu#L336)<!-- prefix: "void check_t" --> verifies the dependency chain from allocation through attention and merge to release, including the scratch size and pointer.

### TMEM operator verification

The independent B200 release check passed the long-context dispatch, graph, numerical, and causal cases. At 65,536, 98,304, and 131,072 visible tokens, normalized RMSE against FP64 was 0.224%, 0.219%, and 0.221%, respectively, over 12,800 sampled values per shape. At the first eligible start, repeated replay preserved the BF16 output and its guards after the output buffer was overwritten. The 131,072-token causal mutation preserved all 399,360 BF16 values in the first 65 rows bitwise.

At 1,024 query rows, the paired median time including the merge fell from 18.206 to 7.171 ms at 65,536 visible tokens and from 37.343 to 8.937 ms at 131,072. Memcheck and synccheck each reported zero errors on the 65,536-token operator probe.

A separate launch fault probe exercised the device and resource fallbacks, checked CUDA error propagation, and confirmed that split or merge failure still requests scratch release.

### TMEM checkpoint verification

The Qwen3.8-27B-FP8 probe compared TMEM with regular Flash from `ec1ca540` on frozen W6 lane 0, truncated to 65,536, 98,304, and 131,072 prompt IDs. Greedy IDs changed at the first two lengths and matched at 131,072. On the same teacher histories the second argmax matched at all three lengths. The separate 193-token `exact` control matched all 496,640 BF16 logits across its two continuation steps.

At 98,304 prompt tokens, TMEM ran on 1,024 of 1,536 prefill GQA calls. The first argmax changed from 7,035 to 1,455. Full-logit cosine was 0.821539, top-10 overlap was 1/10, and negative log likelihood for ID 7,035 rose by 20.994807 nats. The second argmax was 29,877 in both arms after forcing ID 7,035, with cosine 0.998866. Repeated captures matched all 496,640 BF16 logits within each arm.

Identical-input replays used the first eligible tile at position 32,768 in layer 3 and the tile with the largest sampled difference at position 79,872 in layer 59, both zero-based. Full-tile TMEM versus regular Flash normalized RMSE was 0.06130% and 0.09469%, respectively. Against independent FP64 over seven query rows, all 24 query heads, and all visible tokens per row, regular Flash and TMEM errors were 0.16567% and 0.16538% on the first tile and 0.16412% and 0.15900% on the later tile. Page remaps and causal-tail mutations passed in both kernels.

Using TMEM on only the first eligible call preserved first argmax 7,035 with a 0.010311-nat NLL increase. Using it on the first 256 eligible calls changed the argmax to 3,470 with an 18.999520-nat increase. The probes leave the source of the full-model drift unresolved. Two replayed tiles and one document cannot establish corpus quality or serving acceptance. This drift is the reason TMEM requires an experimental opt-in.

#### Reproduce the first-token comparison

Build the generator with the [P2 build command](development.md#p2-gqa-verification). Regenerate the [W6 fixtures](../benchmarks/concurrent/README.md#regenerate-and-measure) with the Qwen3.8-27B-FP8 tokenizer. Set `FIXTURES` to that fixture directory, `CHECKPOINT` to the checkpoint, and use the same `BUILD_WORK_DIR` and `GPU_UUID` as for the checker.

```bash
python - "$FIXTURES/W6/00.request.json" > prompt-96k.ids <<'PY'
import json
import sys

with open(sys.argv[1]) as fixture:
    prompt = json.load(fixture)["prompt"]
assert len(prompt) == 131072
print(*prompt[:98304], sep="\n")
PY
sha256sum -c <<'SUM'
735ac7151bc0611bbeee83bacf148ea0b4d83ec0000b1e3312c72cece82e5670  prompt-96k.ids
SUM
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
unset SGLANG_SERIAL_PREFILL_ROWS
for setting in 0 1; do
    SGLANG_EXPERIMENTAL_TMEM_PREFILL="$setting" \
        "$BUILD_WORK_DIR/build/bin/sglang-generate" --checkpoint "$CHECKPOINT" \
        --device 0 --prompt-file prompt-96k.ids --fp8-prefill tensor-core \
        --mtp-steps 0 --temperature 0 --max-new-tokens 1 --top-logits 248320 \
        --warmup 0 --repeats 1 > "prefill-96k-$setting.json"
done
```

Compare `output_token_ids` and `step_top_logits` across arms. The full vocabulary is emitted in score order, so join by `token_id` before computing cosine and NLL. This command tests the common first-token history. The original second-step teacher comparison and tile replay used instrumented probes.

### Partial Flash verification

The B200 regression passed with the partial Flash implementation. On the base build, it stopped at the 144-row dispatch assertion. The 193-row recovery case preserved all 1,179,648 earlier BF16 outputs and produced exactly 262 nonfinite outputs on the final row.

An independent B200 operator probe observed ordered attention at 143 and 1,025 rows, and Flash at 144 rows. It also checked remapped pages, output guards, unaligned K and V separately, shared memory fallback, and CUDA error propagation.

The sampled 144-row case at 8,192 visible tokens had 0.0188% normalized RMSE against independent FP64. At 147 rows and 32,768 visible tokens, all 256 BF16 outputs of the repaired head on the final active row matched exact. The probe's eager and graph outputs matched bitwise, and its memcheck reported zero errors. The raw logs are in `artifacts/takeover-tail-integration/operator/`.

On four documents with 128 teacher-forced tokens each, switching the 144-row tail to flash changed 17 of 512 top IDs and shifted mean NLL by −0.00923 nats per token. Switching the 1,023-row tail changed 27 of 512 top IDs and shifted mean NLL by +0.02774. The 143-row and 1,024-row controls had identical logits on the evaluated rows. See `artifacts/takeover-partial-flash-test/quality/results/summary.json`.

### Earlier 32,768-token measurements

On a reserved B200, default flash sped up nine eligible operator shapes by 8.92–12.77× by paired median kernel time. The operator probe completed 24 graph replays and retained the previous routes at 1,023 and 1,025 rows and 32,769 visible tokens. The earlier arithmetic-identical kernel had 0.158–0.225% normalized RMSE on sampled FP64 comparisons across those eligible shapes.

The native HTTP comparison used a 32-slot batch, a 32,768-token step budget, K0, decode graphs, and prefix reuse off. W1 and W2 used nine measured requests per arm. W3 used five measured four-lane cohorts per arm.

| Workload | Median TTFT, ordered → flash-enabled (ms) | Median output IDs/s, ordered → flash-enabled | Output comparison |
| --- | ---: | ---: | --- |
| W1 | 19.606 → 19.912 | 128.753 → 128.182 | Both used ordered attention. IDs matched. |
| W2 | 272.398 → 176.219 | 119.070 → 121.847 | The first different ID was 314 → 440 at zero-based output index 13. |
| W3, concurrency 4 | 9,284.164 → 3,200.740 | 110.374 → 132.611 | All 28,672 IDs matched, including warmups. |

W1 p95 TTFT rose from 19.793 to 64.086 ms in these runs. On the common W2 teacher-forced history, 62 of 64 top IDs matched and mean NLL increased by 0.00460 nats per token. These measurements compare two attention routes in the same FP8 checkpoint. Both W3 arms missed the frozen external-reference gates.

At W4 concurrency 16, both arms stopped at the first warmup with zero completed streams. Ordered attention emitted zero IDs across all 16 lanes before their 90-second socket timeouts. With flash enabled, five lanes timed out with zero IDs at 90 seconds. The other 11 received HTTP 200 and emitted 39–55 IDs each before the 360-second cohort deadline. W4 produced no scored results. Overall vLLM acceptance remains open.

Timing and dispatch records are in `artifacts/takeover-flash-prefill-default/handoff.md`. Independent numerical review is in `artifacts/takeover-flash-prefill-review/report.md`.

### W6 warmups and diagnostics

The [frozen W6 protocol](../benchmarks/concurrent/README.md#conditions-and-gates) uses four concurrent 131,072-token prompts and 1,024 output IDs per request. The matched GPU6 campaign used 32,768 scheduled tokens per step, 32 active slots, disjoint prefixes, and MTP off. Both native builds used landed GDN. The candidate extended full-tile Flash visibility to 131,072 tokens.

Both native arms stopped at the first warmup. All four streams in each arm reached the 90-second socket timeout before the first output ID. Neither arm produced a scored cohort or a native output comparison.

Pinned vLLM completed two warmups and five measured cohorts. All 28 streams matched the archived output IDs and repeated exactly across cohorts. The maintained gates remain the frozen W6 thresholds.

| Metric | Pinned vLLM | Frozen W6 gate | Result |
| --- | ---: | ---: | --- |
| Median output IDs/s | 45.467107 | > 48.709172 | Fail |
| p95 TTFT (ms) | 75,540.784412 | ≤ 113,118.073927 | Pass |
| p95 TPOT (ms) | 69.018133 | ≤ 104.925433 | Pass |
| p95 E2E (ms) | 90,104.907074 | ≤ 85,806.011537 | Fail |

The same builds ran separate one-lane, two-ID diagnostics with the 32,768-token step cap. They used a 1,800-second socket timeout and a 1,810-second cohort deadline. At 40,960 prompt tokens, control and candidate both returned `16,25`. TTFT fell from 37.386 s to 6.115 s. At 131,072 prompt tokens, the candidate returned `760,45366` with 45.798 s TTFT. The 131,072-token diagnostic tested the candidate only.

See `artifacts/takeover-w6-flash-selected/report.md` for the integrated source identities and run record.

The earlier GPU6 campaign used an 8,192-token step cap. Both native arms stopped at the first warmup with zero IDs on all four lanes. A separate profile spent 74.021 s of its 95.010 s attention interval in protected prefill. See `artifacts/takeover-w6/report.md`.

A separate prototype diagnostic used single requests with two output IDs, the historical 8,192-token step cap, a 1,800-second socket timeout, and a 1,810-second cohort deadline. Both arms completed and agreed on both IDs at 40,960 and 131,072 prompt tokens. The longer Flash route reduced TTFT on those requests. See `artifacts/takeover-w6-flash-code/report.md`.

### Double-buffered TMEM prototype

On GPU6, the prototype compared with sealed TMA at 1,024 query rows with shuffled physical pages. Paired median attention time, including split merge, fell from 12.814 to 4.674 ms at 65,536 visible tokens. At 131,072 visible tokens, it fell from 26.256 to 9.018 ms. Sampled normalized RMSE against FP64 was 0.221% and 0.220%, respectively. The partitioned reduction changed 2,546,820 and 2,555,831 of 6,291,456 BF16 outputs against sealed TMA. Two graph replays per sampled variant matched their own eager outputs bitwise.

The paired two-ID single-lane runs changed greedy IDs at 65,536 and 98,304 prompt tokens. Both IDs matched at 131,072. The prototype and its sealed TMA control used an earlier source revision. See `artifacts/takeover-w6-prefill-pipeline/report.md` for the source identities, diagnostic IDs, and timings.

All four prototype W6 lanes emitted their first ID at 77.243–77.273 s. At the 360-second deadline, the lanes had emitted 705, 705, 705, and 704 IDs. All 2,819 observed IDs matched the frozen prefixes. The first warmup ended with zero completed requests and no scored cohort.

The frozen throughput gate allows less than 84.091 s for all 4,096 IDs. At the observed first-ID latency, less than 6.82 s remained for 1,023 decode iterations and final delivery. A separate two-output trace attributed 9.008 s of the first decode step to attention.

Report prompt length, concurrency, output budget, and deadlines for diagnostics. W6 acceptance requires complete measured cohorts under the [current launch policy](performance.md#scheduler-and-reporting-policy) and all [frozen W6 gates](../benchmarks/concurrent/README.md#conditions-and-gates). The prototype leaves W6 acceptance open.
