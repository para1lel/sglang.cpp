# Fixed W1–W6 serving comparison

Under the historical 32,768-token scheduling cap, native passed W1 and missed only the time-to-first-token gate on W2. It missed all four speed gates on W3 and W4. W5 and W6 stopped in the first warmup before any HTTP headers or output IDs arrived. For new formal results, follow the [paired 8,192-token protocol](../benchmarks/paired/README.md).

## Matched W1/W2 at 8,192 tokens

On November 22, 2026, vLLM ran first and native ran second on B200 `GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c`. Both used driver 590.48.01, server CPUs 24–33, client CPUs 34–35, local NUMA allocation, and the Qwen3.8-27B-FP8 checkpoint. The checkpoint manifest SHA-256 was `b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303`.

Native used revision `919885edc1b70937915ca6894a40f5f98ec3f202` and binary SHA-256 `03cf404e719d03c60dec21e100212c4af75e68ba8fe97e6632322820d4d3abf8`. vLLM used revision `81a8cb063e6247fe254d7caf7e98a6691d71022a`, PyTorch 2.14.0+cu132, and FlashInfer 0.7.4.

Both arms had an effective cap of 8,192 scheduled tokens and 32 active slots, with MTP and prefix reuse disabled. Native requested batch CUDA graphs and used tensor-core FP8 prefill. vLLM used asynchronous scheduling, chunked prefill, and full and piecewise CUDA graphs. Its context limit was 2,560 tokens and its KV budget was 10 GiB. It used FlashInfer with BF16 queries, E4M3 KV, BF16 convolution history, and FP32 recurrence.

Makespan is the last valid `[DONE]` timestamp minus the earliest client submit timestamp in a cohort. Each arm ran three warmups and nine measured single-request cohorts per row. The median uses only measured cohorts. The reduction is `100 × (1 − native median / vLLM median)`. [Paired timing and gate](../benchmarks/paired/README.md#observe-and-compare)

| Row (prompt IDs → output IDs) | Timed cohorts, native / vLLM | Native median (ms) | vLLM median (ms) | Native reduction |
| --- | ---: | ---: | ---: | ---: |
| W1 (128 → 128) | 9 / 9 | 1,026.356508 | 1,236.829263 | 17.02% |
| W2 (2,048 → 512) | 9 / 9 | 4,132.415844 | 5,001.674231 | 17.38% |

Replay verified the wire contract and repeat output IDs within each arm. Cross-engine ID agreement was outside the replay scope. W1 and W2 qualified under the paired makespan gate. This pair covered W1 and W2 only.

The benchmark host retains the samples and replay verdict at `artifacts/takeover-baseline/paired-w1w2.FrMn3y/native/W1-W2/comparison.json`.

## Matched conditions

The two engines ran sequentially on the same physical B200 for each pair, with vLLM first in every pair. Both used the local Qwen3.8-27B-FP8 checkpoint, 32,768 scheduled tokens per step, 32 service slots, no MTP, and no prefix reuse. The checkpoint manifest SHA-256 was `b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303`. The vLLM revision was `81a8cb063e6247fe254d7caf7e98a6691d71022a`, with PyTorch 2.14.0+cu132 and FlashInfer 0.7.4. Runs used driver 590.48.01. Host load and peer GPU activity differed between sessions. See the [benchmark agreement](performance.md#implementations-and-controls).

| Rows | Native revision | B200 UUID | Server / client logical CPUs | vLLM context limit / KV budget |
| --- | --- | --- | --- | ---: |
| W1–W2 | `bc07a3b56dc18a9bc32986d76c057a23953d8800` | `GPU-ceeff2f7-3f39-50b4-8d29-576159781b9d` (GPU 3) | 24–33 / 34–35 | 2,560 / 10 GiB |
| W3–W4 | `6b4f734c9c1a90ce16ff3b61ed9d3f0ad128c811` | `GPU-d4d87d81-ea84-5dd0-7167-3e1c48fde29b` (GPU 6) | 64–73 / 74–75 | 132,096 / 128 GiB |
| W5–W6 | `6b4f734c9c1a90ce16ff3b61ed9d3f0ad128c811` | `GPU-3c08cbc3-d8b3-c723-c086-a775f6a17f2a` (GPU 7) | 76–81 / 82 | 132,096 / 128 GiB |

W1–W2 CPUs were on NUMA node 0 and W3–W4 CPUs were on node 1. W5–W6 used local NUMA allocation.

Native used tensor-core FP8 prefill and enabled batch CUDA graphs. Its context limit was 262,144 before resource admission. The W3–W6 binary SHA-256 was `3471e1a76baa0b7bcd3177a357558ef85b81317fefff77364e7433eb79ddd1bb`. Native GQA uses MXFP8 KV with per-32-channel E8M0 scales. vLLM uses FlashInfer with BF16 queries and plain E4M3 KV with per-tensor FP32 unit scales. Both use BF16 GDN convolution history and FP32 recurrence. vLLM uses full and piecewise CUDA graphs. Different KV precision can change generation histories and execution paths. Cross-engine ID agreement measures only these prompts. [Cache comparison](performance.md#gqa-cache-comparison)

## Unprofiled HTTP results

W1 and W2 offered one request at a time, with three warmup and nine measured requests per row. W3 through W6 used preconnected sockets and one submission barrier, with two warmup and five measured cohorts per row. Throughput is the median cohort output IDs/s. All latency columns show pooled request p95. The wire protocol checks prompt echo, output count, `length` finish, usage, zero cached prompt IDs, and `[DONE]`. The concurrent client also checks a submission span of at most 50 ms and first output from every lane before any stream ends. [W1/W2 protocol](../benchmarks/streaming/README.md#frozen-measurement-protocol) · [W3–W6 protocol](../benchmarks/concurrent/README.md#regenerate-and-measure)

| Row (lanes × prompt IDs → output IDs per lane) | Arm | Valid timed cohorts | Output IDs/s | TTFT (ms) | TPOT (ms/ID) | E2E (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| W1 (1 × 128 → 128) | Native | 9/9 | 129.505 | 17.903 | 7.662 | 990.840 |
| W1 | vLLM | 9/9 | 113.717 | 20.367 | 8.711 | 1,126.467 |
| W2 (1 × 2,048 → 512) | Native | 9/9 | 123.210 | 141.159 | 7.865 | 4,159.857 |
| W2 | vLLM | 9/9 | 112.477 | 80.006 | 8.757 | 4,554.049 |
| W3 (4 × 8,192 → 1,024) | Native | 5/5 | 139.186 | 2,638.298 | 26.298 | 29,539.543 |
| W3 | vLLM | 5/5 | 376.948 | 1,508.140 | 9.915 | 10,875.096 |
| W4 (16 × 32,768 → 1,024) | Native | 5/5 | 86.999 | 72,523.673 | 122.591 | 188,411.088 |
| W4 | vLLM | 5/5 | 328.267 | 35,510.528 | 45.970 | 49,996.264 |
| W5 (32 × 32,768 → 4,096) | Native | 0/5 | — | — | — | — |
| W5 | vLLM | 5/5 | 836.169 | 70,984.273 | 36.345 | 156,747.864 |
| W6 (4 × 131,072 → 1,024) | Native | 0/5 | — | — | — | — |
| W6 | vLLM | 5/5 | 46.426 | 75,200.146 | 67.350 | 88,303.748 |

W1 and W2 repeated IDs within each arm. Native differed from vLLM at 108/128 W1 positions and 461/512 W2 positions. For W3 and W4, output IDs matched across repeats, between engines, and against the archived outputs. That corpus requests repetitive continuations and offers limited numerical coverage. W5 and W6 vLLM completed all seven cohorts per row and matched their archived outputs. Use matched-history BF16 logits and forced continuations for arithmetic checks. [Frozen concurrent corpus](../benchmarks/concurrent/README.md#corpus-registration) · [Numerical checks](performance.md#correctness-memory-and-diagnosis)

The W1 throughput gain was 13.9% and W2 gained 9.5%. W2 p95 TTFT exceeded the matched ceiling of 81.936 ms by 59.223 ms. W3 and W4 reached 36.9% and 26.5% of matched vLLM throughput. W1–W4 had the same pass or fail pattern against the archived gates. The archived W1/W2 reference used one sequence slot and a 2,560-token cap. The archived W3–W6 reference used 32 sequence slots and an 8,192-token cap. Against those archived gates, the completed W5 and W6 vLLM runs passed TTFT and TPOT and missed throughput and E2E. [Archived W1/W2 reference](../benchmarks/streaming/README.md#gpu3-reference) · [Archived W3–W6 gates](../benchmarks/concurrent/README.md#conditions-and-gates)

A separate replay with MTP disabled measured the same native binary used for W3–W6. It matched the fixed native W1/W2 IDs. Median output rates were 129.396 and 123.155 IDs/s. W1 passed all gates and W2 again missed only p95 TTFT, at 141.051 ms.

### W5 and W6 timeouts

All 32 W5 lanes and all four W6 lanes sent the frozen request bodies within the submission limit. The submit spans were 2.340 ms and 0.530 ms, respectively. Every lane hit the unchanged 90 s socket timeout before receiving HTTP headers. The first failures occurred 90.079 s and 90.090 s after the respective lane's submit. Both rows stopped in the first warmup. No lane received an SSE event or output ID. There are no native throughput or latency statistics for those rows. Sampled native GPU utilization was 100% in the last five seconds before each timeout.

Fresh native servers then ran one diagnostic cohort per row on GPU 7 with `--trace-batches`. Each again reached the first timeout before any output ID. The trace counts only scheduler steps completed by that point.

| Row | Completed prefill steps | Completed prompt rows | Decode lanes and graph slots before timeout |
| --- | ---: | ---: | ---: |
| W5 | 23 | 753,664 / 1,048,576 (71.875%) | 0 |
| W6 | 10 | 327,680 / 524,288 (62.5%) | 0 |

W6 reproduced in a process without prior W5 work or allocations. Both diagnostic runs were still in prefill. Native admission estimated 42.213 GiB for W5 and 17.265 GiB for W6. The server admitted both workloads. The trace counts bound prefill progress at the deadline. The traces did not measure when the first token would arrive or how much time individual kernels took.

## Phase attribution from separate traces

**W2.** The native and vLLM prefill GPU spans were 138.648 and 75.443 ms in separate traces. Their difference is close to the 61.153 ms unprofiled p95 TTFT gap. FP8 projections contributed the largest comparable kernel duration difference at 77.102 versus 46.653 ms. Native used 800 projection launches across two tiles. vLLM used 256 fused or differently tiled launches. Activation quantization, GDN, and GQA also contributed. Decode TPOT was lower on native.

**W3 and W4 decode.** While all W3 lanes were streaming output, the median gap between output IDs ranged from 26.132 to 26.199 ms on native. The vLLM range was 9.142 to 9.153 ms. On W4, the corresponding ranges were 105.705–108.315 and 14.086–14.112 ms. Full-stream W4 TPOT includes periods when other lanes are still prefilling. The diagnostic native W3 sample captured 1,021 width-four graph steps. Native W4 captured 321 complete width-16 eager steps. The separate vLLM traces captured four full-width graph steps per row.

| Mean GPU kernel duration sum per complete decode step (ms) | W3 native | W3 vLLM | W4 native | W4 vLLM |
| --- | ---: | ---: | ---: | ---: |
| FP8 projections | 12.202 | 6.368 | 48.578 | 6.503 |
| Activation quantization | 1.326 | 0.653 | 6.419 | 0.691 |
| Paged GQA core and KV append | 5.121 | 0.509 | 20.973 | 4.582 |
| GDN recurrence core | 1.357 | 0.509 | 5.839 | 1.034 |
| BF16 GDN gates | 2.104 | 0.576 | 7.393 | 0.332 |
| LM head | 0.476 | 0.362 | 6.066 | 0.368 |

Projection and paged GQA show the largest duration differences. W4 used 1,600 native grouped projections and 16 separate LM heads per full eager step. The vLLM graph used 256 fused or differently tiled projections and one head. Concurrent kernel durations can overlap, and CUDA API elapsed time can include already queued device work. The differences do not estimate independent latency savings. Comparing current native W4 with archived vLLM showed similar projection and GQA differences. That comparison spans GPU 6 at 32,768 scheduled tokens and GPU 2 at 8,192.

**W4 prefill.** In two native trace windows of about 2.5 s each, flash attention rose from 42.6% to 64.8% of summed GPU kernel durations. Mean time per flash attention launch rose from 2.975 to 7.369 ms as visible context grew. A full diagnostic totaled 35.659 s of native flash attention across 16 prompts. At equal useful causal work, median native flash time per prompt was 2.225 s. The vLLM trace captured one complete prefill. Its attention kernels totaled 1.004 s. That comparison covers attention only and changes the KV format and kernel layout. It gives no timing for W5 or W6 kernels.

## Experiments declined for adoption

**K128 prefill ticket change.** On native `6b4f734c9c1a90ce16ff3b61ed9d3f0ad128c811`, the patch grouped TMA expectations and MMA completion arrivals without changing arithmetic order. Isolated W2 projection shapes were 4.1–17.2% faster warm and 3.4–14.1% faster after L2 eviction. Based on warm kernel times and W2 launch counts, the estimated projection time fell from 82.00 to 71.90 ms. Operator outputs and model logits on identical teacher-forced histories matched the native controls bit for bit. All 96 production streams matched IDs and the wire contract. Both run orders still missed the frozen W1 and W2 throughput requirements. W2 throughput changed by +0.15% and −0.39% in the two orders. The reverse order also missed the p95 TTFT gate. Shared-host noise and the changed CPU placement and NUMA policy limit cross-order p95 comparisons. The patch was declined.

**FP16 activation sidecar.** On native `f6ee1249f9218a5427f4ead1af3a65e8657229ad`, the quantizer also wrote exact FP16 values so grouped projections could skip repeated activation conversions. E4M3 bytes, FP32 scales, and BF16 projection outputs matched bitwise. Quantization plus projection slowed on six of seven real shapes at both tested widths. Even the shared Q+KV stage slowed despite faster isolated KV. The trace-weighted warm operator estimates rose from 13.779 to 17.121 ms for W3 and from 54.596 to 67.857 ms for W4. The candidate doubled activation load instructions in the measured loop and used more registers. Actual DRAM traffic and achieved occupancy remain unmeasured. The results stayed at operator scope and the sidecar did not enter serving.

## Reproduction of the 32,768-token runs

Use the fixed source and checkpoint above. Build the native release with XMake in the default Pixi environment. Run pinned vLLM in its separate environment. Reserve the same UUID for both arms. Verify effective caps and KV policy before the first request. Run latency measurements without profiling. Keep separate runs for dispatch traces. [Native build](development.md) · [vLLM environment](../benchmarks/vllm/README.md)

For native, use `--fp8-prefill tensor-core --batch-cuda-graphs --mtp-steps 0 --prefix-cache-entries 0 --max-active 32 --max-num-batched-tokens 32768`. For vLLM use `--max-num-batched-tokens 32768 --max-num-seqs 32`, `--enable-chunked-prefill`, and `--no-enable-prefix-caching`. Set `--max-model-len` and `--kv-cache-memory-bytes` from the table. Use `--dtype bfloat16 --kv-cache-dtype fp8_e4m3 --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype float32`. Set `--served-model-name benchmark --enable-prompt-tokens-details` for the frozen wire contract. Also use `--language-model-only --skip-tokenizer-init --generation-config vllm --seed 0 --stream-interval 1`. Set the vLLM attention and compilation policies as follows.

```bash
--attention-config '{"backend":"FLASHINFER","disable_flashinfer_q_quantization":true}' \
--compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

Use the frozen direct-ID clients and fresh output directories. Use each measured run's reproduction entry below for the exact launch commands and verify each worker's live settings. [W1/W2 client](../benchmarks/streaming/README.md#run) · [W3–W6 fixtures and client](../benchmarks/concurrent/README.md#regenerate-and-measure)

The original host retains exact commands, hashes, recorded samples, and offline replay scripts under these ignored paths. The tracked protocols can regenerate the fixtures and run a fresh comparison. Byte-for-byte evidence replay requires the original artifacts.

| Evidence | Directory under `artifacts/` | Reproduction and checked inputs |
| --- | --- | --- |
| W1/W2 and K0 replay | `sol-baseline-perf/` | `meta/reproduction.md`, `both.gkAKn0/comparison.json`, `candidate/c2/report.md` |
| W3/W4 HTTP and separate profiles | `sol-parallel-perf/` | `reproduction.md`, `both.st3IXV/comparison.json`, `both.st3IXV/decode-cadence.json` |
| W3/W4 matched decode extraction | `sol-decode-gap/` | `reproduction.md`, `attribution.md`, `analyze-native.py`, `analyze-vllm.py` |
| W4 per-prompt prefill extraction | `sol-long-prefill-design/` | `report.md`, `analyze.py`, `trace-results.json` |
| W5/W6 HTTP and fresh diagnostics | `sol-long-paired/` | `REPRODUCE.md`, `comparison.json`, `diagnostic-W5.json`, `diagnostic-W6.json`, `meta/evidence.sha256` |
| K128 ticket | `sol-fp8-ticket/`, `sol-fp8-integration/` | `RESULTS.md`, `REPRODUCE.md`, `TASK-16-DECISION.md` |
| FP16 sidecar | `sol-decode-conversion/` | `RESULTS.md`, `decode-conversion.patch`, `run.sh`, `summarize.py` |
