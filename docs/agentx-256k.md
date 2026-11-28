# AgentX 256k benchmark

Compare native serving with LMCache against pinned vLLM with Simple KV Offload on NVIDIA B200. Use the [AgentX methodology][methodology] for replay and the [InferenceX DeepSeek V4.1 Flash page][leaderboard] as a reporting reference. This campaign uses a local Qwen checkpoint and a 30-minute profiling window.

## Frozen inputs

| Input | Identity |
| --- | --- |
| Initial native baseline | `3813c20f53a27f31dadb5698ef3d319af25f4b99` on `bench/agentx-256k-tp1-tp2`. This build was used for the aborted TP2 warmup and operator diagnosis. |
| Historical native source | `fad939383840a545e03ae2d2fe17277b52403cf3`, tree `7bcc82a5f0f05956baf2e7e9d658ce039065d006`, Release binary SHA-256 `c6f750ecb4205a49ab6c6a29643ab1df9f597606bf43a85acbc4ea3ae8e11ca8`. [Original results and receipts][historical-results] |
| Current native source | `bdaec36dbb985edd57e54b3565315ba321290a23`, tree `982baef6e1bfe7113f3b10e145c1ae4a0024bf7d`, Release binary SHA-256 `d83223fdcf988f24b7d919ad97b499757e9abcd1876127ff174ddb0a9282fa5d`. The release receipt is `artifacts/agentx-256k/kvfix/official-peer-release/release-identity.json`. |
| Native LMCache | Wheel `0.5.8`, SHA-256 `4ec775bda3a8c5f1f5b1ef0bbc441b666db86b57c3c2a7bcba47a4f6550b07cd`. Each width used one private daemon. [Daemon setup](serving/lmcache.md#start-the-daemon) |
| vLLM source | `vllm-project/vllm` at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. Use `/data1/shared/dongwu.chen/environments/vllm-81a8cb063e62`. Both runs use `0.34.0rc1.dev787+g81a8cb063`, wheel SHA-256 `25358fa092260117176d78f4efeabfd06a0e49731d29bbcffd717dfedc498ad9`. [Baseline agreement](performance.md#implementations-and-controls) |
| Model and tokenizer | `/data0/shared/Qwen3.8-27B-FP8` for both engines and the replay tokenizer. The checkpoint matches Qwen revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` and supports 262,144 tokens. [Checkpoint record](research/checkpoint.md) |
| Trace | `semianalysisai/cc-traces-weka-062126-256k`, revision `8fecd2fc56694469f758f0afbbb6335ad3043740`, file `traces.jsonl`. SHA-256 `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`. The corpus has 393 trajectories and 68,266 recorded requests. The largest recorded request has 255,999 combined input and output tokens. [Pinned trace][trace] · [Corpus description][corpus] |
| Replay and scoring | `agentx-harness==1.0.14` in its own Python environment. Use InferenceX revision `d490919c996b5169a3e9345ae3c64e32ccfba357` for the published request metrics. [Standalone client][client] · [Metric source][scorer] |

## Run matrix

At each width, both native releases and vLLM used the same ordered GPU UUIDs. vLLM ran first. The current native TP1 and TP2 profiles overlapped on separate NUMA nodes.

| TP | Engine | CPU cache capacity | Physical GPUs in rank order | Live agent clients | Profile |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | Native + LMCache 0.5.8 | 64 GiB | B200 6 | 8 | 1,800 s |
| 1 | vLLM + Simple KV Offload | 64 GiB | B200 6 | 8 | 1,800 s |
| 2 | Native + LMCache 0.5.8 | 128 GiB | B200 1, 2 | 16 | 1,800 s |
| 2 | vLLM + Simple KV Offload | 128 GiB | B200 1, 2 | 16 | 1,800 s |

TP1 uses B200 6 (`GPU-d4d87d81-ea84-5dd0-7167-3e1c48fde29b`) on NUMA node 1. Pin its server to CPUs `72-81` and its client to `82-83`. TP2 uses NUMA node 0. Its rank 0 is B200 1 (`GPU-9a94015a-6a0d-1ae1-82db-b6736b77562e`). Its rank 1 is B200 2 (`GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c`). Pin the TP2 server to CPUs `12-21,24-33` and its client to `22-23,34-35`. Pin the LMCache daemon to the client CPUs. Confirm UUIDs, rank placement, and availability at startup.

## Execution

1. Freeze the launch commands before profiling. Set `--max-num-batched-tokens 32768` on both engines at TP1 and TP2. Verify the resolved limit for every arm. Match and record CPU affinity, NUMA placement, tokenizer, sampling, and context limits. Record each engine's speculative method and draft depth. A published golden SPEED-Bench acceptance calibration is unavailable for this exact checkpoint. Record the effective KV formats, scheduler limits, and differences between engines.
2. Start a fresh engine for each arm. For native, start a private [LMCache daemon](serving/lmcache.md#start-the-daemon) with the matrix capacity and UUIDs. Pass its socket to `sglang-server` with `--lmcache-socket`. Use one daemon and the same ordered UUIDs for both [TP2 ranks](serving/tp2.md#prerequisites-and-command).
3. For vLLM, set `VLLM_USE_SIMPLE_KV_OFFLOAD=1` and `--kv-offloading-backend native`. Set `--kv-offloading-size` to the matrix capacity across all ranks. Enable prefix caching and `--enable-prompt-tokens-details`. Verify that the worker selects `SimpleCPUOffloadConnector`. [Connector selection][offload] · [Capacity option][capacity]
4. Confirm each engine serves streaming `/v1/chat/completions` with the local tokenizer and context capacity. Check the terminal usage for prompt, completion, and cached prompt tokens. Preserve the readiness receipt. Restart with a fresh cache if the readiness requests populated it.
5. Verify the pinned trace bytes before invoking AgentX. Verify the preset resolves seed `42`, a 25–75% trajectory start range, and cache busting per replay. Use ten warmup requests per lane. Retain the trace timing and subagent dependencies. Use one live agent client per lane, with eight lanes per GPU. Subagent fan-out can create more simultaneous HTTP requests than live clients. [Replay methodology][methodology]
6. Verify the resolved plan before starting. Dispatch profiling requests for 1,800 seconds after the warmup barrier. Allow 30 seconds for outstanding requests after dispatch closes. Include valid profiling responses received during that grace period. Exclude primers and warmup requests. Record the phase times and drain deadline.

Run the pinned client in its locked environment. Set `SERVER_URL` and `SERVED_MODEL` to match the active server. Set `CONC` from the matrix and `RUN_DIR` to a fresh output path. Run from the staged working directory with the verified file at `semianalysisai/cc-traces-weka-062126-256k/traces.jsonl`. The official loader resolves this local dataset directory. Use the proxy at `http://127.0.0.1:7890` to fetch dependencies and the pinned corpus. Run the replay offline and send loopback requests directly.

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES='' NO_PROXY='*' no_proxy='*' \
  agentx profile --scenario agentx \
    --endpoint-type chat --url "$SERVER_URL" --model "$SERVED_MODEL" \
    --tokenizer /data0/shared/Qwen3.8-27B-FP8 \
    --public-dataset semianalysis_cc_traces_weka_062126_256k \
    --benchmark-duration 1800 --concurrency "$CONC" \
    --use-server-token-count --export-level raw --no-export-outputs-json \
    --ui none --output-artifact-dir "$RUN_DIR"
```

## Scoring and evidence

Score profiling turns that have successful transport, a finish choice, and `[DONE]`. Read token counts from the final usage chunk. Report failures, timeouts, cancellations, context-overflow skips, and usage coverage. Measure the observation span from the earliest retained request start to the latest retained response end.

| Metric | Unit | Scoring input and method |
| --- | --- | --- |
| P90 end-to-end normalized interactivity | output tokens/s/user | Take the reciprocal of the linearly interpolated 90th percentile of request end-to-end seconds per output token. Use positive latencies and output lengths. [InferenceX metric][scorer] |
| Input token throughput per chip | input tokens/s/chip | Divide server-reported input tokens by the observation span and physical chip count. Include cached input tokens in the count. [Scorer][scorer] |
| Output token throughput per chip | output tokens/s/chip | Divide server-reported completion tokens by the same span and chip count. [Scorer][scorer] |
| Actual KV cache hit rate | % | Divide the sum of server-reported `usage.prompt_tokens_details.cached_tokens` by the sum of `usage.prompt_tokens` over the same turns. Count logical tokens once at TP2. Report usage coverage and use engine counters to corroborate reuse. [Native final stream usage](../apps/server/text_completion.cpp#L591)<!-- prefix: "void TextRes" --> |
| Theoretical KV cache hit rate | % | Divide the replay's hit blocks by its total 64-token blocks over the same finalized turns. Keep the harness aggregate `theoretical_prefix_cache_hit` as a cross-check. Report any difference in the scoring cohorts. [InferenceX metric][scorer] |

Keep one evidence directory per run under `artifacts/agentx-256k/`. Record its path and a SHA-256 manifest with the results. Preserve the following material:

- Exact command lines, effective configs, build and dependency receipts, and binary or wheel hashes. Include model, tokenizer, trace, client, and scorer identities.
- GPU UUIDs and rank order, topology, clock and power settings, driver versions, and competing-process snapshots. Record CPU and memory placement.
- Replay plan, phase timestamps, raw per-request SSE in `profile_export_raw.jsonl`, `profile_export_aiperf.json`, `dataset-manifest.json`, and logs. Preserve request IDs, phase membership, latency, terminal stream status, and final usage.
- Raw `/metrics` samples at warmup and profiling boundaries and after drain. Retain native LMCache activity and vLLM local and offload activity with the engine receipts.
- Scoring invocation, numerators, denominators, sample counts, and any exclusion reason.

## Initial native warmup diagnosis

The original native TP2 warmup on B200s 0 and 1 stopped before profiling. At 131,073 visible tokens, an isolated MXFP8 GQA probe fell from paired TMEM to protected prefill. Median operator latency rose from 3.894 ms at 131,072 tokens to 489.734 ms at 131,073 tokens. The probe used 1,024 query rows, 12 query heads, and two KV heads on B200 3.

The final GQA fix, `835f877b819473fccb1bcbaf297bf50c0d56bebb`, uses paired TMEM for eligible full long tiles and ordered single-CTA TMEM for eligible partial tiles. At 131,073 visible tokens, three-trial medians for full tiles were 8.270 ms at TP1 and 4.149 ms at TP2. The final archive passed six production cases, four direct split cases, and exceptional repair. [Dispatch](ordered-gqa-prefill.md#constraints-and-failure-behavior) · [Archived warmup and operator receipts][historical-warmup]

## Timed AgentX results

Attempt 01 was stopped after abnormal TP2 peer HBM growth. The scores below use only the completed native attempt 02 and the frozen vLLM runs. Closed-loop dependencies can change the mix of finalized requests. Metrics use each run’s own finalized cohort and observation span.

| TP | Engine | P90 E2E normalized interactivity (output tokens/s/user) | Input tokens/s/chip | Output tokens/s/chip | Actual KV hit (%) | Theoretical KV hit (%) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Native + LMCache | 12.67 | 26,915.62 | 245.97 | 85.15 | 95.85 |
| 1 | vLLM + Simple KV Offload | 53.66 | 39,992.77 | 365.52 | 94.24 | 95.94 |
| 2 | Native + LMCache | 9.09 | 20,471.22 | 173.00 | 84.04 | 95.70 |
| 2 | vLLM + Simple KV Offload | 53.05 | 36,530.15 | 331.35 | 94.83 | 96.32 |

Native actual KV hits rose by 2.08 percentage points at TP1 and fell by 0.18 points at TP2 versus the historical native runs. They remained 9.08 and 10.79 points below vLLM and 10.70 and 11.66 points below the replay theory. At TP1 and TP2, native P90 interactivity was 76.4% and 82.9% lower than vLLM. Input throughput per chip was 32.7% and 44.0% lower. Output throughput per chip was 32.7% and 47.8% lower.

### Historical native results

The historical native rows used the former checkpoint and verifier paths. [Original cohort, manifests, and counters][historical-results]

| TP | Engine | P90 E2E normalized interactivity (output tokens/s/user) | Input tokens/s/chip | Output tokens/s/chip | Actual KV hit (%) | Theoretical KV hit (%) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Historical native + LMCache | 13.47 | 26,509.24 | 242.53 | 83.07 | 95.82 |
| 2 | Historical native + LMCache | 11.06 | 20,884.43 | 179.15 | 84.22 | 95.65 |

### Observation and coverage

Attempt 02 used seed `42`. The client logged 1,800.000 s of TP1 dispatch and 1,800.001 s at TP2. The launcher observed profile starts at 01:13:08.500 UTC for TP1 and 01:14:02.535 UTC for TP2 on 2026-11-28. Each finalized turn had timing, final token and cache usage, trace blocks, and a positive-latency P90 sample. The theoretical block totals matched the harness. Both valid native arms and both vLLM arms had zero errors or context-overflow skips.

| TP | Engine | Finalized / sent | Observation (s) | Chip-seconds | Input tokens | Cached input tokens | Output tokens | Theoretical hit / total blocks |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Native | 613 / 613 | 1816.140940800 | 1816.140940800 | 48,882,565 | 41,623,874 | 446,718 | 729,159 / 760,719 |
| 1 | vLLM | 837 / 837 | 1,819.132212561 | 1,819.132212561 | 72,752,129 | 68,558,464 | 664,926 | 1,086,537 / 1,132,567 |
| 2 | Native | 761 / 762 | 1825.311244764 | 3650.622489528 | 74,732,703 | 62,804,320 | 631,546 | 1,113,894 / 1,163,989 |
| 2 | vLLM | 1,260 / 1,260 | 1,804.876589299 | 3,609.753178598 | 131,864,832 | 125,046,272 | 1,196,076 | 1,978,892 / 2,054,481 |

Native TP1 engine counters reconciled exactly with finalized warmup and profiling SSE. Native TP2 canceled one request at the 30-second grace deadline. Drained counters included 185,086 input and 9,256 output tokens beyond finalized SSE, with zero excess cached tokens. The scored cohort excludes that request.

### Effective server and cache settings

Native used 8/16 running-request slots, the 32,768-token batch limit, 262,144 context, and four-token MTP. LMCache 0.5.8 reported native-v4 over SLC3. The dispatcher retained the original seed, warmup, and grace period. [Cache accounting](prefix-cache.md#checkpoint-compatibility-and-resume) · [Ranked restores](serving/lmcache.md#ranked-tp2-transport)

The drained whole replay restored 47,215,357 coherent target tokens at TP1 and 75,923,335 at TP2, with zero LMCache I/O errors. Read-only GQA coverage beyond the coherent span accounted for 2,491,722 and 4,324,357 rows respectively. Against the historical native runs, LMCache writes rose from 0.463 to 0.989 TiB at TP1 and from 0.808 to 1.528 TiB at TP2. Eviction events rose from 33 to 69 and 27 to 54 respectively. These totals cover warmup, profiling, and drain for each closed-loop cohort. Per-request attribution would require key-level data.

Both vLLM widths used four-token Qwen MTP, FlashInfer attention, FP8 E4M3 KV, and eager `SimpleCPUOffloadConnector`. GPU KV capacity was 112 GiB at TP1 and 128 GiB per rank at TP2. The TP1 128 GiB warmup exhausted memory. Both scored arms recorded CPU stores with zero external hits. All measured vLLM cache hits came from GPUs.

### Matched-request cache coverage

A join on trace ID, branch-aware conversation ID, turn, and normalized payload hash matched 599 TP1 requests and 759 TP2 requests across native releases. Theoretical hit totals were equal within each matched cohort. TP1 gained 897,187 cached tokens (+1.87 percentage points). TP2 lost 714,657 (−0.96 points). At TP2, 636 improved requests gained 2,469,849 tokens while 40 partial regressions lost 2,283,020 and eight former hits lost 983,040. Request arrival and cache history could still differ. The join measures changes in reuse, and has no per-key eviction data. See `artifacts/agentx-256k/kvfix/attribution/post-release/summary.json`.

### TP2 peer HBM

Five-second `nvidia-smi` board samples on GPU 2 started at 18,938 MiB, peaked at 58,122 MiB, and settled at 22,304 MiB after drain. The maximum idle sample was 23,728 MiB. At the settled sample, GPU 2 reported 21,346 MiB for the server and 680 MiB for LMCache. In the first 450.074 sampler seconds, the corrected board peaked at 55,100 MiB versus 177,050 MiB in the stopped attempt.

Board samples include live requests, allocator state, and pending releases. The sampler reports their combined usage. TP2 reported zero resident target and draft prefix bytes at sampled phase boundaries. With zero running requests at profiling completion, aggregate pending release was 544,555,960 bytes. The settled scrape had zero pending and failed release bytes. The measured revision's peer reclaimer fenced the releasing stream before clearing the charge. [Archived peer cache role](https://github.com/para1lel/sglang.cpp/blob/bdaec36dbb985edd57e54b3565315ba321290a23/src/runtime/execution/decoder_batch_state.cpp#L106)<!-- prefix: "retired.empl" --> · [Archived release fence](https://github.com/para1lel/sglang.cpp/blob/bdaec36dbb985edd57e54b3565315ba321290a23/src/runtime/backend/retired_request_blocks.cpp#L48)<!-- prefix: "const auto s" -->

In a separate matched two-wave fixture, with identical prompts and output IDs and balanced IPC maps and closes, second-wave settled peer server growth fell from 4,812 MiB to zero. See `hbm-matched-final-shared-reclaimer.json` in the audit directory. That fixture reports server allocations. The campaign reports whole-board peaks.

### Run receipts

Run files for the current native release are under `artifacts/agentx-256k/kvfix/campaign/runs/d83223fdcf988f24-f5533069302f2199/attempt-02/`. The independent reconciliation is `artifacts/agentx-256k/kvfix/verification/audit-bdaec36-attempt-02.json`. The vLLM runs are under `artifacts/agentx-256k/runs/`.

| Run directory | `score.json` SHA-256 | `run.sha256` SHA-256 |
| --- | --- | --- |
| `native-tp1/` | `65efd52e7f68e48c907ed20e7b90790915e860cfcb8d2e3bbd3d5d1e1bbaf79c` | `9eed44d6e701027a05916ac0b9c8f9961289976d9ca7df5414e0992dfd37bfec` |
| `vllm-tp1/` | `5633e6b6cce333d3311e1ea4874a15115aa0f8626b1985255428f45d853ff5cb` | `86be678c2a3c44f9d6278f143bd5897bba09298d46c382e173ee79043e37ecaa` |
| `native-tp2/` | `64fa764b4fb86e353495a378b855eeb8496c513048d24926e51ec51a205d8130` | `2d733b6021fc66b25440eb0c5ca4557358c8b0f3e9604672ef6b98971ebf3cff` |
| `vllm-tp2/` | `c79f6d33492c2f1408559f817b5f1020a98c9f5131f78a582082df49fce38e2b` | `a37ddc962e4d830e236f29e397aa5621c47a8c51706eb803094e2397a8431424` |

A focused historical TP1 cold publisher hit one opcode 11 timeout. The later instrumented restart fixture completed 58 STORE ARMs and 372 IPC closes. Alternate weights missed, and matching weights reused persisted state with exact continuation IDs after restart. See `persistence-restart-7119-tp1/phase-coverage.json` in the audit directory.

### Offline suffix checkpoint replay

The offline model used the 602 TP1 and 774 TP2 finalized historical native requests. It matched exact prefixes to completed earlier requests and assumed no eviction. The [suffix policy](prefix-cache.md#checkpoint-boundaries) modeled 45,703,556 of 48,023,046 input tokens reusable at TP1 (95.17%) and 72,766,151 of 76,311,656 at TP2 (95.35%). It selected about seven checkpoints per request on average, with at most 12. In the TP2 fork from export row 839 to 616, the isolated grid comparison lost 24,576 rows. The exact-token simulation lost 24,571 rows for that fork against its periodic control.

Across 3,999 historical finalized requests, no longer prompt preserved an entire earlier prompt. AgentX reconstructs historical assistant turns, so generated terminal suffixes added zero reused rows to the offline cohorts. Exact live continuation can reuse them. See `artifacts/agentx-256k/kvfix/attribution/{output-interior.json,suffix-policy-tp*.log,fenwick-policy-tp2.log}`.

### Focused cache-pressure diagnostic

The predecessor’s 64 GiB TP1 pool recorded evictions between each prime and follow-up, restoring 0 of 373,710 rows. Its 128 GiB TP2 diagnostic restored 298,038 of 660,414 rows, up from 204,800 on a different GPU pair. One restore used the common GDN endpoint 91,573. [Full pressure evidence][historical-pressure]

### Focused native numerical diagnostic

For trace fork 658 → 332, native IPC restore and independent GPU-resident prefill used the same schedule and selected greedy token 760. FP32 GDN carry, raw GDN state, hidden and residual values matched bitwise across all 48 layers. Cold unsegmented prefill of the same 144,653 IDs selected 1596. The first inspected endpoint difference was in layer 0 FP8 GDN, where changed tile boundaries altered prior rounded carry. The historical `c6f750ec…` binary showed partition sensitivity on two small W3 cases. See `artifacts/agentx-256k/replay-partitions.OD3VXl8i/`.

Transformers `Qwen3_5ForCausalLM` used BF16-dequantized weights and higher-precision GDN on the same IDs. Its top logits with probabilities were `The` (760) 23.75 (66.59%), `Let` (9764) 22.75 (24.50%), and `We` (1596) 20.875 (3.76%). The 760-minus-1596 logit margin was +2.875 in the reference, +1.125 with native restore, and −0.125 with cold prefill.

Across 248,320 vocabulary entries, total variation was 0.1462 warm and 0.3589 cold. Reference-to-native KL was 0.0974 and 0.3889 nats, respectively. Log-probability RMSE was 0.7840 and 0.6428 unweighted, or 0.3668 and 0.7695 weighted by reference probability. See `artifacts/agentx-256k/kvfix/reference-transformers/{result,comparison}.json`.

Pinned vLLM used the same IDs, zero cache hits, 32,768-token chunks, FP8 weights, BF16 activations and FP32 GDN state, with MTP off. It selected 760, ranked 9764 second and 1596 fifth, and gave 760 a 4.625 log-probability lead over 1596. Total variation against native was 0.24550 warm and 0.45807 cold. Both references cover one next token. See `artifacts/agentx-256k/kvfix/reference-vllm/{result,comparison}.json`.

### Focused full-width checkpoint publication

The `caf3f622…` Release binary processed bursts of 1,535-token prompts at observed widths 8 (TP1) and 16 (TP2). Each request wrote five prompt checkpoints at rows 511, 1,023, 1,279, 1,407, and 1,535. This produced 40 physical endpoints at TP1 and 80 on each TP2 rank. All planned prompt stores were acknowledged, with zero declines or failures. See `artifacts/agentx-256k/kvfix/verification/final-release-burst-tp{1,2}/route-evidence/wide-burst.json`.

### Focused terminal continuation

TP1 and TP2 each generated eight tokens from a 257-token prompt. The committed cache ended at row 264. After removing all other GDN checkpoints, an exact token-history continuation restored 264 coherent target rows and 263 shifted draft rows. It computed zero prompt GQA or GDN rows. All eight continuation token IDs matched uninterrupted generation at each width. See `artifacts/agentx-256k/kvfix/verification/ordered-history-terminal-tp{1,2}/route-evidence/summary.json`.

### Focused ReplaySSM verification

The integrated `c3c1ba29…` build generated identical 24-token sequences per lane at each width in K0, mixed eager K4/K2, and captured K4. It ran 14 actual graph launches at TP1 and seven at TP2, with zero target reruns. Cancellation and slot reuse passed. The isolated `13f2969a…` operator passed CPU and bitwise state checks. [Graph proofs and isolated B200 timing][historical-replay]

[methodology]: https://inferencex.semianalysis.com/agentx/methodology
[leaderboard]: https://inferencex.semianalysis.com/zh/infx/inference/deepseek-v41-flash
[trace]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/blob/8fecd2fc56694469f758f0afbbb6335ad3043740/traces.jsonl
[corpus]: https://inferencex.semianalysis.com/agentx/cc-traces-weka-062126-256k
[client]: https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/docs/agentx-standalone.md
[scorer]: https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/infx/results/agentic/request_metrics.py
[offload]: https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/vllm.py#L1231
[capacity]: https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L412
[historical-results]: https://github.com/para1lel/sglang.cpp/blob/bdaec36dbb985edd57e54b3565315ba321290a23/docs/agentx-256k.md#timed-agentx-results
[historical-warmup]: https://github.com/para1lel/sglang.cpp/blob/bdaec36dbb985edd57e54b3565315ba321290a23/docs/agentx-256k.md#initial-native-warmup-diagnosis
[historical-pressure]: https://github.com/para1lel/sglang.cpp/blob/bdaec36dbb985edd57e54b3565315ba321290a23/docs/agentx-256k.md#focused-cache-pressure-diagnostic
[historical-replay]: https://github.com/para1lel/sglang.cpp/blob/bdaec36dbb985edd57e54b3565315ba321290a23/docs/agentx-256k.md#focused-replayssm-verification
