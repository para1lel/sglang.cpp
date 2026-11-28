# AgentX 256k benchmark

Compare native serving with LMCache against pinned vLLM with Simple KV Offload on NVIDIA B200. Use the [AgentX methodology][methodology] for replay and the [InferenceX DeepSeek V4.1 Flash page][leaderboard] as a reporting reference. This campaign uses a local Qwen checkpoint and a 30-minute profiling window.

## Frozen inputs

| Input | Identity |
| --- | --- |
| Initial native baseline | `3813c20f53a27f31dadb5698ef3d319af25f4b99` on `bench/agentx-256k-tp1-tp2`. This build was used for the aborted TP2 warmup and operator diagnosis. |
| Measured native source | `fad939383840a545e03ae2d2fe17277b52403cf3`, tree `7bcc82a5f0f05956baf2e7e9d658ce039065d006`. Both 30-minute runs used Release binary SHA-256 `c6f750ecb4205a49ab6c6a29643ab1df9f597606bf43a85acbc4ea3ae8e11ca8`. The build receipt is under `artifacts/agentx-256k/native-preflight/committed-release-fad939383840.uE4Olan4/`. |
| Native LMCache | Wheel `0.5.8`, SHA-256 `4ec775bda3a8c5f1f5b1ef0bbc441b666db86b57c3c2a7bcba47a4f6550b07cd`. Each width used one private daemon. [Daemon setup](serving/lmcache.md#start-the-daemon) |
| vLLM source | `vllm-project/vllm` at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. Use `/data1/shared/dongwu.chen/environments/vllm-81a8cb063e62`. Both runs use `0.34.0rc1.dev787+g81a8cb063`, wheel SHA-256 `25358fa092260117176d78f4efeabfd06a0e49731d29bbcffd717dfedc498ad9`. [Baseline agreement](performance.md#implementations-and-controls) |
| Model and tokenizer | `/data0/shared/Qwen3.8-27B-FP8` for both engines and the replay tokenizer. The checkpoint matches Qwen revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` and supports 262,144 tokens. [Checkpoint record](research/checkpoint.md) |
| Trace | `semianalysisai/cc-traces-weka-062126-256k`, revision `8fecd2fc56694469f758f0afbbb6335ad3043740`, file `traces.jsonl`. SHA-256 `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`. The corpus has 393 trajectories and 68,266 recorded requests. The largest recorded request has 255,999 combined input and output tokens. [Pinned trace][trace] · [Corpus description][corpus] |
| Replay and scoring | `agentx-harness==1.0.14` in its own Python environment. Use InferenceX revision `d490919c996b5169a3e9345ae3c64e32ccfba357` for the published request metrics. [Standalone client][client] · [Metric source][scorer] |

## Run matrix

At each width, the two engines ran in sequence on the same ordered GPU UUIDs, with vLLM first. The native TP1 and TP2 runs overlapped on separate NUMA nodes.

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

The initial native TP2 warmup on B200s 0 and 1 was stopped before timed profiling. The attempt is recorded under `artifacts/agentx-256k/native-preflight/tp2-hang.g5ylCFka/`.

An isolated MXFP8 grouped-query attention probe ran on B200 3 with 1,024 query rows, 12 query heads, and two KV heads. At 131,073 visible tokens the baseline fell back from `tmem_pair_attention` to `protected_prefill_attention`. That boundary increased operator latency by about 126×. The first fix uses tiled FlashAttention at the intermediate lengths below and paired TMEM at 262,144 tokens.

| Visible tokens | Initial baseline (ms) | First fix build (ms) |
| ---: | ---: | ---: |
| 131,072 | 3.894 | 3.893 |
| 131,073 | 489.734 | 15.926 |
| 170,688 | 650.519 | 20.712 |
| 205,568 | 799.688 | 24.942 |
| 262,144 | 1,022.858 | 7.720 |

The table reports the median of four trials for this operator. Both builds matched the probe's causal BF16 reference. The first fix also passed checks against a CPU FP64 reference for nine shapes at each TP width. The maximum absolute errors were below 0.00658 at TP1 and 0.00539 at TP2. The first fix server binary SHA-256 is `dca13339be171794d16440d6128b292a1887054767e615dbfcbe80478c18bd87`.

`artifacts/agentx-256k/stall/diagnosis/` contains `gqa-cliff-baseline.log`, `gqa-cliff-candidate.log`, and the two `gqa-reference-candidate-tp*.log` receipts.

The final GQA change is `835f877b819473fccb1bcbaf297bf50c0d56bebb`. Its shared production dispatch uses paired TMEM for eligible full 1,024-row long tiles and ordered single-CTA TMEM for eligible long partial tiles. The table below gives three-trial medians from its exact archive. [Ordered GQA dispatch](ordered-gqa-prefill.md#constraints-and-failure-behavior)

| Visible tokens | Query rows | Production route | TP1 (ms) | TP2 (ms) |
| ---: | ---: | --- | ---: | ---: |
| 131,073 | 1,024 | Paired TMEM | 8.270 | 4.149 |
| 131,073 | 65 | Ordered single-CTA TMEM | 7.096 | 6.924 |
| 262,144 | 1,024 | Paired TMEM | 15.475 | 7.751 |

The final archive passed six production-route cases, four direct split cases, and the exceptional-repair checker. The six production-route cases had maximum absolute error below 0.00658 at TP1 and 0.00539 at TP2 against FP64. `artifacts/agentx-256k/stall/diagnosis/structural-exact-mODDtWC6/` contains `model-tp*.log`, `split-tp*.log`, `checker-repair.log`, and `identities.sha256`.

## Timed AgentX results

All four runs used seed `42` and a 1,800-second dispatch window after warmup. The native rows predate the [suffix checkpoint policy](prefix-cache.md#checkpoint-boundaries) and ReplaySSM. Closed-loop replay submits dependent turns as predecessors finish. Engines can therefore complete different numbers and mixes of requests. Each row uses its own finalized profiling cohort and response observation span.

| TP | Engine | P90 E2E normalized interactivity (output tokens/s/user) | Input tokens/s/chip | Output tokens/s/chip | Actual KV hit (%) | Theoretical KV hit (%) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Native + LMCache | 13.47 | 26,509.24 | 242.53 | 83.07 | 95.82 |
| 1 | vLLM + Simple KV Offload | 53.66 | 39,992.77 | 365.52 | 94.24 | 95.94 |
| 2 | Native + LMCache | 11.06 | 20,884.43 | 179.15 | 84.22 | 95.65 |
| 2 | vLLM + Simple KV Offload | 53.05 | 36,530.15 | 331.35 | 94.83 | 96.32 |

At TP1, native delivered 33.7% lower input and 33.6% lower output throughput per chip than vLLM. At TP2 the gaps were 42.8% and 45.9%. Native P90 normalized interactivity was 74.9% lower at TP1 and 79.2% lower at TP2. Native actual KV hits were 11.16 and 10.61 percentage points lower, while theoretical rates differed by 0.12 and 0.67 points.

### Effective server and cache settings

Native used 8 and 16 running-request slots at TP1 and TP2, respectively, with the 32,768-token batch limit and 262,144-token context. Both widths used MTP4 with INT8 head verification and online GQA, Tensor Core FP8 prefill, and MXFP8 GQA KV. The server reported `batch_cuda_graphs=true` at TP1 and `false` at TP2.

Native served active KV on the GPUs and persisted reusable prefixes in LMCache CPU L1. Before-to-after counters cover warmup, profiling, and drain. LMCache restored coherent target prefixes of 45,228,032 tokens at TP1 and 76,547,687 at TP2. It acknowledged 1,612 and 2,612 stores, respectively, with zero I/O errors. GQA coverage was 49,421,056 and 82,739,303 tokens, respectively. GQA coverage includes the coherent span and can extend past it. The TP2 profiling-to-drain interval recorded 716 transfers from CPU L1 and 1,952 stores per rank. Read bytes sum over ranks, while logical token spans count once. [Cache accounting](prefix-cache.md#checkpoint-compatibility-and-resume) · [Ranked restores](serving/lmcache.md#ranked-tp2-transport)

Both vLLM runs resolved 32,768 batched and scheduled tokens, with eight sequences at TP1 and 16 at TP2. Both used the 262,144-token context, four-token Qwen MTP, FlashInfer attention, and FP8 E4M3 KV. The scored TP1 restart used 112 GiB of GPU KV after the 128 GiB attempt exhausted memory during warmup. TP2 used 128 GiB of GPU KV per rank. The engine reported 3,038,820 logical GPU KV token slots at TP1 and 6,947,548 at TP2.

Both runs selected `SimpleCPUOffloadConnector` in eager mode. Host capacity was 64 GiB total at TP1 and 128 GiB total at TP2. The before-to-after counters include primers, warmup, and profiling. GPU-local prefix hits were 74,409,088 tokens at TP1 and 138,632,832 at TP2. External lookups covered 5,607,899 and 10,097,443 tokens, respectively, with zero external hits in both runs. The connector recorded 1,899 and 2,979 `stored` outcomes, respectively. All cache reuse in the scored vLLM runs was GPU-local.

### Observation and coverage

Every finalized profiling request had timestamps, trace blocks, final token and cache usage, and a positive-latency P90 sample. The theoretical block totals matched the harness in all four rows. All four runs reported zero request errors and zero context-overflow skips.

| TP | Engine | Finalized / sent | Observation (s) | Chip-seconds | Input tokens | Cached input tokens | Output tokens | Theoretical hit / total blocks |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Native | 602 / 602 | 1,811.559007976 | 1,811.559007976 | 48,023,046 | 39,895,040 | 439,361 | 716,082 / 747,320 |
| 1 | vLLM | 837 / 837 | 1,819.132212561 | 1,819.132212561 | 72,752,129 | 68,558,464 | 664,926 | 1,086,537 / 1,132,567 |
| 2 | Native | 774 / 775 | 1,826.998717913 | 3,653.997435826 | 76,311,656 | 64,267,879 | 654,628 | 1,136,963 / 1,188,620 |
| 2 | vLLM | 1,260 / 1,260 | 1,804.876589299 | 3,609.753178598 | 131,864,832 | 125,046,272 | 1,196,076 | 1,978,892 / 2,054,481 |

The native TP1 profiling-start scrape included 37,641 prompt tokens and 13 generated tokens from the first profiling request. The drained whole-replay deltas reconcile with the finalized warmup and profiling records. Native TP2 canceled one request at the 30-second grace deadline. Its drained engine counters exceeded finalized warmup and profiling SSE by 84,580 input and 4,310 output tokens. The excess input included 81,920 cached tokens. The headline metrics use finalized SSE, so they exclude unfinished work counted by the engine.

### Run receipts

Run files are under `artifacts/agentx-256k/runs/`. The independent native reconciliations are `artifacts/agentx-256k/scoring/native-tp1-reconciliation.json` and `native-tp2-reconciliation.json`. Both reconciled the five reported metrics against the canonical scores.

| Run directory | `score.json` SHA-256 | `run.sha256` SHA-256 |
| --- | --- | --- |
| `native-tp1/` | `930cf7baa9261f12e28da1ce34d92561373935c61c23b0fc8cadb2289b992f94` | `aeb5dd51bce05483b9a7f5aed5cdfbb528fe87db82b75d0575c2412f932d31ce` |
| `vllm-tp1/` | `5633e6b6cce333d3311e1ea4874a15115aa0f8626b1985255428f45d853ff5cb` | `86be678c2a3c44f9d6278f143bd5897bba09298d46c382e173ee79043e37ecaa` |
| `native-tp2/` | `3fa866530cb00c248d032f3265dab31af3b21995a15c0d9db28cc5a5d807ac8f` | `eed33a3a6235593d2abed377663ccca5c02a74f94edb4ca881e1798a74987d13` |
| `vllm-tp2/` | `c79f6d33492c2f1408559f817b5f1020a98c9f5131f78a582082df49fce38e2b` | `a37ddc962e4d830e236f29e397aa5621c47a8c51706eb803094e2397a8431424` |

### Offline suffix checkpoint replay

The offline replay used the same 602 TP1 and 774 TP2 finalized native requests. It matched exact token prefixes to completed earlier requests and assumed that matching records remained resident. The [suffix policy](prefix-cache.md#checkpoint-boundaries) modeled 45,703,556 of 48,023,046 input tokens reusable at TP1 (95.17%). At TP2 it modeled 72,766,151 of 76,311,656 (95.35%). These estimates describe cache coverage under the replay assumptions. Runtime eviction, transfer timing and throughput require a timed run.

The planner can select up to 12 snapshots at 262,144 tokens and selects about seven per request on average in the offline replay. A deep fork can still favor the former periodic grid. For the TP2 fork from trace export row 839 to row 616, the isolated grid comparison loses 24,576 reusable tokens. The exact-token cohort simulation reports a 24,571-token loss for that row against its periodic control.

An independent check reconstructed prompt lengths for 3,999 finalized requests across all four arms. No strictly longer later scored prompt contained the entire prefix of an earlier prompt, including comparisons with in-flight requests. The trace reconstructs historical assistant turns and does not feed back live responses. Generated terminal suffixes therefore added zero reusable tokens in the native offline cohorts. A real follow-up can reuse a terminal checkpoint when it preserves the exact generated token history. See `artifacts/agentx-256k/kvfix/attribution/output-interior.json`, `suffix-policy-tp*.log`, and `fenwick-policy-tp2.log`.

### Focused cache-pressure diagnostic

On GPU 7 with a 64 GiB LMCache pool, the `b0c69845…` predecessor published complete GQA, GDN and draft state at two prime endpoints, 142,099 and 222,663. LMCache recorded two and three evictions between each publication acknowledgement and its follow-up lookup, respectively. Neither follow-up found a usable prefix (0 of 373,710 prompt rows). The sequence wrote 86.52 GiB with the predecessor and 98.82 GiB with the original native build. See `artifacts/agentx-256k/kvfix/verification/compare-pressure-retention-tp1.json` and `retention-pressure-tp1/`.

The predecessor's TP2 diagnostic with a 128 GiB pool restored 298,038 coherent rows out of 660,414, up from 204,800 in its baseline control. One follow-up resumed at the highest common GDN endpoint of 91,573, with GQA coverage of 222,517 and 126,976 on the two ranks. All selected outputs matched their cold controls. These runs used different physical GPU pairs. The comparison reports restored rows only. See `artifacts/agentx-256k/kvfix/verification/compare-pressure-retention-tp2.json` and `retention-pressure-tp2/`.

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

In the integrated `c3c1ba29…` build, K0, mixed eager K4/K2, and captured K4 produced identical sequences of 24 greedy tokens per lane at each width. TP1 recorded 14 actual graph launches and TP2 recorded seven. Both widths rejected tentative rows and recorded zero target reruns. Cancellation and slot reuse passed at both widths. The TP1 state checker passed six checkpoint cases, including captured five-row cuts at 5, 2, and 0. See `artifacts/agentx-256k/replay-integrated.BdBlzC0M/` for `native-graph-proof.tp*.log`, `gdn-verify-check.tp1.log`, and `immutable-check.log`.

The exact operator release, `13f2969a…`, passed isolated CPU-reference and bitwise FP32, BF16, and FP8 checks. Its B200 timing probe used five tentative rows per lane and cycled accepted counts from zero to five. Each entry is the median CUDA-event time in microseconds per verification round. The TP2 rows time rank 0 on one GPU. See `artifacts/replay-operator.kHHz3mSs/release/` for `mixed-timing.log`, `mixed_probe.cu`, the parity logs and `source.sha256`.

| GDN shape | Lanes | Dense verify | Dense verify + GPU repair | ReplaySSM verify + commit |
| --- | ---: | ---: | ---: | ---: |
| TP1 | 8 | 36.61 | 54.72 | 52.03 |
| TP1 | 16 | 79.92 | 122.59 | 93.66 |
| TP2 rank 0 | 8 | 26.67 | 37.62 | 34.42 |
| TP2 rank 0 | 16 | 48.02 | 69.01 | 53.65 |

[methodology]: https://inferencex.semianalysis.com/agentx/methodology
[leaderboard]: https://inferencex.semianalysis.com/zh/infx/inference/deepseek-v41-flash
[trace]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/blob/8fecd2fc56694469f758f0afbbb6335ad3043740/traces.jsonl
[corpus]: https://inferencex.semianalysis.com/agentx/cc-traces-weka-062126-256k
[client]: https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/docs/agentx-standalone.md
[scorer]: https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/infx/results/agentic/request_metrics.py
[offload]: https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/vllm.py#L1231
[capacity]: https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L412
