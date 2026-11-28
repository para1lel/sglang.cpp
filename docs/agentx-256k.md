# AgentX 256k benchmark

Compare native serving with LMCache against pinned vLLM with Simple KV Offload on NVIDIA B200. Use the [AgentX methodology][methodology] for replay and the [InferenceX DeepSeek V4.1 Flash page][leaderboard] as a reporting reference. This campaign uses a local Qwen checkpoint and a 30-minute profiling window.

## Frozen inputs

| Input | Identity |
| --- | --- |
| Native source | `bdaec36dbb985edd57e54b3565315ba321290a23`, tree `982baef6e1bfe7113f3b10e145c1ae4a0024bf7d`, Release binary SHA-256 `d83223fdcf988f24b7d919ad97b499757e9abcd1876127ff174ddb0a9282fa5d`. The release receipt is `artifacts/agentx-256k/kvfix/official-peer-release/release-identity.json`. |
| Native LMCache | Wheel `0.5.8`, SHA-256 `4ec775bda3a8c5f1f5b1ef0bbc441b666db86b57c3c2a7bcba47a4f6550b07cd`. Each width used one private daemon. [Daemon setup](serving/lmcache.md#start-the-daemon) |
| vLLM source | `vllm-project/vllm` at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. Use `/data1/shared/dongwu.chen/environments/vllm-81a8cb063e62`. Both runs use `0.34.0rc1.dev787+g81a8cb063`, wheel SHA-256 `25358fa092260117176d78f4efeabfd06a0e49731d29bbcffd717dfedc498ad9`. [Baseline agreement](performance.md#implementations-and-controls) |
| Model and tokenizer | `/data0/shared/Qwen3.8-27B-FP8` for both engines and the replay tokenizer. The checkpoint matches Qwen revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` and supports 262,144 tokens. [Checkpoint record](research/checkpoint.md) |
| Trace | `semianalysisai/cc-traces-weka-062126-256k`, revision `8fecd2fc56694469f758f0afbbb6335ad3043740`, file `traces.jsonl`. SHA-256 `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`. The corpus has 393 trajectories and 68,266 recorded requests. The largest recorded request has 255,999 combined input and output tokens. [Pinned trace][trace] · [Corpus description][corpus] |
| Replay and scoring | `agentx-harness==1.0.14` in its own Python environment. Use InferenceX revision `d490919c996b5169a3e9345ae3c64e32ccfba357` for the published request metrics. [Standalone client][client] · [Metric source][scorer] |

## Run matrix

At each width, native and vLLM used the same ordered GPU UUIDs. vLLM ran first. The native TP1 and TP2 profiles overlapped on separate NUMA nodes.

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

## Timed AgentX results

Closed-loop dependencies can change the mix of finalized requests. Metrics use each run’s own finalized cohort and observation span.

| TP | Engine | P90 E2E normalized interactivity (output tokens/s/user) | Input tokens/s/chip | Output tokens/s/chip | Actual KV hit (%) | Theoretical KV hit (%) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Native + LMCache | 12.67 | 26,915.62 | 245.97 | 85.15 | 95.85 |
| 1 | vLLM + Simple KV Offload | 53.66 | 39,992.77 | 365.52 | 94.24 | 95.94 |
| 2 | Native + LMCache | 9.09 | 20,471.22 | 173.00 | 84.04 | 95.70 |
| 2 | vLLM + Simple KV Offload | 53.05 | 36,530.15 | 331.35 | 94.83 | 96.32 |

Actual native KV hit rates were 9.08 percentage points below vLLM at TP1 and 10.79 points below at TP2. They were 10.70 and 11.66 points below the replay theory. Native P90 interactivity was 76.4% and 82.9% lower than vLLM at TP1 and TP2. Input throughput per chip was 32.7% and 44.0% lower. Output throughput per chip was 32.7% and 47.8% lower.

### Observation and coverage

The client used seed `42` and logged 1,800.000 s of native TP1 dispatch and 1,800.001 s at TP2. Each finalized turn had timing, final token and cache usage, trace blocks, and a positive-latency P90 sample. The theoretical block totals matched the harness. All four arms had zero errors or context-overflow skips.

| TP | Engine | Finalized / sent | Observation (s) | Chip-seconds | Input tokens | Cached input tokens | Output tokens | Theoretical hit / total blocks |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Native | 613 / 613 | 1816.140940800 | 1816.140940800 | 48,882,565 | 41,623,874 | 446,718 | 729,159 / 760,719 |
| 1 | vLLM | 837 / 837 | 1,819.132212561 | 1,819.132212561 | 72,752,129 | 68,558,464 | 664,926 | 1,086,537 / 1,132,567 |
| 2 | Native | 761 / 762 | 1825.311244764 | 3650.622489528 | 74,732,703 | 62,804,320 | 631,546 | 1,113,894 / 1,163,989 |
| 2 | vLLM | 1,260 / 1,260 | 1,804.876589299 | 3,609.753178598 | 131,864,832 | 125,046,272 | 1,196,076 | 1,978,892 / 2,054,481 |

Native TP1 engine counters reconciled exactly with finalized warmup and profiling SSE. Native TP2 canceled one request at the 30-second grace deadline. Drained counters included 185,086 input and 9,256 output tokens beyond finalized SSE, with zero excess cached tokens. The scored cohort excludes that request.

### Effective server and cache settings

Native used 8/16 running-request slots, the 32,768-token batch limit, 262,144 context, and four-token MTP. LMCache 0.5.8 reported native-v4 over SLC3.

The drained whole replay restored 47,215,357 coherent target tokens at TP1 and 75,923,335 at TP2, with zero LMCache I/O errors. Read-only GQA coverage beyond the coherent span accounted for 2,491,722 and 4,324,357 rows respectively. These totals cover warmup, profiling, and drain for each closed-loop cohort.

Both vLLM widths used four-token Qwen MTP, FlashInfer attention, FP8 E4M3 KV, and eager `SimpleCPUOffloadConnector`. GPU KV capacity was 112 GiB at TP1 and 128 GiB per rank at TP2. Both arms recorded CPU stores with zero external hits. All measured vLLM cache hits came from GPUs.

### Run receipts

Run files for the current native release are under `artifacts/agentx-256k/kvfix/campaign/runs/d83223fdcf988f24-f5533069302f2199/attempt-02/`. The independent reconciliation is `artifacts/agentx-256k/kvfix/verification/audit-bdaec36-attempt-02.json`. The vLLM runs are under `artifacts/agentx-256k/runs/`.

| Run directory | `score.json` SHA-256 | `run.sha256` SHA-256 |
| --- | --- | --- |
| `native-tp1/` | `65efd52e7f68e48c907ed20e7b90790915e860cfcb8d2e3bbd3d5d1e1bbaf79c` | `9eed44d6e701027a05916ac0b9c8f9961289976d9ca7df5414e0992dfd37bfec` |
| `vllm-tp1/` | `5633e6b6cce333d3311e1ea4874a15115aa0f8626b1985255428f45d853ff5cb` | `86be678c2a3c44f9d6278f143bd5897bba09298d46c382e173ee79043e37ecaa` |
| `native-tp2/` | `64fa764b4fb86e353495a378b855eeb8496c513048d24926e51ec51a205d8130` | `2d733b6021fc66b25440eb0c5ca4557358c8b0f3e9604672ef6b98971ebf3cff` |
| `vllm-tp2/` | `c79f6d33492c2f1408559f817b5f1020a98c9f5131f78a582082df49fce38e2b` | `a37ddc962e4d830e236f29e397aa5621c47a8c51706eb803094e2397a8431424` |

[methodology]: https://inferencex.semianalysis.com/agentx/methodology
[leaderboard]: https://inferencex.semianalysis.com/zh/infx/inference/deepseek-v41-flash
[trace]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/blob/8fecd2fc56694469f758f0afbbb6335ad3043740/traces.jsonl
[corpus]: https://inferencex.semianalysis.com/agentx/cc-traces-weka-062126-256k
[client]: https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/docs/agentx-standalone.md
[scorer]: https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/infx/results/agentic/request_metrics.py
[offload]: https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/vllm.py#L1231
[capacity]: https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L412
