# AgentX 256k benchmark

Compare native serving with LMCache against previously measured vLLM with Simple KV Offload on NVIDIA B200. Use the [AgentX methodology][methodology] for replay and the [InferenceX DeepSeek V4.1 Flash page][leaderboard] as a reporting reference. Both engines used the local Qwen checkpoint and a 30-minute profiling window.

## Frozen inputs

| Input | Identity |
| --- | --- |
| Native source | `b0596c25eab50dfbe4b5a340da3a06f84f89b339`, tree `b250c6c96860e860ca4983d78c9b2bac02e6e06f`, Release binary SHA-256 `59748dd6424d97019d49360fe42d87ea9d6aaaf5472ed98cdfc19b41a3a806d8`. Release manifest `artifacts/pool-w6/release/b0596c25eab50dfbe4b5a340da3a06f84f89b339/official-release.sha256`, SHA-256 `9f40fb17faec4914bfb845584bc1afacc13c30f744b89f2d062d0fd7639671e3`. |
| Native LMCache | Wheel `0.5.8`, SHA-256 `4ec775bda3a8c5f1f5b1ef0bbc441b666db86b57c3c2a7bcba47a4f6550b07cd`. Each width used one private daemon. [Daemon setup](serving/lmcache.md#start-the-daemon) |
| vLLM source | `vllm-project/vllm` at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. Use `/data1/shared/dongwu.chen/environments/vllm-81a8cb063e62`. Both runs use `0.34.0rc1.dev787+g81a8cb063`, wheel SHA-256 `25358fa092260117176d78f4efeabfd06a0e49731d29bbcffd717dfedc498ad9`. [Pinned vLLM environment](../benchmarks/vllm/README.md) |
| Model and tokenizer | `/data0/shared/Qwen3.8-27B-FP8` for both engines and the replay tokenizer. The checkpoint matches Qwen revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` and supports 262,144 tokens. [Checkpoint record](research/checkpoint.md) |
| Trace | `semianalysisai/cc-traces-weka-062126-256k`, revision `8fecd2fc56694469f758f0afbbb6335ad3043740`, file `traces.jsonl`. SHA-256 `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`. The corpus has 393 trajectories and 68,266 recorded requests. The largest recorded request has 255,999 combined input and output tokens. [Pinned trace][trace] · [Corpus description][corpus] |
| Replay and scoring | `agentx-harness==1.0.14` in its own Python environment. The [client lock](../benchmarks/agentx/pixi.lock) has SHA-256 `710a36ec0e441a4a482362524c811ee9db12863d9050f37a7c736f62ebfe6863`. Use InferenceX revision `d490919c996b5169a3e9345ae3c64e32ccfba357` for the published request metrics. [Standalone client][client] · [Metric source][scorer] |

## Run matrix

At each width, native and vLLM used the same ordered GPU UUIDs. The two native profiles overlapped on separate NUMA nodes.

| TP | Engine | CPU cache capacity | Physical GPUs in rank order | Live agent clients | Profile |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | Native + LMCache 0.5.8 | 64 GiB | B200 6 | 8 | 1,800 s |
| 1 | vLLM + Simple KV Offload | 64 GiB | B200 6 | 8 | 1,800 s |
| 2 | Native + LMCache 0.5.8 | 128 GiB | B200 1, 2 | 16 | 1,800 s |
| 2 | vLLM + Simple KV Offload | 128 GiB | B200 1, 2 | 16 | 1,800 s |

TP1 uses B200 6 (`GPU-d4d87d81-ea84-5dd0-7167-3e1c48fde29b`) on NUMA node 1. Pin its server to CPUs `72-81` and its client to `82-83`. TP2 uses NUMA node 0. Its rank 0 is B200 1 (`GPU-9a94015a-6a0d-1ae1-82db-b6736b77562e`). Its rank 1 is B200 2 (`GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c`). Pin the TP2 server to CPUs `12-21,24-33` and its client to `22-23,34-35`. Pin the LMCache daemon to the client CPUs. Confirm UUIDs, rank placement, and availability at startup.

## Execution

1. Freeze the launch commands before profiling. Set `--max-num-batched-tokens 32768` on both engines at TP1 and TP2. Use 8 running-request slots at TP1 and 16 at TP2. Verify the resolved limits and 262,144-token context capacity in every arm. Match and record CPU affinity, NUMA placement, and tokenizer. The replay relies on server sampling defaults. Record the effective sampling parameters, KV format, scheduler limits, speculative method, and draft depth for each engine. A published golden SPEED-Bench acceptance calibration is unavailable for this exact checkpoint.
2. Start a fresh engine for each arm. For native, start a private [LMCache daemon](serving/lmcache.md#start-the-daemon) with the matrix capacity and UUIDs. Set `--gpu-memory-utilization 0.88` and use automatic GQA page and GDN checkpoint capacities. Pass the daemon socket to `sglang-server` with `--lmcache-socket`. Use one daemon and the same ordered UUIDs for both [TP2 ranks](serving/tp2.md#prerequisites-and-command).
3. For vLLM, use `vllm serve` in the pinned environment with `--tensor-parallel-size` set to the matrix width, `--max-num-seqs` set to the running-request slots, and `--max-model-len 262144`. Set `VLLM_USE_SIMPLE_KV_OFFLOAD=1` and `--kv-offloading-backend native`. Set `--kv-offloading-size` to the matrix capacity across all ranks. Enable prefix caching and `--enable-prompt-tokens-details`. Verify that the worker selects `SimpleCPUOffloadConnector`. [Pinned environment](../benchmarks/vllm/README.md) · [Connector selection][offload] · [Capacity option][capacity]
4. Confirm each engine serves streaming `/v1/chat/completions` with the local tokenizer and context capacity. Check the terminal usage for prompt, completion, and cached prompt tokens. Preserve the readiness receipt. Restart with a fresh cache if the readiness requests populated it.
5. Verify the pinned trace bytes before invoking AgentX. Verify the preset resolves seed `42`, a 25–75% trajectory start range, and `first_turn_prefix` cache busting. Use ten warmup requests per lane. Retain the trace timing and subagent dependencies. Use one live agent client per lane, with eight lanes per GPU. Subagent fan-out can create more simultaneous HTTP requests than live clients. [Replay methodology][methodology]
6. Verify the resolved plan before starting. Dispatch profiling requests for 1,800 seconds after the warmup barrier. Allow 30 seconds for outstanding requests after dispatch closes. Include valid profiling responses received during that grace period. Exclude primers and warmup requests. Record the phase times and drain deadline.

Put Pixi on `PATH` and stage the pinned trace at `/data1/shared/dongwu.chen/cache/agentx-256k-client/traces.jsonl`. Use the proxy at `http://127.0.0.1:7890` to fetch dependencies and the pinned corpus. Set `TP` to `1` or `2`, `CLIENT_CPUS` from the run matrix, `SERVER_URL` and `SERVED_MODEL` to the active server, and `RUN_DIR` to a new output directory. Run from the repository root. The wrapper verifies the trace digest, stages the dataset name recognized by the upstream loader, and uses the locked client environment for offline replay. [Pinned trace][trace] · [AgentX replay command](../benchmarks/agentx/run-agentx.sh#L39)<!-- prefix: "agentx profi" -->

```bash
taskset --cpu-list "$CLIENT_CPUS" bash benchmarks/agentx/run-agentx.sh \
  "$TP" "$SERVER_URL" "$SERVED_MODEL" "$RUN_DIR"
```

## Scoring and evidence

Score profiling turns that have successful transport, a finish choice, and `[DONE]`. Read token counts from the final usage chunk. Report failures, timeouts, cancellations, context-overflow skips, and usage coverage. Measure the observation span from the earliest retained request start to the latest retained response end.

Score the wrapper's raw export with the captured dataset manifest. [AgentX scorer](../benchmarks/agentx/score.py#L302)<!-- prefix: "def main() -" -->

```bash
python3 benchmarks/agentx/score.py "$RUN_DIR" --chips "$TP" \
  --manifest "$RUN_DIR/dataset-manifest.json"
```

| Metric | Unit | Scoring input and method |
| --- | --- | --- |
| P90 end-to-end normalized interactivity | output tokens/s/user | Take the reciprocal of the linearly interpolated 90th percentile of request end-to-end seconds per output token. Use positive latencies and output lengths. [InferenceX metric][scorer] |
| Input token throughput per chip | input tokens/s/chip | Divide server-reported input tokens by the observation span and physical chip count. Include cached input tokens in the count. [Scorer][scorer] |
| Output token throughput per chip | output tokens/s/chip | Divide server-reported completion tokens by the same span and chip count. [Scorer][scorer] |
| Actual KV cache hit rate | % | Divide the sum of server-reported `usage.prompt_tokens_details.cached_tokens` by the sum of `usage.prompt_tokens` over the same turns. Count logical tokens once at TP2. Report usage coverage and use engine counters to corroborate reuse. [Native final stream usage](../apps/server/text_completion.cpp#L591)<!-- prefix: "void TextRes" --> |
| Theoretical KV cache hit rate | % | Divide the replay's hit blocks by its total 64-token blocks over the same finalized turns. Keep the harness aggregate `theoretical_prefix_cache_hit` as a cross-check. Report any difference in the scoring cohorts. [InferenceX metric][scorer] |

The theoretical KV hit rate is fixed for an identical scored trace cohort, including the
same replay turns, input block counts, and trace-derived hit blocks. A difference between
closed-loop runs can reflect which turns finalized. Use theoretical hit rate to describe
the scored cohort and compare actual hit rate on matched turns to assess cache reuse.
Engine improvements can raise actual reuse, interactivity, or throughput while the
theoretical rate stays fixed. [Pinned scorer][scorer]

Keep one evidence directory per run under `artifacts/agentx-256k/`. Record its path and a SHA-256 manifest with the results. Preserve the following material:

- Exact command lines, effective configs, build and dependency receipts, and binary or wheel hashes. Include model, tokenizer, trace, client, and scorer identities.
- GPU UUIDs and rank order, topology, clock and power settings, driver versions, and competing-process snapshots. Record CPU and memory placement.
- Replay plan, phase timestamps, raw per-request SSE in `profile_export_raw.jsonl`, `profile_export_aiperf.json`, `dataset-manifest.json`, and logs. Preserve request IDs, phase membership, latency, terminal stream status, and final usage. The dataset manifest SHA-256 is `b6605c6ebae6c731079e5edf7d441f290a794d239ef01092a114e4f4e939b703`.
- Raw `/metrics` samples at warmup and profiling boundaries and after drain. Retain native LMCache activity and vLLM local and offload activity with the engine receipts.
- Scoring invocation, numerators, denominators, sample counts, and any exclusion reason.

## Timed AgentX results

Closed-loop dependencies can change the mix of finalized requests. Metrics use each run’s own finalized cohort and observation span.

| TP | Engine | P90 E2E normalized interactivity (output tokens/s/user) | Input tokens/s/chip | Output tokens/s/chip | Actual KV hit (%) | Theoretical KV hit (%) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Native + LMCache | 15.92 | 32,105.11 | 280.34 | 89.62 | 96.06 |
| 1 | vLLM + Simple KV Offload | 53.66 | 39,992.77 | 365.52 | 94.24 | 95.94 |
| 2 | Native + LMCache | 11.86 | 21,646.81 | 188.91 | 89.76 | 95.76 |
| 2 | vLLM + Simple KV Offload | 53.05 | 36,530.15 | 331.35 | 94.83 | 96.32 |

Native actual KV hit rates were 4.61 percentage points below vLLM at TP1 and 5.07 points below at TP2. They were 6.43 and 6.00 points below replay theory. Compared with vLLM, native P90 interactivity was 70.3% lower at TP1 and 77.6% lower at TP2. Input throughput per chip was 19.7% and 40.7% lower. Output throughput per chip was 23.3% and 43.0% lower.

### Observation and coverage

The client used seed `42` and dispatched for 1,800.000 s at both native widths. Each finalized turn had timing, final token and cache usage, trace blocks, and a positive-latency P90 sample. The theoretical block totals matched the harness. All four arms had zero errors or context-overflow skips.

| TP | Engine | Finalized / sent | Observation (s) | Chip-seconds | Input tokens | Cached input tokens | Output tokens | Theoretical hit / total blocks |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Native | 700 / 700 | 1,819.387376379 | 1,819.387376379 | 58,411,631 | 52,351,147 | 510,054 | 873,305 / 909,140 |
| 1 | vLLM | 837 / 837 | 1,819.132212561 | 1,819.132212561 | 72,752,129 | 68,558,464 | 664,926 | 1,086,537 / 1,132,567 |
| 2 | Native | 794 / 795 | 1,820.045293517 | 3,640.090587034 | 78,796,354 | 70,725,911 | 687,645 | 1,175,325 / 1,227,395 |
| 2 | vLLM | 1,260 / 1,260 | 1,804.876589299 | 3,609.753178598 | 131,864,832 | 125,046,272 | 1,196,076 | 1,978,892 / 2,054,481 |

Native TP1 engine counters matched finalized warmup and profiling SSE exactly. Native TP2 canceled one request at the 30-second grace deadline. That request had no exported raw row and was excluded from the score. TP2 whole-run engine counters exceeded finalized warmup and profiling SSE by 136,993 input, 7,483 output, and 136,993 cached input tokens. The audit found this gap consistent with the cancellation.

### Effective server and cache settings

Native used 8/16 running-request slots, a 32,768-token batch limit, 262,144 context, and four-token MTP with model sampling defaults. Target GQA used `grouped-split`. At `--gpu-memory-utilization 0.88`, the automatic HBM prefix pools reported 16,384 GQA pages and 49 GDN checkpoint slots at TP1. TP2 reported 32,768 pages and 107 slots. The native GQA and draft caches used MXFP8. LMCache 0.5.8 reported `0.5.8+native-v5` over SLC3. [Native prefix cache](prefix-cache.md#checkpoint-compatibility-and-resume)

The archived replay payloads omitted temperature, top-k, and top-p. Native used checkpoint defaults (`temperature=1.0`, `top_k=20`, `top_p=0.95`). Both vLLM arms used `--generation-config vllm`, which selected vLLM defaults (`temperature=1.0`, `top_k=0`, `top_p=1.0`). The measurements compare those effective policies. [Run receipts](#run-receipts) · [vLLM default resolution](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/model.py#L1978)

At TP1, each engine used one GPU for the target and MTP draft. At TP2, native split the target across both ranks and ran the MTP draft on rank 0. The pinned vLLM TP2 configuration recorded tensor parallel size 2 for both the target and draft.

Across warmup, profiling, and drain, native reused 58,096,283 coherent target prompt tokens at TP1 and 84,306,125 at TP2. Read-only GQA reuse extended beyond the coherent span by 3,607,563 rows at TP1 and 4,106,343 at TP2. Both widths read GDN data from LMCache and recorded zero I/O errors.

Both vLLM widths used four-token Qwen MTP, FlashInfer attention, FP8 E4M3 KV, and `SimpleCPUOffloadConnector`. GPU KV capacity was 112 GiB at TP1 and 128 GiB per rank at TP2. Both arms recorded CPU stores with zero external hits. All measured vLLM cache hits came from GPUs.

### Run receipts

The native runs are under `artifacts/agentx-256k/b0596c25/campaign/runs/attempt-02/`. The independent reconciliation is `artifacts/agentx-256k/b0596c25/verification/audit-b0596c25.json`, SHA-256 `20d0bf903909acada7d494f63170805311bf2f556fe35a05009fe93da1cc81bc`. The vLLM receipts under `artifacts/agentx-256k/runs/` are from the previously measured runs.

The native run directories each contain `server-command.sh` and `lmcache-command.sh`. Each vLLM run directory contains `server/server-command.sh` and `server/live-argv.bin`.

| Run directory | `score.json` SHA-256 | `run.sha256` SHA-256 |
| --- | --- | --- |
| `native-tp1/` | `5cfef92c7af992392d4ec846ce0fdb039a42b3fd2474904fce0e9e93fbf5e100` | `1a5ecd0024f01c9081e5877b8b3bab38b96470337a1d66bbe53f403d3b5fddfe` |
| `vllm-tp1/` | `5633e6b6cce333d3311e1ea4874a15115aa0f8626b1985255428f45d853ff5cb` | `86be678c2a3c44f9d6278f143bd5897bba09298d46c382e173ee79043e37ecaa` |
| `native-tp2/` | `22c4624f6eb1a8864508f66d9df1aad84e1571bda46a0278a7731c819857eaf2` | `c97d42587cccd55d8a3fac5a3a0ab4d437bb36d2a3dce0615295c0dde1ccbf9f` |
| `vllm-tp2/` | `c79f6d33492c2f1408559f817b5f1020a98c9f5131f78a582082df49fce38e2b` | `a37ddc962e4d830e236f29e397aa5621c47a8c51706eb803094e2397a8431424` |

[methodology]: https://inferencex.semianalysis.com/agentx/methodology
[leaderboard]: https://inferencex.semianalysis.com/zh/infx/inference/deepseek-v41-flash
[trace]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/blob/8fecd2fc56694469f758f0afbbb6335ad3043740/traces.jsonl
[corpus]: https://inferencex.semianalysis.com/agentx/cc-traces-weka-062126-256k
[client]: https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/docs/agentx-standalone.md
[scorer]: https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/infx/results/agentic/request_metrics.py
[offload]: https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/vllm.py#L1231
[capacity]: https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L412
