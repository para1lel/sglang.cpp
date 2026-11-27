# AgentX 256k benchmark protocol

Compare native serving with LMCache against pinned vLLM with Simple KV Offload on NVIDIA B200. Use the [AgentX methodology][methodology] for replay and the [InferenceX DeepSeek V4.1 Flash page][leaderboard] as a reporting reference. This campaign uses a local Qwen checkpoint and a 30-minute profiling window.

## Frozen inputs

| Input | Identity |
| --- | --- |
| Native source | `3813c20f53a27f31dadb5698ef3d319af25f4b99` on `bench/agentx-256k-tp1-tp2`. Build the server and take the LMCache integration from this source. |
| vLLM source | `vllm-project/vllm` at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. Use `/data1/shared/dongwu.chen/environments/vllm-81a8cb063e62`. [Baseline agreement](performance.md#implementations-and-controls) |
| Model and tokenizer | `/data0/shared/Qwen3.8-27B-FP8` for both engines and the replay tokenizer. The checkpoint matches Qwen revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` and supports 262,144 tokens. [Checkpoint record](research/checkpoint.md) |
| Trace | `semianalysisai/cc-traces-weka-062126-256k`, revision `8fecd2fc56694469f758f0afbbb6335ad3043740`, file `traces.jsonl`. SHA-256 `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`. The corpus removes turns above 256,000 combined input and output tokens. [Pinned trace][trace] · [Corpus description][corpus] |
| Replay and scoring | `agentx-harness==1.0.14` in its own Python environment. Use InferenceX revision `d490919c996b5169a3e9345ae3c64e32ccfba357` for the published request metrics. [Standalone client][client] · [Metric source][scorer] |

## Run matrix

Run the four arms serially in the order below. Lock the assigned GPUs and use the same ordered UUIDs for both engines at each TP width.

| TP | Engine | CPU cache capacity | Physical GPUs in rank order | Live agent clients | Profile |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | Native + LMCache 0.5.8 | 64 GiB | B200 6 | 8 | 1,800 s |
| 1 | vLLM + Simple KV Offload | 64 GiB | B200 6 | 8 | 1,800 s |
| 2 | Native + LMCache 0.5.8 | 128 GiB | B200 6, 7 | 16 | 1,800 s |
| 2 | vLLM + Simple KV Offload | 128 GiB | B200 6, 7 | 16 | 1,800 s |

B200 6 is `GPU-d4d87d81-ea84-5dd0-7167-3e1c48fde29b`. B200 7 is `GPU-3c08cbc3-d8b3-c723-c086-a775f6a17f2a`. Both are on NUMA node 1 with an NV18 link between them. Confirm the UUIDs, rank placement, and availability at startup.

## Execution

1. Freeze the launch commands before profiling. Match and record CPU affinity, NUMA placement, tokenizer, sampling, context limits, and speculative decoding settings. Record the effective KV formats, scheduler limits, and differences between engines.
2. Start a fresh engine for each arm. For native, start a private [LMCache daemon](serving/lmcache.md#start-the-daemon) with the matrix capacity and UUIDs. Pass its socket to `sglang-server` with `--lmcache-socket`. Use one daemon and the same ordered UUIDs for both [TP2 ranks](serving/tp2.md#prerequisites-and-command).
3. For vLLM, set `VLLM_USE_SIMPLE_KV_OFFLOAD=1` and `--kv-offloading-backend native`. Set `--kv-offloading-size` to the matrix capacity across all ranks. Enable prefix caching and `--enable-prompt-tokens-details`. Verify that the worker selects `SimpleCPUOffloadConnector`. [Connector selection][offload] · [Capacity option][capacity]
4. Confirm each engine serves streaming `/v1/chat/completions` with the local tokenizer and context capacity. Check the terminal usage for prompt, completion, and cached prompt tokens. Preserve the readiness receipt. Restart with a fresh cache if the readiness requests populated it.
5. Verify the pinned trace bytes before invoking AgentX. Use the seeded 25–75% trajectory start range, cache busting per replay, and ten warmup requests per lane. Retain the trace timing and subagent dependencies. Use one live agent client per lane, with eight lanes per GPU. Subagent fan-out can create more simultaneous HTTP requests than live clients. [Replay methodology][methodology]
6. Verify the resolved client plan and profile for 1,800 seconds after the warmup barrier. Drain dispatched requests under the same client policy for every arm. Record the phase times, drain deadline, and whether responses after dispatch closes enter the score. Keep warmup and primers out of the scored cohort.

Run the pinned client with `SERVER_URL` and `SERVED_MODEL` matching the active server. Set `CONC` from the matrix, `TRACE` to the verified local JSONL file, and `RUN_DIR` to a fresh run directory. Use the proxy at `http://127.0.0.1:7890` for downloads as needed. Send loopback requests directly.

```bash
set -o pipefail
mkdir -p "$RUN_DIR"
agentx profile \
  --url "$SERVER_URL" --endpoint /v1/chat/completions \
  --model "$SERVED_MODEL" --tokenizer /data0/shared/Qwen3.8-27B-FP8 \
  --input-file "$TRACE" --custom-dataset-type weka_trace \
  --concurrency "$CONC" --benchmark-duration 1800 --random-seed 42 \
  --use-server-token-count --server-metrics "$SERVER_URL/metrics" \
  --export-level raw \
  --output-artifact-dir "$RUN_DIR/aiperf_artifacts" \
  2>&1 | tee "$RUN_DIR/aiperf.log"
```

## Scoring and evidence

Use valid completed profiling turns and report their coverage. Keep failure, timeout, and cancellation counts with each result. Save the actual observation span used by the throughput scorer, from the earliest retained request start to the latest retained response end. The 1,800-second setting controls the dispatch window.

| Metric | Unit | Scoring input and method |
| --- | --- | --- |
| P90 end-to-end normalized interactivity | output tokens/s/user | Use `p90_e2e_norm_intvty` from the pinned scorer. It takes the reciprocal of the 90th percentile of request end-to-end seconds per output token. [Scorer][scorer] |
| Input token throughput per chip | input tokens/s/chip | Divide server-reported input tokens by the observation span and physical chip count. Include cached input tokens in the count. [Scorer][scorer] |
| Output token throughput per chip | output tokens/s/chip | Divide server-reported completion tokens by the same span and chip count. [Scorer][scorer] |
| Revenue per chip per hour | CNY/chip/hour | Estimate gross revenue at the Qwen3.8-27B Singapore list tariff. Treat actual cached prompt tokens as implicit cache reads. [Alibaba Cloud pricing][pricing] |
| Actual KV cache hit rate | % | Divide the sum of server-reported `usage.prompt_tokens_details.cached_tokens` by the sum of `usage.prompt_tokens` over the same turns. Count logical tokens once at TP2. Report usage coverage and use engine counters to corroborate reuse. [Native final stream usage](../apps/server/text_completion.cpp#L591)<!-- prefix: "void TextRes" --> |
| Theoretical KV cache hit rate | % | Use the replay's aggregate `theoretical_prefix_cache_hit` and the pinned scorer. Retain the trace-block hit numerator and total block count. [Scorer][scorer] |

For the revenue estimate, let `I`, `C`, and `O` be the retained input, cached input, and output token totals. Apply `3.646` CNY per million uncached input tokens, `0.729` per million implicit cache reads, and `21.875` per million output tokens. Multiply `(3.646 × (I − C) + 0.729 × C + 21.875 × O) / 1,000,000` by `3,600 / (observation seconds × chips)`. Record the pricing snapshot with the estimate. [Singapore tariff][pricing]

Keep one evidence directory per run under `artifacts/agentx-256k/`. Record its path and a SHA-256 manifest with the results. Preserve the following material:

- Exact command lines, effective configs, build and dependency receipts, and binary or wheel hashes. Include model, tokenizer, trace, client, and scorer identities.
- GPU UUIDs and rank order, topology, clock and power settings, driver versions, and competing-process snapshots. Record CPU and memory placement.
- Replay plan, phase timestamps, raw per-request SSE export, aggregate metrics, and client and server logs. Preserve request IDs, phase membership, latency, terminal stream status, and final usage.
- Raw `/metrics` samples at warmup and profiling boundaries and after drain. Retain native LMCache activity and vLLM local and offload activity with the engine receipts.
- Scoring invocation, numerators, denominators, sample counts, and any exclusion reason. Preserve the tariff source and access date.

[methodology]: https://inferencex.semianalysis.com/agentx/methodology
[leaderboard]: https://inferencex.semianalysis.com/zh/infx/inference/deepseek-v41-flash
[trace]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/blob/8fecd2fc56694469f758f0afbbb6335ad3043740/traces.jsonl
[corpus]: https://inferencex.semianalysis.com/agentx/cc-traces-weka-062126-256k
[client]: https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/docs/agentx-standalone.md
[scorer]: https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/infx/results/agentic/request_metrics.py
[offload]: https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/vllm.py#L1231
[capacity]: https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L412
[pricing]: https://help.aliyun.com/en/model-studio/qwen3-8-27b#pricing
