# AgentX 256k benchmark protocol

Compare sglang.cpp with direct GPU LMCache against pinned vLLM with Simple KV Offload on B200s. Serve the same Qwen3.8-27B-FP8 checkpoint from `/data0/shared/Qwen3.8-27B-FP8`. This protocol defines a local comparison using the official AgentX trace and harness. Each arm has a 1,800-second profiling window, overriding the AgentX preset’s 3,600 seconds.

## Run matrix

| Arm | TP | B200s | Live agent clients | CPU KV configuration |
| --- | ---: | ---: | ---: | --- |
| sglang.cpp | 1 | 1 | 8 | One LMCache 0.5.8 daemon with a 64 GiB L1 |
| vLLM | 1 | 1 | 8 | Simple KV Offload with a 64 GiB server-wide budget |
| sglang.cpp | 2 | 2 | 16 | One LMCache 0.5.8 daemon with a 128 GiB shared L1 |
| vLLM | 2 | 2 | 16 | Simple KV Offload with a 128 GiB server-wide budget |

Run the two arms at each TP width serially on identical physical GPU UUIDs in the same rank order. Record the UUIDs and TP2 interconnect. AgentX concurrency counts live agent clients. Subagent branches can increase the number of simultaneous HTTP requests.

Use the native Release server with `--tp` and `--max-active` set to the matrix width and client count, respectively, and a private `--lmcache-socket`. Match the daemon’s visible GPU set to the server’s. Use vLLM revision `81a8cb063e6247fe254d7caf7e98a6691d71022a` in its separate pinned environment. Set `--tensor-parallel-size` and `--max-num-seqs` to the matrix width and client count, respectively. Set `VLLM_USE_SIMPLE_KV_OFFLOAD=1`, `--kv-offloading-backend native`, and `--kv-offloading-size` to the matrix’s server-wide GiB budget. Enable prefix caching and `--enable-prompt-tokens-details`. Verify `SimpleCPUOffloadConnector` and record its effective per-rank capacity. [Native LMCache setup](serving/lmcache.md#start-the-daemon) · [Pinned vLLM environment](performance.md#implementations-and-controls)

Use the same tokenizer and sampling policy, and preserve the trace’s prompt and output budgets. Use `--max-model-len 262144` on vLLM and verify the native server’s 262,144-token context limit. Record each engine’s effective cache format, scheduling limits, attention backend, and MTP settings. Record differences that affect comparability, including any unsupported speculative acceptance control.

## Replay

Use `agentx-harness==1.0.14` in a dedicated Python environment. Download `traces.jsonl` from `semianalysisai/cc-traces-weka-062126-256k` at revision `8fecd2fc56694469f758f0afbbb6335ad3043740` and verify its LFS SHA-256 `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`. This is the corpus registered as `semianalysis_cc_traces_weka_062126_256k`. The published 256k variant removes requests above 256,000 combined input and output tokens and preserves surviving requests’ relative timing. [Harness release](https://github.com/SemiAnalysisAI/agentx-harness/releases/tag/agentx-v1.0.14) · [Pinned trace](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/tree/8fecd2fc56694469f758f0afbbb6335ad3043740)

Stage all 393 original JSONL rows as one JSON file per row in `AGENTX_TRACE_DIR`. Preserve the row bytes and use zero-padded filenames to preserve source order. Use the same staged directory for all four arms. The official local and public Weka loaders share the converter that reconstructs deterministic token payloads. [Official Weka replay guide](https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/docs/tutorials/weka-trace.md)

This is a local comparison. The AgentX validator requires `--unsafe-override` for the local `weka_trace` loader and records `submission_valid: false` because it cannot authenticate local corpus identity. [Scenario validation](https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/src/aiperf/common/scenario/validator.py)

Start each arm with a fresh server and empty KV stores. Wait for server and offload readiness. Keep the preset’s seeded 25–75% start range, cache primers, 10 warmup requests per lane, 1,800-second warmup grace, closed-loop timing, and cache busting. Retain warmed cache state during profiling. The 1,800-second submission window begins after the warmup barrier. Successful requests can finish during the default 30-second drain grace, extending the request time span used for throughput. Faster arms can complete different requests within the same window. [AgentX methodology](https://inferencex.semianalysis.com/infx/agentx/methodology) · [Pinned preset](https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/src/aiperf/common/scenario/inferencex_agentx_mvp.py)

Set `AGENTX` to the installed harness executable and `SERVER_URL` to the server’s base URL. Use the engine’s `SERVED_MODEL`, the matrix’s `CONC`, and a new `OUT` directory under `artifacts/` for each arm.

```bash
"$AGENTX" profile --scenario agentx \
    --url "$SERVER_URL" --endpoint /v1/chat/completions \
    --model "$SERVED_MODEL" --tokenizer /data0/shared/Qwen3.8-27B-FP8 \
    --custom-dataset-type weka_trace --input-file "$AGENTX_TRACE_DIR" --unsafe-override \
    --concurrency "$CONC" --random-seed 42 --benchmark-duration 1800 \
    --server-metrics "${SERVER_URL}/metrics" \
    --output-artifact-dir "$OUT" --ui none
```

Keep the full launch commands and environment, source and executable hashes, checkpoint and tokenizer identities, and runtime versions. Retain `profile_export.jsonl`, `profile_export_aiperf.json`, `server_metrics_export.json`, timestamped raw `/metrics` snapshots bracketing profiling, server logs, and offload receipts. Record the phase boundaries and counts of attempted, completed, failed, context-rejected, cancelled, and in-flight requests. Confirm the full profiling window, request coverage, and active offload before interpreting an arm.

## InferenceX metrics

Report all six fields for each arm from its profiling phase. Mark fields without valid evidence as unavailable and record the cause. For interactivity and throughput, apply the InferenceX ETL record filter to `profile_export.jsonl`. It keeps error-free profiling rows and treats a missing phase marker as profiling for older exports. [ETL record filter](https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/infx/results/agentic/artifacts.py)

| Metric | Evidence and conditions |
| --- | --- |
| P90 end-to-end normalized interactivity (tokens/s/user) | For each eligible request with finite positive latency and output length, divide request latency in seconds by output tokens. Take the reciprocal of the P90 of these ratios. Report the eligible sample count. |
| Input token throughput per chip (tokens/s/chip) | Sum actual input tokens from eligible rows and divide by the elapsed seconds from earliest request start to latest request end, then by the physical GPU count. |
| Output token throughput per chip (tokens/s/chip) | Sum actual output tokens from the same rows and divide by the same elapsed time and GPU count. |
| Normalized revenue per chip-hour (USD/chip-hour) | Use the dashboard’s normalized rates of $1 per million uncached input tokens, $0.10 per million cached input tokens, and $1 per million output tokens. Split input throughput using the measured actual hit fraction, price each of the three per-chip rates, and scale to one hour. This is a normalized comparison under a common tariff. |
| Actual KV cache hit rate (%) | Use profiling-window server counter deltas. For vLLM, divide `local_cache_hit` plus `external_kv_transfer` by the sum of all `vllm:prompt_tokens_by_source` labels. If those labels are unavailable, retain the ETL’s separate local and external hit/query ratios and report a combined rate only where their populations and coverage support it. |
| Theoretical KV cache hit rate (%) | Use the profiling aggregate `theoretical_prefix_cache_hit`. Prefer hit blocks (`sum`) divided by total blocks (`count`). Otherwise use `avg`, converting from percent to a fraction if `unit` is `%`. Render the resulting fraction as a percentage. |

For native sglang.cpp, label the actual rate as a native counter calculation. Divide the delta of `vllm:prefix_cache_hits_total` (coherently restored prompt tokens, including LMCache) by the delta of `vllm:prompt_tokens_total`. The stock vLLM ETL cannot recover this rate from the native export because the source breakdown and prefix query count are absent. Restore and prompt counters update at different request stages. Reconcile failed, cancelled, and in-flight requests at the window boundaries, using terminal `prompt_tokens_details.cached_tokens` and `prompt_tokens` where available. Record any per-request fallback and its covered population. Leave the rate unavailable if the evidence cannot establish a comparable numerator and denominator. Keep LMCache component counters as diagnostics because they can count the same restored tokens. [Native metrics](../apps/server/loads.cpp#L264)<!-- prefix: "std::string " --> · [Restore accounting](../src/decoder_batch.cpp#L122)<!-- prefix: "void Decoder" --> · [Prompt accounting](../apps/server/batch_service.cpp#L270)<!-- prefix: "const bool p" -->

[InferenceX request metrics](https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/infx/results/agentic/request_metrics.py) · [vLLM cache metrics](https://github.com/SemiAnalysisAI/InferenceX/blob/d490919c996b5169a3e9345ae3c64e32ccfba357/inferencex-e2e/infx/results/agentic/backends/vllm.py) · [Normalized revenue](https://github.com/SemiAnalysisAI/InferenceX-app/blob/f28b04f384b739d629c1512adee59b9f2878c874/packages/app/src/components/inference/token-revenue.ts) · [Cached price ratio](https://github.com/SemiAnalysisAI/InferenceX-app/blob/f28b04f384b739d629c1512adee59b9f2878c874/packages/app/src/lib/calculator/cache-pricing.ts)

## Relationship to the W1–W5 gates

The W1–W5 gates use fixed cold-prefix cohorts, vLLM KV offload disabled, and a native makespan limit of 1.10× pinned vLLM. AgentX measures cache-enabled, closed-loop replay over a fixed profiling window. The existing acceptance gates retain their own workload and measurement procedure. [W1–W5 benchmark agreement](performance.md#named-workloads)
