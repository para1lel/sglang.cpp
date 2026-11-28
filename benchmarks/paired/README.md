# Paired serving benchmarks

Use the [matched K4 driver](#matched-k4-timing-protocol) for the current W1–W6 suite. For the [K0 control](#observe-and-compare), run pinned vLLM, then native on the same reserved B200. Match CPU placement and use new output directories.

```bash
export PAIRED_CHECKPOINT_MANIFEST="$PWD/artifacts/test-bench-baseline/checkpoint.sha256"
bash benchmarks/paired/run_vllm.sh 2 artifacts/paired-vllm-8192
PAIRED_VLLM_RESULTS="$PWD/artifacts/paired-vllm-8192" \
    bash benchmarks/paired/run_native.sh /path/to/build 2 artifacts/paired-native-8192
```

Use the default Pixi environment or provide the native binary's runtime libraries.
The driver needs Bash, `nvidia-smi`, `flock`, `setsid`, `numactl`, `taskset`, `curl`,
`jq`, GNU coreutils and client Python 3.11 or newer for identity checks. The vLLM
directory needs its pinned checkout, installed environment, wheel and native hash
receipts, and a writable shared lock for the physical GPU UUID. The vLLM arm
also uses the retained compiler cache seed. Set `PAIRED_FIXTURES` to share frozen
requests or let each arm regenerate them with the pinned tokenizer. The comparison
checks the fixture and request bytes. Both arms record resolved settings before timing.

| Environment variable | Default |
| --- | --- |
| `PAIRED_HOST_ROOT` | `/data1/shared/dongwu.chen` |
| `PAIRED_CHECKPOINT` | `/data0/shared/Qwen3.8-27B-FP8` |
| `PAIRED_VLLM_ROOT` | `$PAIRED_HOST_ROOT/environments/vllm-81a8cb063e62`, with `venv/bin/python` and GPU locks |
| `PAIRED_CUDA_COMPAT` | `$PAIRED_HOST_ROOT/software/cuda-compat-13.4/usr/local/cuda-13.4/compat` |
| `PAIRED_CLIENT_PYTHON` | `/usr/bin/python3` |
| `PAIRED_SERVER_CPUS` / `PAIRED_CLIENT_CPUS` | `24-33` / `34-35`, both with `numactl --localalloc`. Matched K4 chooses defaults by GPU index. Set both to use coordinated, disjoint physical cores on the GPU's NUMA node. |
| `PAIRED_PORT` | `17393` |
| `PAIRED_PROTOCOL_ROOT` | Repository root. Use a frozen export for a paired comparison. |
| `PAIRED_FIXTURES` | Directory with the matching cohort fixtures. Older drivers can generate archived W3–W6 fixtures with pinned vLLM Python. Prepare the current W6 serial fixture separately in this directory. |
| `PAIRED_WARMUPS` / `PAIRED_MEASURED` | `1` / `1` for new depth-zero captures. |
| `PAIRED_MAX_NUM_BATCHED_TOKENS` | `8192`. Any other explicit cap is exploratory or historical. The native range is 1–32768. With 32 sequences vLLM requires 32–32768. |
| `PAIRED_VLLM_SEED_CACHE` | `$PAIRED_HOST_ROOT/cache/vllm-81a8cb063e62`, copied into each vLLM group's private cache. |
| `PAIRED_VLLM_HTTP_PROXY` | Unset. Set to `http://127.0.0.1:7890` for comparator cubin downloads in matched K4 TP1 and TP2 runs. Local requests bypass the proxy. |
| `PAIRED_VLLM_RESULTS` | Unset. Set the saved vLLM evidence root to request a verified same-device comparison. |
| `PAIRED_COMPARISON_MODE` | `matched` when saved results are supplied. For the archived GPU2 reference, use `frozen` with its W3–W6 fixtures and omit saved results. |
| `PAIRED_CHECKPOINT_MANIFEST` | The saved `checkpoint.sha256` or `meta/checkpoint.sha256` in matched mode. Supply the published manifest explicitly in frozen mode. |
| `PAIRED_ALLOW_COUNT_MISMATCH` | `0`. Set to `1` to label an otherwise verified comparison exploratory when cohort counts differ. |

## Matched K4 timing protocol

Use a fresh campaign directory, a verified Release binary, the published checkpoint manifest, and the pinned vLLM environment and FlashInfer cubin cache.

```bash
export PAIRED_CHECKPOINT_MANIFEST=/path/to/published/checkpoint.sha256
export PAIRED_NATIVE_SOURCE='verified-40-character-source-revision'
export PAIRED_NATIVE_SHA256='verified-64-character-binary-digest'
bash benchmarks/paired/run_k4.sh /path/to/sglang-server 2 artifacts/paired-k4-110 W4
```

W4 can start a fresh campaign. W5 requires W4 to pass. W6 requires W5 to pass at the same width and under the same campaign identity. Start fresh servers for each row. Run vLLM first on W1, W3, and W5. Follow the [campaign order, matching controls and SSE timing](../../docs/performance.md#measurement). The TP1 driver defaults to loopback port 17424 and holds the campaign, port, and GPU coordination locks.

The TP1 launcher selects one physical B200 with `GPU_INDEX`. The default `PAIRED_K4_NATIVE_PROFILE=server-defaults` uses a private 64 GiB LMCache 0.5.8 daemon. The campaign records `matched-k4-native-defaults-lmcache-hbm-pool` and `sampling: single-round-v1`. Native uses its [default GPU pools](../../docs/serving/request-memory.md). Pinned vLLM uses local prefix caching with KV offload disabled. Set `PAIRED_LMCACHE_PYTHON` and `PAIRED_LMCACHE_WHEEL` to the interpreter and official wheel. `PAIRED_LMCACHE_PYTHONPATH` is optional. `PAIRED_LMCACHE_PORT` selects a separate daemon port (17524 for TP1). [LMCache inputs](run_k4.sh#L114-L122)<!-- prefix: "lmcache_pyth" --> · [Protocol identity](run_k4.sh#L220-L226)<!-- prefix: "| if $lmcach" -->

For TP2, use [`benchmarks/tp2/run.sh` and its acceptance checks](../../docs/serving/tp2.md#acceptance-target). Use a fresh `tp2-paired-v5` root. Set `PAIRED_LMCACHE_PORT` to override the daemon port (17655). The campaign records one 128 GiB daemon and the SLC3 `0.5.8+native-v5` handshake. Readiness and completion bind one daemon PID to both ranks. The `daemon/` receipts bind both UUIDs, the socket peer, and process configuration. Replay archived campaigns at their original revisions. [Campaign identity](../../benchmarks/tp2/run.sh#L83)<!-- prefix: "'{protocol:\"" --> · [Daemon receipts](../../benchmarks/tp2/native.py#L88)<!-- prefix: "def verify_d" --> · [Historical two-daemon protocol](https://github.com/para1lel/sglang.cpp/blob/57ac5fbbdf20a5a80c5bc079b49073a44c06680f/benchmarks/tp2/run.sh#L73)

Use the [named workloads](../../docs/performance.md#named-workloads). For TP1 W4/W5, point `PAIRED_FIXTURES` at the original corpus with its archived directory names. The driver checks request digests.

For W6, [prepare and verify the serial fixture](../../docs/performance.md#w6-serial-history) under `$PAIRED_FIXTURES` before starting the campaign. After W5 passes, run `run_k4.sh` with row `W6` at TP1 or `run.sh measure` with row `W6` at TP2. Keep the raw `warmup-01.json.gz` and `measured-01.json.gz` for each engine. The measured clock spans the first POST to the final HTTP EOF. W1–W5 score the earliest submission through the last valid `[DONE]`.

The TP1 native timing arm supplies only `--port` and disables tracing. A separate untimed native W1 trace checks active draft depth. The driver checks launch settings against API and worker receipts and records vLLM verification counters. Follow the [scheduler controls](../../docs/performance.md#scheduler-and-reporting-policy) and [wire, ID and cold-cache gates](../../docs/performance.md#measurement). Require an acknowledged `POST /reset_prefix_cache` with `{}` after drain before each cohort. Replay checks the reset response, metrics and control timestamps.

`Wn/comparison.json` records `native-median-at-most-110-percent`, the ratio, and the gate verdict in `passed`. The workload's `strict_passed` records whether native is strictly faster. Legacy median fields hold one measured makespan per arm in a new capture. Advancement checks prior receipts under the same campaign identity, including the expected workload and token budget. A timing miss returns status 1 and keeps captures and the comparison. Invalid evidence stops the row before comparison.

Replay a saved row from the repository root with `PYTHONPATH=.` and
`python benchmarks/paired/k4.py assess Wn FIXTURES ROW_DIR`. Replay checks native
process arguments, vLLM activity counters, and client exit status. The saved
identity selects the profile and cohort schedule. Captures without the sampling
field retain their original sample counts, controls, and verdicts.

Current receipts check the drained, empty GPU pools after each reset, positive pool capacities, and settled backup counters. W1–W5 require zero reuse. W6 can reuse GPU and CPU state inside its measured chain. Historical `drain-suffix` and `tp2-paired-v4` receipts retain their original per-checkpoint acknowledgement checks. [Cold cache controls](../../docs/performance.md#measurement)

### Driver health

On TP2, each arm's server log is scanned for authentic CUDA launch failures at entry and after each cohort's capture is written. Paired assessment and legacy replay re-check the same log. Any invalid-handle, launch-failure, launch-timeout, misaligned- or illegal-address, illegal-instruction, hardware-stack-error, or assert marker rejects the whole arm. A failed-launch record also rejects: a `launch failed: CUresult` line with a non-success code counts as a failure whatever its error class. A rejected arm certifies no time, even when every request returned HTTP 200 with `[DONE]`. The rejection is run-level: merged stdout and stderr do not attribute a lane or rank. Benign capacity probes — a bare invalid-value or out-of-resources mention outside a failed-launch record — never reject, and a missing server log rejects. Warmup and measured cohorts are treated alike. A clean log proves only the absence of these markers, not positive dispatch or numerical accuracy. Output token contents remain reference only. [Serious markers](driver_health.py#L6)<!-- prefix: "\"CUDA_ERROR_" --> · [Failed-launch record](driver_health.py#L25)<!-- prefix: "_FAILED_LAUN" --> · [Health gate](driver_health.py#L45)<!-- prefix: "def require_" -->

### Historical K4 profiles

Use the archived recipes with their matching Release and saved evidence. They cover earlier projection precision, draft, split attention, cache and cohort policies. The historical profile `server-defaults-cache-on-v5` records `matched-k4-native-defaults-cache-on-v5`. [Historical profile flags](run_k4.sh#L15-L19)<!-- prefix: "echo \"Histor" --> · [Archived K4 profiles](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/benchmarks/paired/README.md#historical-k4-profiles)

### Prefix-reuse diagnostic

Run this separately against each verified cache-enabled K4 endpoint on the same
B200 and NUMA placement. Use the frozen W3 fixtures and a fresh evidence
directory for each endpoint. Start a dedicated native server with
`--port 17424 --trace-batches --lmcache-socket SOCKET`. Use the [native daemon setup](../../docs/serving/lmcache.md#start-the-daemon). Capture server standard error in `NATIVE_LOG`.

```sh
PYTHONPATH="$PWD" python3 -B benchmarks/paired/prefix_reuse.py native http://127.0.0.1:17424 FIXTURES NATIVE_EVIDENCE --native-trace NATIVE_LOG
PYTHONPATH="$PWD" python3 -B benchmarks/paired/prefix_reuse.py vllm   http://127.0.0.1:17424 FIXTURES VLLM_EVIDENCE
```

The diagnostic resets and captures cold W3. It resets again and sends an
8,192-token prime. The four 9,984-token branch prompts share those 8,192
IDs and each diverges at the next token. Each branch returns 1,024 output IDs.
The diagnostic saves SSE, per-lane cached token counts, prefix-counter deltas,
and warm-only depth-four verification evidence from each engine. It requires
exactly 8,192 cached prompt tokens on each native lane and a positive native
draft restore count. Pinned vLLM must report positive reuse on every lane, up
to the common prefix. Its block geometry determines how much it can reuse.
The last reset must make the identical prime cold again and preserve its output.
The native cold captures include suffix-checkpoint drain receipts. Use the diagnostic
as untimed support evidence for `matched-k4-prefix-reuse-v3`.

The historical [W3 prefix and K1 recipe](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/benchmarks/variants/README.md)
replays the archived 6,400-token prefix and one-draft-position captures. Use that
checkout with its saved evidence. The
[frozen comparison](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/benchmarks/variants/comparison.json#L13)
records both rows and their original gates.

## Replay the historical GPU 2 arm

For the saved 32,768-token vLLM arm, reuse its original GPU UUID, CPU placement and NUMA policy. Use the archived checkout and saved evidence. [GPU 2 replay and receipt checks](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/benchmarks/paired/README.md#replay-the-historical-gpu-2-arm)

## Compare against the published frozen reference

For the separate 8,192-token W3–W6 GPU 2 reference, use `PAIRED_COMPARISON_MODE=frozen` and the published manifest. The receipt identifies cross-cap and cross-device comparisons. Its `makespan.passed` is `null`. [Frozen reference invocation and checks](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/benchmarks/paired/README.md#compare-against-the-published-frozen-reference)

## Observe and compare

Append any of `W1-W2 W3 W4 W5 W6` to select groups. Each group starts a fresh server with 32 active slots. The protocol uses an 8,192-token cap and one warmup plus one measured cohort per workload. Both arms disable MTP and prefix reuse. Native enables batch CUDA graphs under Release automatic operator policy.

W1/W2 measure the first valid `[DONE]` and drain the response. A second terminal, later data, or short body invalidates the sample. Fresh captures include HTTP response end receipts. W3–W6 use a 90 s socket timeout, 15 s barrier timeout and 360 s cohort deadline. Preconnect all lanes before the shared barrier. Submit within 50 ms and send all bodies before the first valid completion.

The driver binds the comparison plan before launch, saves the client command with `PYTHONPATH` and startup GPU samples, and retains raw streams once measurement begins. Each engine must repeat its own greedy IDs from the first warmup. On a mismatch, `repeat_id_mismatch` records the first differing token and the driver keeps the cohort. Match separate diagnostic IDs against that warmup before using them as repeatability evidence.

Replay checks device, placement, source, settings, checkpoint, request bytes and cohort schedule. Native records `automatic-k0-v1` and `/server_info` in `native-k0-policy.json`. Profile kernel dispatch separately. Protocol caps and cohort counts yield `same-device matched-cap`. Overrides are exploratory.

For each matched K0 workload, native makespan must be strictly below pinned vLLM, from earliest submit to last valid `[DONE]`. `GROUP/comparison.json` records samples and verdicts under `makespan.workloads`. New captures put the single measured value in the median fields. W1 and W2 have independent verdicts in the same group. Startup, measurement, evidence or makespan failures stop the native driver before the next group with a nonzero exit. A makespan miss, including a tie, preserves the comparison receipt.

Use [stream diagnostics](../../docs/performance.md#measurement) and retain p95 latency and the intervals when all streams are open or generating. A lane can first generate after another completes. Find W1/W2 rates and latencies under `workloads.W1` and `workloads.W2`, and W3–W6 under `throughput` and `latency_ms`. Historical speed checks remain in `wire_and_timing_gates` and the archive's `fresh_vllm_thresholds`.
