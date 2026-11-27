# Paired serving benchmarks

Use the [matched K4 driver](#matched-k4-timing-protocol) for the current W1–W5 suite.
For a standalone depth-zero control, run pinned vLLM first, then native on the
same reserved B200. Both launchers use an 8,192-token budget, 32 in-service
requests, one warmup, and one measured cohort per workload. Match the GPU and
CPU placement. Each output directory must be new.

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
| `PAIRED_FIXTURES` | Generate W3–W6 requests from the checkpoint tokenizer with pinned vLLM Python. Set a directory to reuse them. |
| `PAIRED_WARMUPS` / `PAIRED_MEASURED` | `1` / `1` for new depth-zero captures. |
| `PAIRED_MAX_NUM_BATCHED_TOKENS` | `8192`. Any other explicit cap is exploratory or historical. The native range is 1–32768. With 32 sequences vLLM requires 32–32768. |
| `PAIRED_VLLM_SEED_CACHE` | `$PAIRED_HOST_ROOT/cache/vllm-81a8cb063e62`, copied into each vLLM group's private cache. |
| `PAIRED_VLLM_HTTP_PROXY` | Unset. Set to `http://127.0.0.1:7890` for comparator cubin downloads in matched K4 TP1 and TP2 runs. Local requests bypass the proxy. |
| `PAIRED_VLLM_RESULTS` | Unset. Set the saved vLLM evidence root to request a verified same-device comparison. |
| `PAIRED_COMPARISON_MODE` | `matched` when saved results are supplied. Use `frozen` with W3–W6 and no saved results for the published GPU2 reference. |
| `PAIRED_CHECKPOINT_MANIFEST` | The saved `checkpoint.sha256` or `meta/checkpoint.sha256` in matched mode. Supply the published manifest explicitly in frozen mode. |
| `PAIRED_ALLOW_COUNT_MISMATCH` | `0`. Set to `1` to label an otherwise verified comparison exploratory when cohort counts differ. |

## Matched K4 timing protocol

Start a fresh campaign directory and run one row at a time on the same reserved
B200. Use a Release binary with a verified source revision and binary digest,
the published checkpoint manifest, and the pinned vLLM environment and
FlashInfer cubin cache.

```bash
export PAIRED_CHECKPOINT_MANIFEST=/path/to/published/checkpoint.sha256
export PAIRED_NATIVE_SOURCE='verified-40-character-source-revision'
export PAIRED_NATIVE_SHA256='verified-64-character-binary-digest'
bash benchmarks/paired/run_k4.sh /path/to/sglang-server 2 artifacts/paired-k4-110 W4
```

W4 can start a fresh campaign. Run W5 after W4 passes in that campaign.
The short rows follow W1, W2, then W3, advancing after each pass. A row starts
fresh servers for both timed arms. The vLLM arm goes first on W1, W3 and W5. Both K4 arms use
the same checkpoint files, physical GPU, CPU placement, request bytes, cohort
counts, and client clock. The clients drain each SSE response after measuring
from the earliest submit to the last valid `[DONE]`. The runner uses port 17424
on loopback by default and holds the campaign, port, and both GPU coordination
locks. `PAIRED_VLLM_SEED_CACHE` can point to the retained compiler cache seed.

The default profile `server-defaults` records `matched-k4-native-defaults-cache-on-v5`.
Campaign and comparison receipts record `sampling: single-round-v1` for new captures.
Choose the participating B200 by `GPU_INDEX`. Both engine processes use only
that device. The row width is the requests per participating GPU multiplied by
the number of participating GPUs. The current launcher supports one GPU.

For two devices, use [`benchmarks/tp2/run.sh`](../../docs/serving/tp2.md#acceptance-target).
The `tp2-paired-v2` driver uses a fresh campaign root and one private LMCache
0.5.8 process, socket, loopback port, and `sglang` namespace. Its 128 GiB CPU L1
is shared by both GPU UUIDs. An acknowledged reset covers both ranks. Verify
rank-local receipts and match the active cache gate on each rank. The separate
TP1 driver retains a private 64 GiB daemon. Set `PAIRED_LMCACHE_PORT` to
choose the TP2 daemon’s loopback port (default 17655).

The campaign records `daemon_port` and `lmcache_capacity_bytes`. Native
readiness and completion receipts report one `daemon_pid` and two ranks.
The `daemon/` receipts bind both UUIDs, the socket peer, and the process
configuration. Replay `tp2-paired-v1` at its pinned revision with the original
evidence. [Campaign identity](../../benchmarks/tp2/run.sh#L75)<!-- prefix: "'{protocol:\"" --> · [Daemon receipts](../../benchmarks/tp2/native.py#L88)<!-- prefix: "def verify_d" --> · [Historical two-daemon protocol](https://github.com/para1lel/sglang.cpp/blob/57ac5fbbdf20a5a80c5bc079b49073a44c06680f/benchmarks/tp2/run.sh#L73)

See the [named workloads](../../docs/performance.md#named-workloads) for row sizes.

W4 selects lanes 00–07 from the immutable archived W5 corpus. W5 selects all
four archived W6 lanes. `PAIRED_FIXTURES` takes the original corpus directory,
including those archived directory names. Request bytes are checked by digest.
Each row runs one warmup and one measured cohort per engine. Only the measured
cohort supplies the comparison, so each reported makespan is a point estimate.

Both arms use MTP4 and a 32,768-token budget. The native arm launches with only
`--port` and verifies the resolved budget and active-request cap of eight times
the participating GPU count. vLLM receives that cap through `--max-num-seqs` and
uses async scheduling with both token caps explicitly set to 32,768. The driver
checks the resolved scheduler and CUDA graph settings against the API and worker
receipts and checks the launch environment. Both engines enable prefix caching.
Drain outstanding work before each cohort. Require an acknowledged
`POST /reset_prefix_cache` with an empty JSON object before timing. Each request
must report zero cached tokens, and the prefix hit counter must stay unchanged.
Replay checks the reset response, metrics, and control timestamps.
All request bodies
must be sent before the first valid completion, within the frozen submit-span
bound. The comparison records whether every lane started generating before the
first completion and reports per-request latencies and throughput from SSE
replay.
The native timing arm uses Release defaults with tracing disabled. A separate
untimed W1 native trace checks active depth. The driver also checks effective
MTP4 and records the vLLM verification counters. Both timing arms must pass
wire, length, usage, and drain checks. Compare each engine's measured IDs with
its own warmup.

Each current row passes when native makespan is at most 1.10 times pinned vLLM
makespan. `Wn/comparison.json` retains `native-median-at-most-110-percent` and the
legacy median field names. Each median holds the single measured value. The
comparison also records the ratio and `strict_passed`. Advancement checks prior
rows under the same campaign identity. W5 requires a passing v5 W4 with the
32,768-token budget and eight 32,768-input/4,096-output lanes on one GPU. A timing
miss returns status 1 and keeps the captures and comparison. Invalid evidence
stops the row before comparison.

Replay a saved row from the repository root with `PYTHONPATH=.` and
`python benchmarks/paired/k4.py assess Wn FIXTURES ROW_DIR`. Replay checks native
process arguments, vLLM activity counters, and client exit status. The saved
identity selects the profile and cohort schedule. Captures without the sampling
field retain their original sample counts, controls, and verdicts.

### Driver health

On TP2, each arm's server log is scanned for authentic CUDA launch failures at entry and after each cohort's capture is written. Paired assessment and legacy replay re-check the same log. Any invalid-handle, launch-failure, launch-timeout, misaligned- or illegal-address, illegal-instruction, hardware-stack-error, or assert marker rejects the whole arm. A failed-launch record also rejects: a `launch failed: CUresult` line with a non-success code counts as a failure whatever its error class. A rejected arm certifies no time, even when every request returned HTTP 200 with `[DONE]`. The rejection is run-level: merged stdout and stderr do not attribute a lane or rank. Benign capacity probes — a bare invalid-value or out-of-resources mention outside a failed-launch record — never reject, and a missing server log rejects. Warmup and measured cohorts are treated alike. A clean log proves only the absence of these markers, not positive dispatch or numerical accuracy. Output token contents remain reference only. [Serious markers](driver_health.py#L6)<!-- prefix: "\"CUDA_ERROR_" --> · [Failed-launch record](driver_health.py#L25)<!-- prefix: "_FAILED_LAUN" --> · [Health gate](driver_health.py#L45)<!-- prefix: "def require_" -->

### Historical K4 profiles

Use the archived recipes with their matching Release. They cover exact FP8,
ordered Tensor Core and INT8 draft, bounded and long split attention, and the
older cache and cohort policies. Replay saved receipts with `k4.py assess`.
[Archived K4 profiles](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/benchmarks/paired/README.md#historical-k4-profiles)

### Prefix-reuse diagnostic

Run this separately against each verified cache-enabled K4 endpoint on the same
B200 and NUMA placement. Use the frozen W3 fixtures and a fresh evidence
directory for each endpoint. Start a dedicated native server with
`--port 17424 --trace-batches` and capture its standard error in `NATIVE_LOG`.

```sh
PYTHONPATH="$PWD" python3 -B benchmarks/paired/prefix_reuse.py native http://127.0.0.1:17424 FIXTURES NATIVE_EVIDENCE --native-trace NATIVE_LOG
PYTHONPATH="$PWD" python3 -B benchmarks/paired/prefix_reuse.py vllm   http://127.0.0.1:17424 FIXTURES VLLM_EVIDENCE
```

The diagnostic resets and captures cold W3. It resets again and sends an
8,193-token prime. The four 9,984-token branch prompts share its first 8,192
IDs and each diverges at the next token. Each branch returns 1,024 output IDs.
The diagnostic saves SSE, per-lane cached token counts, prefix-counter deltas,
and warm-only depth-four verification evidence from each engine. It requires
exactly 8,192 cached prompt tokens on each native lane and a positive native
draft restore count. Pinned vLLM must report positive reuse on every lane, up
to the common prefix. Its block geometry determines how much it can reuse.
The last reset must make the identical prime cold again. Use the diagnostic
as untimed support evidence for `matched-k4-prefix-reuse-v2`.

The historical [W3 prefix and K1 recipe](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/benchmarks/variants/README.md)
replays the archived 6,400-token prefix and one-draft-position captures. Use that
checkout with its saved evidence. The
[frozen comparison](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/benchmarks/variants/comparison.json#L13)
records both rows and their original gates.

## Replay the historical GPU 2 arm

Use the saved vLLM arm with its original GPU UUID, CPU placement, and NUMA
policy. The recipe includes the evidence locations and receipt checks.
[Archived GPU 2 replay](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/benchmarks/paired/README.md#replay-the-historical-gpu-2-arm)

## Compare against the published frozen reference

The archived recipe compares with the published 8,192-token W3–W6 GPU 2
reference. The receipt identifies cross-cap and cross-device comparisons and
sets `makespan.passed` to `null`.
[Frozen reference recipe](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/benchmarks/paired/README.md#compare-against-the-published-frozen-reference)

## Observe and compare

Append any of `W1-W2 W3 W4 W5 W6` to select groups. Each group starts a fresh
server with 32 active slots and an 8,192-token scheduling cap. The native arm uses
the Release's automatic operator policy and enables batch CUDA graphs.
Both arms disable MTP and prefix reuse.
Use one warmup and one measured cohort for each workload. The W1/W2 client
times the first `[DONE]` and drains the HTTP response. A second terminal,
later data event, or short body invalidates the sample. Fresh W1/W2 captures
include HTTP response end receipts. W3–W6 use a 90 s socket timeout, a 15 s
barrier timeout, and a 360 s cohort deadline. For paired captures, all lanes connect
before the shared barrier. Their submit timestamps must span at most 50 ms and
every request body must be sent before the first valid `[DONE]`.

The driver saves the client command with `PYTHONPATH`, GPU samples from server
startup, and raw streams once measurement begins. If IDs differ from a lane's
first warmup, the observer retains the offending cohort and records the first
differing token under `repeat_id_mismatch`. Compare separately captured diagnostic
IDs with the first warmup before using them as repeatability evidence.

The driver binds the comparison plan before launch. It replays both arms and
checks the saved device, placement, source, settings, checkpoint, request bytes,
and cohort schedule. Native K0 records the `automatic-k0-v1` policy and
`/server_info` in `native-k0-policy.json`. Use separate profiling for kernel
dispatch. Verified comparisons at the protocol cap and cohort counts use
`same-device matched-cap`. A non-protocol cap or cohort schedule is exploratory.

For a matched K0 comparison, require native measured makespan to be strictly
below pinned vLLM measured makespan. The cohort starts at its earliest client
submit and ends at its last valid `[DONE]`. `GROUP/comparison.json` records the
samples and verdict under `makespan.workloads`. Its median fields hold the single
measured value for a new capture. W1 and W2 have independent verdicts in the
same group. On a makespan miss, including a tie, the native driver
keeps the comparison receipt. Any failed startup, measurement, validation, or
makespan gate stops the native driver before the next group and exits nonzero.

Use output IDs/s, TTFT, TPOT, end-to-end latency, p95, and the intervals when all
streams are open or generating as diagnostics. A lane can start producing
output after another lane completes. The W1/W2 summary reports rates and latency
under `workloads.W1` and `workloads.W2`. W3–W6 reports them under `throughput`
and `latency_ms`. `wire_and_timing_gates` and the retained archive's
`fresh_vllm_thresholds` keep the historical speed checks available for diagnosis.
