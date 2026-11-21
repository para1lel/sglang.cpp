# Paired baseline replay

Run the native arm of the [fixed W1–W6 benchmark](../../docs/performance-comparison.md)
against a release build containing `bin/sglang-server`. On the benchmark host, this
command uses B200 GPU 2 and the published two warmup and five measured cohorts
for each concurrent row. The output directory must be new, with an existing parent.

```bash
bash benchmarks/paired/run_native.sh /path/to/build 2 artifacts/paired-candidate
```

Use the default Pixi environment or provide the native binary's runtime libraries.
The driver needs Bash, `nvidia-smi`, `flock`, `setsid`, `numactl`, `taskset`, `curl`,
`jq`, GNU coreutils and client Python 3. The vLLM directory must exist and allow
opening or creating a shared lock for the physical GPU UUID. Regenerating fixtures
also requires the pinned vLLM environment with `tokenizers`.

| Environment variable | Default |
| --- | --- |
| `PAIRED_HOST_ROOT` | `/data1/shared/dongwu.chen` |
| `PAIRED_CHECKPOINT` | `/data0/shared/Qwen3.8-27B-FP8` |
| `PAIRED_VLLM_ROOT` | `$PAIRED_HOST_ROOT/environments/vllm-81a8cb063e62`, with `venv/bin/python` and GPU locks |
| `PAIRED_CUDA_COMPAT` | `$PAIRED_HOST_ROOT/software/cuda-compat-13.4/usr/local/cuda-13.4/compat` |
| `PAIRED_CLIENT_PYTHON` | `/usr/bin/python3` |
| `PAIRED_SERVER_CPUS` / `PAIRED_CLIENT_CPUS` | `24-33` / `34-35`, both with `numactl --localalloc` |
| `PAIRED_PORT` | `17393` |
| `PAIRED_PROTOCOL_ROOT` | Repository root. Use a frozen export for a paired comparison. |
| `PAIRED_FIXTURES` | Generate W3–W6 requests from the checkpoint tokenizer with pinned vLLM Python. Set a directory to reuse them. |
| `PAIRED_WARMUPS` / `PAIRED_MEASURED` | `2` / `5` for W3–W6 |
| `PAIRED_VLLM_RESULTS` | Unset. Set the saved evidence root to check each requested group's cohort counts. |
| `PAIRED_ALLOW_COUNT_MISMATCH` | `0`. Set to `1` for exploratory runs with differing counts. |

## Replay the retained GPU 2 arm

On the benchmark host, `artifacts/test-bench-baseline` holds the saved vLLM arm
for all six rows. See `REPRODUCE.md` there for the pinned vLLM commands,
`comparison.json` for the paired statistics, and
`vllm-GROUP/GROUP/measurement/summary.json` for each vLLM group's client output.
The retained run used physical UUID `GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c`,
CPUs 24–33 for the server and 34–35 for the client, and local NUMA allocation.
Reuse that UUID, CPU placement and memory policy. Measure a new vLLM arm on the
chosen card for any other placement or the published 2/5 concurrent protocol.

Export the frozen protocol commit into a new directory and check the saved
evidence, checkpoint and protocol bytes before running a candidate. Run these
commands in Bash from the repository root.

```bash
set -eo pipefail
saved=$PWD/artifacts/test-bench-baseline
mkdir -p "$PWD/build"
frozen=$(mktemp -d "$PWD/build/paired-protocol.XXXXXX")
git archive d6e6eb41cb4ac96ed9f992aee908ac2ea850dca4 | tar -x -C "$frozen"
(cd "$saved" && sha256sum --check --quiet evidence.sha256)
(cd "${PAIRED_CHECKPOINT:-/data0/shared/Qwen3.8-27B-FP8}" &&
    sha256sum --check --quiet "$saved/checkpoint.sha256")
awk 'match($0, /\/benchmarks\/(streaming|concurrent)\/[^/]+$/) {
         print $1 "  ." substr($0, RSTART) }' \
    "$saved/vllm-W3/protocol.sha256" |
    (cd "$frozen" && sha256sum --check --quiet)
manifest=benchmarks/concurrent/fixture-manifest.json
fixture_hash=$(jq -er '.fixture_manifest_sha256' \
    "$saved/vllm-W3/W3/measurement/summary.json")
printf '%s  %s\n' "$fixture_hash" "$manifest" |
    (cd "$frozen" && sha256sum --check --quiet)

PAIRED_PROTOCOL_ROOT="$frozen" PAIRED_FIXTURES="$saved/fixtures" \
PAIRED_VLLM_RESULTS="$saved" PAIRED_WARMUPS=1 PAIRED_MEASURED=3 \
    bash benchmarks/paired/run_native.sh /path/to/build 2 artifacts/paired-candidate-gpu2
```

The concurrent observer validates every request body against the frozen fixture
manifest. For W1/W2, verify that `manifest.client_sha256`, `manifest.fixture_sha256`
and `manifest.request_sha256` under the native `summary.json` match the saved
vLLM summary. These identity checks allow the two engines to emit different IDs.
With `PAIRED_VLLM_RESULTS` set, count mismatches fail before startup unless the
exploratory override is set. A paired performance claim requires matching counts.

## Observe and compare

Append any of `W1-W2 W3 W4 W5 W6` to select groups. Each group starts a fresh
server with 32 active slots, a 32,768-token scheduling cap, tensor-core FP8 prefill,
batch CUDA graphs, and MTP and prefix reuse disabled. W1 and W2 use three warmups
and nine measured requests each. W3–W6 use a 90 s socket timeout, a 15 s barrier
timeout and a 360 s cohort deadline. The submission span must be at most 50 ms,
and every lane must produce output before any stream ends.

The driver saves the client command with `PYTHONPATH`, GPU samples from server
startup, and raw streams once measurement begins. If IDs differ from a lane's
first warmup, the observer retains the offending cohort and records the first
differing token under `repeat_id_mismatch`. Compare separately captured diagnostic
IDs with the first warmup before using them as repeatability evidence. The driver
continues to the next group and exits nonzero if any group fails startup,
measurement or validation.

Compare median output IDs/s and pooled request p95 latency. In the raw W1/W2
summary, use `workloads.W1` or `workloads.W2`, then `output_tokens_per_second.p50` and
`ttft_ms.p95`, `tpot_ms.p95` and `e2e_ms.p95`. In W3–W6, use
`throughput.output_tokens_per_second.p50` and the same latency keys under `latency_ms`.
The saved `comparison.json` exposes `rows.W1.vllm` through `rows.W6.vllm` with
`output_ids_per_second_p50` and `latency_p95_ms`. Each row's `fresh_vllm_thresholds`
uses the saved same-card vLLM variability. W3–W6 `wire_and_timing_gates` contains
archived speed checks after successful wire validation. Those speed misses do
not fail the driver. The native W1/W2 client leaves threshold generation to the
vLLM arm. Use the fresh same-card thresholds for a paired claim.
