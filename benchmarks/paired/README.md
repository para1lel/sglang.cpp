# Paired W1–W6 benchmark

Use the [matched K4 driver](#matched-k4-timing-protocol) to compare the native
Release defaults against pinned vLLM MTP4. For a depth-zero control, run pinned
vLLM first, then run the native release build on the same reserved B200.
Both commands default to 8,192 scheduled tokens and 32 in-service requests, with
two warmups and five measured cohorts for each concurrent row. Select the same GPU
and CPU placement for both arms. Each output directory must be new.

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
| `PAIRED_SERVER_CPUS` / `PAIRED_CLIENT_CPUS` | `24-33` / `34-35`, both with `numactl --localalloc` |
| `PAIRED_PORT` | `17393` |
| `PAIRED_PROTOCOL_ROOT` | Repository root. Use a frozen export for a paired comparison. |
| `PAIRED_FIXTURES` | Generate W3–W6 requests from the checkpoint tokenizer with pinned vLLM Python. Set a directory to reuse them. |
| `PAIRED_WARMUPS` / `PAIRED_MEASURED` | `2` / `5` for W3–W6 |
| `PAIRED_MAX_NUM_BATCHED_TOKENS` | `8192`. Any other explicit cap is exploratory or historical. The native range is 1–32768. With 32 sequences vLLM requires 32–32768. |
| `PAIRED_VLLM_SEED_CACHE` | `$PAIRED_HOST_ROOT/cache/vllm-81a8cb063e62`, copied into each vLLM group's private cache. |
| `PAIRED_VLLM_RESULTS` | Unset. Set the saved vLLM evidence root to request a verified same-device comparison. |
| `PAIRED_COMPARISON_MODE` | `matched` when saved results are supplied. Use `frozen` with W3–W6 and no saved results for the published GPU2 reference. |
| `PAIRED_CHECKPOINT_MANIFEST` | The saved `checkpoint.sha256` or `meta/checkpoint.sha256` in matched mode. Supply the published manifest explicitly in frozen mode. |
| `PAIRED_ALLOW_COUNT_MISMATCH` | `0`. Set to `1` to label an otherwise verified comparison exploratory when cohort counts differ. |

## Matched K4 timing protocol

Run the protocol one row at a time on the same reserved B200. Use a Release
binary with a verified source revision and binary digest, the published checkpoint
manifest, and the pinned vLLM environment and FlashInfer cubin cache.

```bash
export PAIRED_CHECKPOINT_MANIFEST=/path/to/published/checkpoint.sha256
export PAIRED_NATIVE_SOURCE='verified-40-character-source-revision'
export PAIRED_NATIVE_SHA256='verified-64-character-binary-digest'
bash benchmarks/paired/run_k4.sh /path/to/sglang-server 2 artifacts/paired-k4 W1
```

Supply W2 through W6 in order after each prior row passes. A row starts fresh servers
for both timed arms. The vLLM arm goes first on W1, W3 and W5. Both K4 arms use
the same checkpoint files, physical GPU, CPU placement, request bytes, cohort
counts, and client clock. The clients drain each SSE response after measuring
from the earliest submit to the last valid `[DONE]`. The runner uses port 17424
on loopback by default and holds the campaign, port, and both GPU coordination
locks. `PAIRED_VLLM_SEED_CACHE` can point to the retained cache seed used for v4.

The launcher selects `server-defaults` when `PAIRED_K4_NATIVE_PROFILE` is unset.
A separate W1 native trace checks depth-four verification. The timed native arm
uses the Release defaults with tracing disabled. The vLLM arm uses explicit MTP
depth four on the pinned revision, with its resolved worker configuration and
fresh verification counters recorded. Its streams must pass the full wire,
length, usage, and drain checks. vLLM output IDs may vary across valid streams
and may differ from native and archived IDs. The runner launches only K4 on vLLM.

`Wn/comparison.json` records the strict median makespan gate. A slower or tied
native median records a timing miss and returns status 1. Invalid evidence stops
the row before comparison. The campaign retains its raw captures and exits at
the first failure. Saved historical rows retain the acceptance criteria for their
recorded profiles.

Replay a saved row with `python benchmarks/paired/k4.py assess Wn FIXTURES ROW_DIR`
from the repository root with `PYTHONPATH=.`. New rows require saved native process
arguments, the W1 native trace, raw vLLM activity counters, and successful client
statuses. Replay verifies these against the launch, server settings, and activity
summaries. Historical version 2 exact and opt-in rows also capture the native
split dispatch setting at readiness and after measurement. Their native cohorts
bind the digests of the process arguments and readiness dispatch receipts.
The native trace must contain a complete depth-four event. Verification headers
and their lane and row counts must agree with the trace events.
Replay also checks any saved comparison and final row exit status. Older rows remain
replayable and any process or activity receipts they contain are checked.

### Historical K4 profiles

Use the measured Release that accepted `--fp8-prefill` to reproduce these
profiles. `PAIRED_K4_NATIVE_PROFILE=exact` selects the historical native exact-FP8
profile, which requires every native K4 output to match the same lane in a fresh
native exact-FP8 K0 control.

#### Ordered Tensor Core with INT8 draft

Use a fresh campaign directory and set `PAIRED_K4_NATIVE_PROFILE=ordered-tc-int8`
with a verified Release server that supports `--mtp-int8-head`. The native K4
arm selects `--fp8-prefill tensor-core-mtp-target --mtp-steps 4 --mtp-int8-head`.
The native K0 control uses the same FP8 policy with depth zero. The runner
requires all native K4 IDs to match this control on every lane and capture.
vLLM still uses the pinned standard K4 worker and has the same stream contract.

A separate fresh exact-FP8 K0 control writes `exact-k0-compatibility.json` before
timing. It records per-lane mismatch counts and the first differing IDs.
The comparison also includes this diagnostic. The opt-in timing result applies
only to the named policy. The historical exact-FP8 comparison retains its original
adoption gate. Each opt-in cohort carries its campaign and arm identity. Use the
same profile and directory for each subsequent row.

#### Ordered Tensor Core with split target attention and INT8 draft

Set `PAIRED_K4_NATIVE_PROFILE=ordered-tc-int8-split-gqa` in a fresh campaign
directory and use a Release server that implements split target attention.
Use this same profile for W1 through W6. The launcher sets
`SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA=1` for both native policy arms and the W1
native trace. It requires
`experimental_target_split_gqa: true` in their live server information.
The exact K0 diagnostic runs with split disabled. Caller-supplied dispatch
overrides are rejected.

The split policy changes target attention arithmetic at eligible visible
lengths. Native K4 must match split K0 on every lane. The separate exact K0
diagnostic measures compatibility, and the pinned vLLM K4 arm measures timing.
Neither establishes equivalence to the published ordered profile. The
campaign binds the split policy and the fixed native settings to all receipts,
including rows where attention falls back to the ordered implementation.

#### Ordered Tensor Core with long split target attention and INT8 draft

Set `PAIRED_K4_NATIVE_PROFILE=ordered-tc-int8-long-split-gqa` in a fresh campaign
directory for W1 through W6. Use a Release server with the long split path.
The native K4 timing and W1 trace arms select `--mtp-steps 4`,
`--fp8-prefill tensor-core-mtp-target` and `--mtp-int8-head`. Both K4 arms
and the depth-zero native policy control use
`SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA=long`. The effective server information
must report `experimental_long_target_split_gqa: true`, with bounded split
disabled. The separately reported exact K0 diagnostic disables both split modes.

Long split extends eligible target attention through the model's supported
text context. All native K4 captures must match their own long-split K0
control. The pinned vLLM K4 arm supplies the paired timing observations.
Campaign metadata and version 2 evidence distinguish long and bounded split.

### Native server defaults against matched vLLM K4

Use a fresh campaign directory. `server-defaults` is selected by default and
writes `matched-k4-native-defaults-cache-on-v2` receipts. Use a verified Release
whose defaults resolve to MTP4, 32 active requests, an 8,192-token scheduling
budget, four prefix-cache entries, and CUDA graphs. The native timed process
receives only `--port` with a clean device and library environment. Its W1
activity trace adds `--trace-batches` in a separate process.

Pinned vLLM explicitly enables prefix caching and MTP4. Startup validation binds
its effective aligned GDN cache mode and block sizes to both API and worker
observations. Both arms use the same physical B200, local NUMA placement, CPU
sets, checkpoint, inputs, row order, and measurement windows. vLLM provides
timing. Native K0 and cross-engine output equality are outside this profile.

Both arms drain, issue `POST /reset_prefix_cache` with an empty JSON object, require a
successful acknowledgement, and sample idle cache counters before every warmup,
measured cohort, and the separate W1 trace. The client starts its measurement
clock after those controls. Every fixed W1–W6 request must report zero reused
prompt tokens and the engine's prefix hit counter must remain unchanged during
its cohort. W3–W6 lane prompts have distinct first tokens. Resetting between
replays keeps repeated fixed bodies cold even after a previous cohort populated
cache. Raw cohorts retain the reset response, before and after metrics, and
control timestamps. Offline `assess` checks these alongside the launch, device,
stream, and MTP evidence.

The archived `matched-k4-native-defaults-v1` protocol used a cache-disabled
Release launched with the same port-only arguments, an explicit cache-off vLLM
launch, and no cohort resets. Its receipts replay with the same `assess` command.
To capture that profile on its historical Release, use a fresh campaign and set
`PAIRED_K4_NATIVE_PROFILE=server-defaults-cache-off`. Use the archived protocol
source and pinned Release when reconstructing an exact historical campaign.

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

## Replay the historical GPU 2 arm

On the benchmark host, `artifacts/test-bench-baseline` holds the saved 32,768-token vLLM arm
for all six rows. See `REPRODUCE.md` there for the pinned vLLM commands,
`comparison.json` for the paired statistics, and
`vllm-GROUP/GROUP/measurement/summary.json` for each vLLM group's client output.
The retained run used physical UUID `GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c`,
CPUs 24–33 for the server and 34–35 for the client, and local NUMA allocation.
Reuse that UUID, CPU placement and memory policy. Measure a new vLLM arm on the
chosen card for other placement and use two fresh 8,192-token arms for the formal protocol.

Export the historical protocol to check the saved source hashes. Use the current
checkout for the candidate protocol. Run these commands in Bash from its root.

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

PAIRED_PROTOCOL_ROOT="$PWD" PAIRED_FIXTURES="$saved/fixtures" \
PAIRED_VLLM_RESULTS="$saved" PAIRED_WARMUPS=1 PAIRED_MEASURED=3 \
PAIRED_MAX_NUM_BATCHED_TOKENS=32768 \
    bash benchmarks/paired/run_native.sh /path/to/build 2 artifacts/paired-candidate-gpu2
```

The driver resolves the physical GPU before checking saved evidence. It checks the
saved device inventory and the worker's compute UUID at readiness and measurement
end, CPU and NUMA placement, the vLLM source commit receipt, observed engine
settings, the published checkpoint file hashes, the protocol and request hashes,
and the cohort schedule. It replays
saved wire captures against the frozen fixture and recorded summary. Results can
differ between engines. Each engine's own greedy IDs must repeat across cohorts.
The retained 1/3 concurrent capture predates the collector's in-run repeat check.
Its collector digest is recognized, and offline replay verifies the missing check
and reconstructs the fixture manifest identity from the summary and request bytes.

The driver writes `comparison-plan.json` before launch and keeps its SHA-256 and
the selected mode for the completion check. It replays the native wire captures
and checks repeat IDs, measured statistics, and process, device, settings and
protocol evidence before producing `GROUP/comparison.json`. Its `comparison`
field uses `same-device matched-cap` when both arms have verified 8,192-token
caps and the required cohort counts. The historical example yields `exploratory
(non-protocol cap 32768)`. At 8,192, equal override counts yield `exploratory
(non-protocol counts)`. A permitted difference yields `exploratory (count
mismatch)`. The result records which arms were replayed and whether cross-engine
IDs were compared. Current native K0 plans use the `automatic-k0-v1` policy
receipt. The driver records `/server_info` and the reported FP8 prefill, target
attention, and paired-prefill policies in `native-k0-policy.json`. Historical
plans retain their explicit launch settings. Actual kernel dispatch requires
independent profiling.

## Compare against the published frozen reference

The published W3–W6 GPU2 vLLM reference uses an 8,192-token scheduler cap. Select
one or more concurrent groups and use the published manifest for an archival comparison.
The following command uses the historical native 32,768-token scheduling cap
with the current Release's automatic operator policy:

```bash
PAIRED_COMPARISON_MODE=frozen \
PAIRED_CHECKPOINT_MANIFEST="$saved/checkpoint.sha256" \
PAIRED_MAX_NUM_BATCHED_TOKENS=32768 \
    bash benchmarks/paired/run_native.sh /path/to/build 7 artifacts/frozen-w5 W5
```

The comparison records the archive identity and uses the `published frozen reference`
label with `cross-cap` or `cross-device` when those settings differ from the reference.
The published replay keeps its historical collector and speed gates. Its
`makespan.passed` is `null`, as it is for exploratory comparisons.

## Observe and compare

Append any of `W1-W2 W3 W4 W5 W6` to select groups. Each group starts a fresh
server with 32 active slots and an 8,192-token scheduling cap. The native arm uses
the Release's automatic operator policy and enables batch CUDA graphs.
Both arms disable MTP and prefix reuse.
W1 and W2 use three warmups and nine measured requests each. The client times the
first `[DONE]` and keeps reading through the end of the HTTP response. A second
terminal, other data event after `[DONE]`, or short HTTP body invalidates the sample.
Formal W1/W2 comparison requires fresh captures with HTTP response end receipts.
W3–W6 use two warmups and five measured cohorts each, a 90 s socket timeout, a 15 s barrier
timeout and a 360 s cohort deadline. For paired W3–W6 captures, all lanes connect
before the shared barrier. Their submit timestamps must span at most 50 ms and
every request body must be sent before the first valid `[DONE]`.

The driver saves the client command with `PYTHONPATH`, GPU samples from server
startup, and raw streams once measurement begins. If IDs differ from a lane's
first warmup, the observer retains the offending cohort and records the first
differing token under `repeat_id_mismatch`. Compare separately captured diagnostic
IDs with the first warmup before using them as repeatability evidence.

The sole paired performance gate is native median makespan strictly below the
same-card vLLM median for each workload. A cohort's makespan starts at its earliest
client submit and ends at its last valid `[DONE]`. Use measured cohorts only.
`GROUP/comparison.json` records the samples and medians in nanoseconds and the
verdict under `makespan.workloads`. W1 and W2 have independent verdicts even
though they share a group. On a makespan miss, including a tie, the native driver
keeps the comparison receipt. Any failed startup, measurement, validation, or
makespan gate stops the native driver before the next group and exits nonzero.

Use output IDs/s, TTFT, TPOT, end-to-end latency, p95, and the intervals when all
streams are open or generating as diagnostics. A lane can start producing
output after another lane completes. The W1/W2 summary reports rates and latency
under `workloads.W1` and `workloads.W2`. W3–W6 reports them under `throughput`
and `latency_ms`. `wire_and_timing_gates` and the retained archive's
`fresh_vllm_thresholds` keep the historical speed checks available for diagnosis.
