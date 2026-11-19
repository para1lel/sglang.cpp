# Batch CUDA Graph checks

Use the managed Pixi `default` environment and reserve an idle B200 for the entire
run. Set `GPU_UUID` to its device UUID while holding the shared UUID lock. Use the
pinned Qwen3.8-27B-FP8 checkpoint.

From the repository root, build the normal release server and the two opt-in checks.

```bash
out="$PWD/build/batch-graph-check"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
xmake f -P "$PWD" -o "$out" -p linux -a x86_64 -m release \
    --mxfp8-gqa-check=y --batch-graph-lanes=y
xmake build -P "$PWD" sglang-server
xmake build -P "$PWD" mxfp8-gqa-check
xmake build -P "$PWD" batch-graph-lanes
GPU_UUID="${GPU_UUID:?Set the reserved B200 UUID}" \
    bash tests/batch/run_batch_graph.sh "$out" "$checkpoint" "$out/evidence"
```

The runner uses fresh processes for eager and graph under both exact and tensor-core
prefill. It leaves the command inputs, HTTP SSE, full BF16 scores, scheduler traces,
and CUPTI runtime/device records in the evidence directory.

- The CUDA operator probe rebinds distinct MXFP8 cache and page-table addresses,
  crosses positions 127 and 128, and checks KV bytes, inactive storage, and BF16
  attention. It rebinds separate GDN history and recurrence banks and compares
  output, BF16 history, and FP32 state against ordinary decode.
- The model probe changes sparse active sets at widths two, three, and four.
  It checks cancellation, zero-budget retirement, positive slot reuse, greedy IDs,
  and complete BF16 scores against eager. Each graph mask must equal the slots
  with a decode result. Within this short scenario, once a width has replayed,
  each subsequent decode-only step at that width must replay. CUPTI must record
  one actual CUDA graph per width through capture, launch, and destruction.
- Fault probes inject unsupported capture, failed instantiation, replay rejection,
  scratch and instantiation allocation failures, and an uncertain post-launch
  synchronization. They check eager fallback, graph cleanup, continued replay at
  other widths, and fail-closed behavior after uncertain completion.
- The long-score probe crosses positions 5,119–5,121 with tensor-core prefill.
  It checks lazy allocation, changing score strides, mixed steps, score
  allocation failure, cancellation, and slot reuse. It matches greedy IDs and
  complete BF16 rows to eager, correlates masks with CUPTI launches, and
  requires zero request bytes at cleanup.
- The public HTTP probe sends distinct prompts with all three widths and checks
  independent EOS and explicit stop, cancellation, overlapping replacement, and
  recovery. Both server routes must return identical SSE and release request state.

Expect `Batch CUDA Graph operator, lane, fault and HTTP checks passed`. The
[evidence checker](check_graph_evidence.py#L1)<!-- prefix: "\"\"\"Check rea" -->
can be run again against an existing evidence directory.

Run the long-score boundary with a fresh evidence destination.

```bash
GPU_UUID="${GPU_UUID:?Set the reserved B200 UUID}" \
    bash tests/batch/run_batch_graph.sh "$out" "$checkpoint" "$out/boundary" boundary
python3 -IB tests/batch/check_graph_evidence.py "$out/boundary" --boundary-only
```

Run the slot-mask regression without a checkpoint or GPU.

```bash
python3 -IB tests/batch/check_graph_evidence.py --check-slot-mask
```

## Wide K0 serving

Use the Release `sglang-server` and a fresh evidence directory for the CLI checks.
CUDA remains hidden while the runner checks valid limits, invalid inputs, and
positive MTP depths at narrow and wide active limits.

```bash
python3 -IB tests/batch/run_wide.py --binary "$out/bin/sglang-server" \
    --cpu-only --evidence "$out/wide-cli"
```

From the same source and build, compile the native lane probe and CUPTI observer.
The probe wraps two native entry points to check one full-width model call and
packed FP8 projection fan-out.

```bash
toolkit="${CONDA_PREFIX:?Activate the managed Pixi default environment}/targets/x86_64-linux"
forward=_ZNK6sglang7Decoder5Model7forwardEPKNS_7decoder11ForwardLaneEiP11CUstream_st
gemm=_ZN6sglang13text_fp8_gemmEPKhPK13__nv_bfloat16iiPKNS_17TextFp8ProjectionEiP11CUstream_st
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude -I"$toolkit/include" \
    tests/batch/check_wide_lanes.cpp -L"$out/lib" -L"$toolkit/lib" \
    -Wl,-rpath,"$toolkit/lib" -Wl,--wrap="$forward" -Wl,--wrap="$gemm" \
    -lsglang_decoder -lsglang_checkpoint -lsglang_json -lsglang_cuda -lsglang_prefill_fp8_sm100 \
    -lcudart -o "$out/bin/check-wide-lanes"
"$CXX" -std=c++17 -O2 -fPIC -shared -I"$toolkit/include" \
    tests/batch/graph_activity.cpp -L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib" \
    -lcupti -o "$out/bin/wide-activity.so"
CUDA_VISIBLE_DEVICES="${GPU_UUID:?Set the reserved B200 UUID}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python3 -IB tests/batch/run_wide.py --binary "$out/bin/sglang-server" \
    --checkpoint "$checkpoint" --lanes "$out/bin/check-wide-lanes" \
    --activity "$out/bin/wide-activity.so" --evidence "$out/wide"
```

The GPU runner checks both 16 and 32 lanes by default. It compares native and
HTTP token IDs with serial exact-prefill results. All clients must emit a token
before the first `[DONE]`. The runner checks queueing, cancellation, and slot
reuse. Request bytes must return to zero after retirement.

The HTTP trace must show a full-width scheduler step and eager decode above
the four-lane graph limit. Each step can contain at most 128 prompt and decode
rows with the explicit test budget. The native probe requires one full-width
eager decode call. A sparse pair that includes the highest slot must replay at
graph width two. The mask must include the highest slot bit, and CUPTI must
record the replay. The existing graph and MTP runners cover the narrow serving paths.

## CPU score slab check

From the repository root in the managed Pixi `default` environment, build and
run the score-slab check with CUDA hidden. It needs the CUDA headers and uses
host stubs for allocation and graph APIs. It requires no checkpoint or GPU.

```bash
toolkit="${CONDA_PREFIX:?Activate the managed Pixi default environment}/targets/x86_64-linux"
score_check=build/check-score-slab
mkdir -p "$(dirname "$score_check")"
"$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" -std=c++17 -O2 -Wall -Wextra -Werror \
    -ffunction-sections -fdata-sections -Wl,--gc-sections \
    -Iinclude -I"$toolkit/include" \
    tests/batch/check_score_slab.cpp src/decoder_resources.cpp \
    src/checkpoint.cpp src/json.cpp src/utf8.cpp -o "$score_check"
CUDA_VISIBLE_DEVICES= "$score_check"
```

Expect a rejection at each of the four packed lanes followed by
`Score bounds, bounded ownership and executable reuse passed`.

## Scheduler admission

The full batch runner checks scheduler admission. To run only admission with a
Release server and the pinned checkpoint, select `admission` on the reserved GPU.

```bash
GPU_UUID="${GPU_UUID:?Set the reserved B200 UUID}" \
    bash tests/batch/run_batch_graph.sh "$out" "$checkpoint" "$out/admission" admission
```

The check records `/health` for omitted limits and explicit widths one and four.
It verifies low-concurrency progress and a waiting request at full occupancy.
After resetting an active stream, surviving streams advance, the waiter finishes,
and request bytes return to zero. The small-budget arm checks prompt boundaries
and counts prompt, decode, and MTP verification rows in scheduler traces. Media
requests exercise the serial route and share a scheduler step with a text
request at the default width. The CLI accepts K1 through K4 with up to 32
active requests. The scheduler shares one row budget across all active slots
and verifies drafts in groups of up to four lanes. The wide K1 arm fills the
default 32 slots and checks a waiting request and slot reuse. The trace must
account for every ready decoder, show verification in every slot, and show
more than four verifier lanes in one iteration. HTTP stream workers can make
slots ready on different iterations. The full batch runner also selects all
32 ready lanes directly in one `DecoderBatch` step. It requires 64 verifier
rows, compares each lane's IDs with target-only decoding, and checks that
cancellation releases all request bytes.

Run `run_admission.py --case pressure` separately for 32 requests with
32,769 prompt IDs in total. The trace records actual work per iteration, which
can also be limited by arrival order and per-request tiles.

### KV admission fault

Run this opt-in check with the Release server's build directory and matching
source tree. Keep the GPU reservation for all three cases.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?Set the reserved B200 UUID}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python3 -IB tests/batch/run_admission.py --source "$PWD" --binary "$out/bin/sglang-server" \
    --case kv-capacity --case kv-context --case kv-drain \
    --checkpoint "$checkpoint" --evidence "$out/kv-admission-fault"
```

The runner replays the recorded XMake link and verifies the unmodified
executable byte for byte. A second link wraps CUDA allocation, release and
stream synchronization.
The server runs with its default limits. A four-token prompt and a 1,021-token
output budget allocate eight KV pages. The probe allows the key payload
allocation and injects a failure on the value payload allocation while a
peer is decoding.

A capacity failure must return HTTP 503, release the partial allocation,
and let the peer decode and a fresh request complete. The peer is then
cancelled, and the runner checks that request bytes return to zero and a new
stream succeeds. One terminal control injects the CUDA illegal-address
status at the same allocation. The other injects a memory allocation
failure there, then reports an illegal address during stream synchronization.
Both require HTTP 500 for the triggering and fresh requests and an SSE
error for the active peer.
