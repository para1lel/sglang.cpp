# Batch CUDA Graph checks

From the repository root in managed Pixi `default`, hold the shared UUID lock on an idle B200 for the entire run and set `GPU_UUID`. Use the pinned Qwen3.8-27B-FP8 checkpoint and the Release server with both opt-in probes.

```bash
out="$PWD/build/batch-graph-check"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
xmake f -P "$PWD" -o "$out" -p linux -a x86_64 -m release \
    --mxfp8-gqa-check=y --batch-graph-lanes=y
xmake build -P "$PWD" sglang-server mxfp8-gqa-check batch-graph-lanes
GPU_UUID="${GPU_UUID:?Set the reserved B200 UUID}" \
    bash tests/batch/run_batch_graph.sh "$out" "$checkpoint" "$out/evidence"
```

The runner uses fresh eager and graph processes with the selected prefill operators.
It leaves the command inputs, HTTP SSE, full BF16 scores, scheduler traces, and CUPTI
runtime/device records in the evidence directory.

- The CUDA operator probe rebinds distinct MXFP8 cache and page-table addresses,
  crosses positions 127 and 128, and checks KV bytes, inactive storage, and BF16
  attention. It rebinds separate GDN history and recurrence banks and compares
  output, BF16 history, and FP32 state against ordinary decode.
- The model probe changes sparse active sets at widths one through four.
  It checks cancellation, zero-budget retirement, positive slot reuse, greedy IDs,
  and complete BF16 scores against eager. CUPTI must record one singleton graph
  through capture, launch, and destruction. The width sweep checks every logical
  width from two through 32, where regular B200 K0 decode runs eagerly.
- Fault probes inject unsupported capture, failed instantiation, replay rejection,
  scratch and instantiation allocation failures, and an uncertain post-launch
  synchronization. They check eager fallback, graph cleanup, and fail-closed
  behavior after uncertain completion.
- The singleton probe uses slot 31 in a 32-slot batch. It checks eager warmup,
  replay across request replacement, mixed prefill, zero-budget retirement,
  the inline score limit, and width-local launch rejection. CUPTI pairs each
  reported mask with a device launch. The full BF16 score comparison includes
  both ends of the 1,536–2,560 visible-token page-parallel attention window.
- The K1 target verification probe distinguishes two-row target launches from K0
  replay. It checks sparse slots, request replacement, mixed K0 and K1 decode,
  and eager execution for verifier groups with multiple lanes, more than two
  target rows, or a one-row budget.
  Preparation, capture, and instantiation rejection fall back to eager. The
  fixture checks same-slot replacement after a successful drain and batch
  disablement after launch-call errors, uncertain teardown, a terminal CUDA
  error, or a terminal error during recovery. It verifies that the disabled
  batch rejects further decode and admission.
  The HTTP arm compares greedy and seeded SSE, visibility limits, rejection,
  EOS, output caps, and cancellation.
- The public HTTP probe sends distinct prompts across decode widths one through
  four and checks independent EOS and explicit stop, cancellation, overlapping
  replacement, and recovery. It also runs a seeded sampled request alongside a
  greedy request and checks eager execution during their shared decode. Both server
  routes must return identical SSE and release request state.

Expect `Batch CUDA Graph operator, lane, fault and HTTP checks passed`. The
[evidence checker](check_graph_evidence.py#L1)<!-- prefix: "\"\"\"Check rea" -->
can be run again against an existing evidence directory.

Run the singleton regression with a fresh evidence destination.

```bash
GPU_UUID="${GPU_UUID:?Set the reserved B200 UUID}" \
    bash tests/batch/run_batch_graph.sh "$out" "$checkpoint" "$out/singleton" singleton
python3 -IB tests/batch/check_graph_evidence.py "$out/singleton" --singleton-only
```

Run the slot-mask regression without a checkpoint or GPU.

```bash
python3 -IB tests/batch/check_graph_evidence.py --check-slot-mask
```

Run the CPU output terminal regression with the managed Pixi environment active.
It holds a token callback while the real batch worker stages the next step.
It checks that abort drops the queued ID and that published stop and length
results drain their queued IDs exactly once.

```bash
mkdir -p "$out"
bash tests/batch/run_service_output_terminal.sh "$PWD" "$out/service-output-terminal"
```

Run the K1 probe with a fresh evidence destination.

```bash
GPU_UUID="${GPU_UUID:?Set the reserved B200 UUID}" \
    bash tests/batch/run_batch_graph.sh "$out" "$checkpoint" "$out/k1" k1
python3 -IB tests/batch/check_graph_evidence.py "$out/k1" --k1-only
```

## Wide K0 serving

Use the Release `sglang-server` and a fresh evidence directory for the CLI checks.
CUDA remains hidden while the runner checks valid limits, invalid inputs, and
positive MTP depths at narrow and wide active limits.

```bash
python3 -IB tests/batch/run_wide.py --binary "$out/bin/sglang-server" \
    --cpu-only --evidence "$out/wide-cli"
```

Link the lane probe and CUPTI observer against the same Release build. The first wraps two symbols to check one full-width model call and packed FP8 projection fan-out.

```bash
toolkit="${CONDA_PREFIX:?Activate the managed Pixi default environment}/targets/x86_64-linux"
nccl=$(python -I -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl")')
forward=_ZNK6sglang7Decoder5Model7forwardEPKNS_7decoder11ForwardLaneEiP11CUstream_st
gemm=_ZN6sglang13text_fp8_gemmEPKhPK13__nv_bfloat16iiPKNS_17TextFp8ProjectionEiP11CUstream_st
cuda_lib=(-L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib")
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude -I"$toolkit/include" \
    tests/batch/check_wide_lanes.cpp "${cuda_lib[@]}" -L"$toolkit/lib/stubs" \
    -Wl,--wrap="$forward" -Wl,--wrap="$gemm" \
    -Wl,--start-group "$out"/lib/libsglang_*.a -Wl,--end-group \
    -Wl,-rpath,"$nccl/lib" "$nccl/lib/libnccl.so.2" -lcuda -lcudart -pthread -o "$out/bin/check-wide-lanes"
"$CXX" -std=c++17 -O2 -fPIC -shared -I"$toolkit/include" \
    tests/batch/graph_activity.cpp "${cuda_lib[@]}" -lcupti -o "$out/bin/wide-activity.so"
CUDA_VISIBLE_DEVICES="${GPU_UUID:?Set the reserved B200 UUID}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python3 -IB tests/batch/run_wide.py --binary "$out/bin/sglang-server" \
    --checkpoint "$checkpoint" --lanes "$out/bin/check-wide-lanes" \
    --activity "$out/bin/wide-activity.so" --evidence "$out/wide"
```

Both widths, 16 and 32, match native and HTTP IDs with serial output. All clients emit a token before the first `[DONE]`. Check queueing, cancellation, slot reuse, and request bytes returning to zero. The trace shows a full-width step and eager multi-lane decode, with at most 128 prompt and decode rows per step. The native probe requires one full-width eager call and eager pairs. A warmed highest-slot singleton replays with its mask bit set and a corresponding CUPTI launch.

## CPU score slab check

Use the Release checkpoint and JSON archives in managed Pixi. CUDA stays hidden. Host stubs cover allocation and graph APIs without weights or a GPU.

```bash
toolkit="${CONDA_PREFIX:?Activate the managed Pixi default environment}/targets/x86_64-linux"
build="${out:?Set out to the Release build directory}"
score_check=build/check-score-slab
mkdir -p "$(dirname "$score_check")"
"$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
    -ffunction-sections -fdata-sections -Wl,--gc-sections \
    -Iinclude -I"$toolkit/include" \
    tests/batch/check_score_slab.cpp src/runtime/backend/cuda_resources.cpp \
    src/runtime/backend/cuda_error.cpp src/runtime/backend/retired_request_blocks.cpp \
    src/runtime/model/qwen_request.cpp src/text/text_dimensions.cpp \
    "$build/lib/libsglang_checkpoint.a" "$build/lib/libsglang_json.a" \
    -L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib" -lcudart -o "$score_check"
CUDA_VISIBLE_DEVICES= "$score_check"
```

Expect an allocation rejection for each of the four bound lanes followed by
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
requests run at capacity one and share a scheduler step with a text
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

The row-accounting cases use explicit cache-off controls. For the separate eager K4
case, pass `--case eager-cache --lmcache-socket "$LMCACHE_SOCKET"` to `run_admission.py`
with an isolated, ready [daemon on the same GPU](../../docs/serving/lmcache.md#start-the-daemon). Each run uses separate salts for the cold and primed requests. The runner waits for checkpoint publication before checking the 8,192-token hit. It also checks deterministic output, draft activity, and zero graph replays.

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
output budget reserve eight KV pages. The probe allows the 32-byte page table
allocation, then fails the combined 34,603,008-byte GQA key, value, and scale
allocation while a peer is decoding.

A capacity failure must return HTTP 503, queue the page table for recycling,
and let the peer decode and a fresh request complete. The runner cancels the
peer, checks that request bytes return to zero, and verifies a new stream succeeds.
Once the active lanes retire, the runner requires confirmed release of that page table on the GPU.
A successful synchronous free suffices. An asynchronous free must also complete on its owning stream.
One terminal control injects the CUDA illegal-address
status at the same allocation. The other injects a memory allocation
failure there, then reports an illegal address during stream synchronization.
Both require HTTP 500 for the triggering and fresh requests and an SSE
error for the active peer.

## Checkpoint lifecycle

Start a dedicated [LMCache 0.5.8 daemon](../../docs/serving/lmcache.md#start-the-daemon) on the reserved B200.
Use a private absolute path for `LMCACHE_SOCKET` and wait for the daemon to be ready.
The runner rejects the standard daemon sockets. It drains requests and pending stores,
then clears the daemon store before W1 and between independent cases. This removes other
workers' records too. Keep the daemon exclusive to this run and use a fresh evidence directory.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?Set the reserved B200 UUID}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python3 -IB tests/batch/checkpoint_lifecycle.py "$out/bin/sglang-server" \
    "$checkpoint" "$out/checkpoint-lifecycle" --lmcache-socket "${LMCACHE_SOCKET:?}"
```

The runner checks LMCache availability, K4, and the default 32,768-row budget.
It compares successful STORE counts after running and waiting requests and pending stores reach zero.
The saved HTTP metrics must show zero LMCache I/O errors and zero declined stores.
Batch traces show K4 verification, the one-row decode at a crossed boundary, and terminal retirement.
A daemon STORE receipt can arrive after retirement.

- W1 saves one terminal checkpoint at 255 processed tokens. Two session branches restore 255 target rows and 254 draft rows.
  Both must emit the same token IDs as their cold controls.
  Replaying W1 must acknowledge one duplicate STORE with zero new GQA tokens and zero bytes written for GQA, GDN, or draft data.
- The periodic cases probe the terminal endpoint and every 8,192-token boundary reached in prefill or decode. Divergent prompts probe the deepest boundary first.
- An abort leaves no terminal checkpoint. A disconnect after the first boundary preserves its periodic checkpoint.
  Both interrupted turns are ineligible as session parents.
- Reversing two stop strings selects different endpoints in one K4 packet. Each continuation restores the selected endpoint and matches cold output IDs.

The evidence directory contains the server arguments, requests, responses, raw SSE, scheduler traces, and cache metrics.

