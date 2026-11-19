# Third source checkpoint

This checkpoint preserves scratch experiments anchored at published commit
`c2045c1cd769209b3476d799237304723e79c680`, tree `4d41bd5baab2d45d085c50ccb7b7a17d98a26385`.
The replay uses one clean tree per arm. Each arm has its own base, patch order and
final path, Git executable mode and SHA-256 inventory in `manifest.json`. The source
trees are independent experiments.

The `serial-cta` and `w3-fused` prerequisites come from the sealed published
`2026-11-19-performance` replay. The latter includes the intermediate exact,
combined, bounded flash, 8,192-row flash and layer-fusion sources. The W6 parent
reuses two `serial-cta` blobs and applies a small identity delta before applying
the distinct W6 overlay. `historical.json` records original receipt hashes and
the specific scratch comparisons. The W3/W6 scope omits three `.orig` backups.
Adaptive MTP keeps the patch against the published verifier base and drops its
historical prose hunks. The model suite adds ten test files to each base.

The overlays are from this project's scratch work. Keep the base and file
notices when restoring or reviewing. The published anchor has no root license
grant. The upstream SGLang and vLLM revisions and the model snapshot are
references or separately supplied runtime inputs. The package contains source,
token recipes, and digests. Weights, model vectors, binaries, captures, and
external environments are supplied separately where needed.

## Restore offline

Use Git with complete local objects for the anchor and its ancestors, Python
3.10 or newer, GNU tar and coreutils. Set `seal` from the separately published
SHA-256 of `SHA256SUMS`. Use new sibling directories outside the local clone and
the checkpoint directory.

```bash
export PYTHONDONTWRITEBYTECODE=1
bundle=/absolute/path/to/checkpoints/2026-11-19-progress-3
repo=/absolute/path/to/local/clone-or-bare-repo
run=/absolute/path/to/new/replay-work
anchor=c2045c1cd769209b3476d799237304723e79c680
seal=REPLACE_WITH_PUBLISHED_CHECKSUM_FILE_SHA256
mkdir "$run"
python3 -IB "$bundle/replay.py" --repo "$repo" --seal "$seal" --output "$run/trees"
python3 -IB "$bundle/replay.py" --seal "$seal" \
  --check-tree "$run/trees/w3-async2" w3-async2
mkdir "$run/anchor"
git -C "$repo" archive "$anchor" | tar -xf - -C "$run/anchor"
```

Select named arms after `--output` for a smaller restore. A missing object,
changed input or mode, unlisted path, bad output receipt or incomplete series
returns an error. The final output appears only after all selected trees verify.
Source replay uses no build, model or GPU.

For runtime work, activate the pinned project Pixi C++/CUDA environment with
GCC, XMake, LLVM 23 and the host CUDA 13.4 compatibility driver library. Use an
already leased single B200. Use a free local port for each HTTP run and stop
the service between arms. Record the exact compiler, XMake, driver, device UUID,
checkpoint manifest, options, and environment with new evidence.

```bash
export CHECKPOINT=/data0/shared/Qwen3.8-27B-FP8
export CUDA_VISIBLE_DEVICES=GPU_UUID_FROM_YOUR_LEASE
export PATH=/data1/shared/dongwu.chen/bin:$PATH
export PYTHONPATH="$run/anchor"
mkdir "$run/build" "$run/cache" "$run/probes"
export CUDA_CACHE_PATH="$run/cache/cuda"
build_arm() {
  bash "$bundle/build.sh" "$seal" "$run/trees/$1" "$run/build/$1"
}
bin() {
  printf '%s/build/%s/out/bin/%s\n' "$run" "$1" "$2"
}
(
  export GIT_CEILING_DIRECTORIES="$run"
  cd "$run/probes"
  git apply --check "$bundle/diagnostics/probes.patch" &&
    git apply "$bundle/diagnostics/probes.patch"
)
```

`build_arm` makes a separate build with devices hidden and hashes its executable
targets. For concurrent fixtures use an existing offline Python with `tokenizers`
that passes the published tokenizer and request digest checks.

```bash
(cd "$run/anchor" && "$fixture_python" -B -m benchmarks.concurrent.freeze_prompts \
  "$CHECKPOINT/tokenizer.json" "$run/concurrent")
```

## W3 two-slot staging

GPU0, candidate first, two warmups and five measured four-lane HTTP cohorts
at 8,192 input and 1,024 output IDs per lane. The fused control produced 145.99
IDs/s median (138.84–150.35). Async2 produced 159.00 (157.30–165.57).
Median TTFT went from 3,920.13 to 4,519.43 ms. P95 TPOT went from 25.01 to
21.30 ms and p95 end-to-end from 29,501.31 to 26,038.98 ms. All 28,672 emitted
IDs agreed across the trial set. A separate teacher replay compared 127,139,840
BF16 words bitwise. The native arms missed the published gates. The pinned
vLLM HTTP result of 379.03 IDs/s came from a separate GPU4 run.

```bash
for arm in w3-async2 w3-fused-control w3-async2-scores w3-score-control; do
  build_arm "$arm"
done
# Launch each HTTP arm in the recorded order on the same device.
"$(bin "$arm" sglang-server)" --checkpoint "$CHECKPOINT" --device 0 --port 38123 \
  --max-active 4 --fast-prefill-rows 8192 --fast-gdn --flash-prefill \
  --split-decode --graph-decode
# Run in another shell for that arm. Preserve complete captures even on a gate exit.
python3 -B -m benchmarks.concurrent.client W3 http://127.0.0.1:38123 \
  "$run/concurrent" "$run/http-$arm"
python3 -IB "$bundle/diagnostics/compare.py" w3-http \
  "$run/http-w3-async2" "$run/http-w3-fused-control"
python3 -IB "$bundle/fixtures.py" w3 "$run/concurrent" "$run/w3-history.txt"
for arm in w3-async2-scores w3-score-control; do
  mkdir "$run/$arm"
  (cd "$run/$arm" && "$(bin "$arm" batch-replay)" \
    "$CHECKPOINT" 8192 15 128 "$run/w3-history.txt")
done
python3 -IB "$bundle/diagnostics/compare.py" w3 "$run/w3-async2-scores" "$run/w3-score-control"
```

## W2 large-M pair staging

The retained 2,048-row tile and the pair-staging overlay are separate arms.
The same-device ABBA W2 comparison gave 151.633 to 131.750 ms median TTFT
and 151.725 to 132.106 ms p95, with 18 observations per arm. The separately
launched pinned vLLM reference gave 79.454 ms median. Native and pinned-teacher
replays compared 317,849,600 BF16 words and 1,280 native winners. Recovery
tests forced paired-kernel rejection and wide scratch allocation failure.

```bash
for arm in w2-2048 w2-pair w2-2048-scores w2-pair-scores; do build_arm "$arm"; done
export SGLANG_W2_PREFILL_ROWS=2048
# For the four runs use w2-2048, w2-pair, w2-pair, w2-2048 in that order.
"$(bin "$arm" sglang-server)" --checkpoint "$CHECKPOINT" --device 0 --port 38123 \
  --fp8-prefill fast-native --max-active 1 --mtp-steps 0
python3 -B -m benchmarks.streaming.client --url http://127.0.0.1:38123 \
  --output "$run/$label" --freeze-baseline
python3 -IB "$bundle/fixtures.py" w2 "$run/anchor/benchmarks/streaming" "$run/w2"
for teacher_set in native vllm; do
  for arm in w2-2048-scores w2-pair-scores; do
    cp -a "$run/w2/$teacher_set" "$run/$arm-$teacher_set"
    "$(bin "$arm" serial-score-probe)" "$CHECKPOINT" fast-native "$run/$arm-$teacher_set" score
  done
  python3 -IB "$bundle/diagnostics/compare.py" w2 \
    "$run/w2-2048-scores-$teacher_set" "$run/w2-pair-scores-$teacher_set"
done
"$(bin w2-pair-scores serial-score-probe)" "$CHECKPOINT" fast-native "$run/w2/reset" reset
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
for fault in reject_pair_kernel fail_wide_allocation; do
  "$CXX" -std=c++20 -O2 -shared -fPIC -I"$toolkit/include" \
    "$run/probes/$fault.cpp" -ldl -o "$run/$fault.so"
done
```

For the fault controls, launch `w2-pair` with one `LD_PRELOAD` interposer at a
time and submit W1, W2, W2, W1 using the streaming client `--probe` option.
Check the native history IDs and length finish. The paired-kernel control
logged 768 rejections. The allocation control logged three 553,844,736-byte
rejections and at least four successful 34,615,296-byte allocations.

## Adaptive serial MTP and segmented prefill

Adaptive MTP was compared with the earlier ordered target verifier and the
published always-parallel verifier. On the primary lease, 162 measured and
54 warmup HTTP requests preserved the greedy K0 sequence. At depth four,
median end-to-end latency for adaptive versus always-parallel was 800.3
versus 969.8 ms on Field, 2,524.2 versus 2,809.2 ms on W1, and 634.9 versus
609.6 ms on Code. Within the adaptive arm, depth zero had lower median
end-to-end latency than depth four on all three fixtures. At depth zero,
the medians were nearly identical across the three arms. The repeat only
partly reproduced the primary depth-one regression. The independent probe
checked lazy allocation and fallback, depth-zero cleanup and session state.

Segmented exact prefill uses the final donor at `da3b8b0`, with the independent
tree `1c447e9663e3ef4879fd3ec387ed9fed0c0ae5de`. The planner passed 2,885
cases and the reserve mutant failed 101. The checkpoint probe matched 61 IDs,
15,147,520 BF16 words and 57 request-state observations. In paired shortened
256-input/8-output HTTP, packed versus separate width two gave 7.584 versus
8.384 IDs/s and 1,577.452 versus 1,636.097 ms p95 TTFT. Width four gave 9.437
versus 8.592 IDs/s and 3,114.809 versus 3,472.667 ms p95 TTFT. The final CLI
rejected widths 16 and 32 before CUDA initialization.

```bash
for arm in mtp-adaptive mtp-parallel-control mtp-serial-control segmented-prefill; do
  build_arm "$arm"
done
mkdir "$run/mtp-http" "$run/segmented-http"
MTP_BUILDS="$run/build" ADAPTIVE_TEST_RUN="$run/mtp-http" \
  python3 -B "$bundle/diagnostics/mtp_http.py" --reverse 4
mkdir "$run/mtp-depth-one-repeat"
MTP_BUILDS="$run/build" ADAPTIVE_TEST_RUN="$run/mtp-depth-one-repeat" \
  python3 -B "$bundle/diagnostics/mtp_http.py" --depths 0 1
python3 -IB "$bundle/diagnostics/segmented_host.py" \
  "$run/trees/segmented-prefill/src/decoder_batch.cpp" "$run/segmented-host"
SEGMENTED_RUN_DIR="$run/segmented-http" python3 -B "$bundle/diagnostics/segmented_http.py" \
  "$(bin segmented-prefill sglang-server)" "$CHECKPOINT" "$run/concurrent"
```

Compile the standalone probes from the separately extracted C++ files.
The MTP probe uses the fixed checkpoint path shown above.

```bash
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
ln -s "$run/build/mtp-adaptive/source" "$run/probes/candidate"
link_probe() {
  local kind=$1 arm=$2
  shift 2
  local built="$run/build/$arm"
  "$CXX" -std=c++17 -O3 -I"$built/source" -I"$built/source/include" \
    -I"$toolkit/include" "$@" "$run/probes/${kind}_probe.cpp" -o "$run/$kind-probe" \
    -L"$built/out/lib" -L"$toolkit/lib" -L"$CONDA_PREFIX/lib" -L"$CONDA_PREFIX/lib64" \
    -Wl,-rpath,"$toolkit/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" \
    -lsglang_decoder -lsglang_checkpoint -lsglang_cuda -lsglang_prefill_fp8_sm100 \
    -lLLVM-23 -lcudart -lcudadevrt -lcudart_static -lrt -lpthread -ldl
}
link_probe mtp mtp-adaptive -Wl,--wrap=cudaMalloc,--wrap=cudaMallocHost,--wrap=cudaMemcpyAsync \
  -Wl,--wrap=cudaStreamSynchronize,--wrap=cudaGraphLaunch
"$run/mtp-probe"
link_probe segmented segmented-prefill -fno-access-control
"$run/segmented-probe" "$CHECKPOINT" correctness
for width in 2 4; do "$run/segmented-probe" "$CHECKPOINT" timing "$width" 128 1024; done
```

## Model-reference suite

Both native builds produced identical per-row measurements. The historical
H2047 check failed on both tested trees. On that archived row, both native
exact-prefill builds chose `271`, while pinned SGLang chose `279`, with
reference regret 2.875. The other 19 archived rows passed. Both native builds
passed all 52 repeated holdout rows and matched 51 winners, including every
repeat-confirmed clear winner. The next-position substitution and vocabulary
permutation failed the frozen checks.

```bash
build_arm model-reference
build_arm model-reference-graphs
suite="$run/trees/model-reference/tests/model_reference"
python3 -IB "$bundle/fixtures.py" model "$run/model-inputs.json"
```

Follow the restored suite's README for the authenticated SGLang Python tree
and checkpoint manifest, offline reference dependencies, capture caches and
GPU lease. The pinned SGLang source archive, revision and checkpoint manifest
digests are in `historical.json`. Supply an existing isolated reference
environment and keep `TMPDIR` short enough for Unix-domain sockets.

```bash
"$reference_python" -B "$suite/reference.py" "$CHECKPOINT" "$checkpoint_manifest" \
  "$source_archive" "$run/model-inputs.json" "$run/fresh-capture" --all --repeat 2
python3 -IB "$bundle/fixtures.py" capture "$suite" "$run/model-inputs.json" \
  "$run/fresh-capture" "$run/model-drift.json" --output "$run/model-bundle"
```

The digest comparison selects the historical archived positions and retains
both repeats for the new cases. A capture can drift, especially on the five
cases collected under older conditions. A drift report preserves the expected
and observed digests and exits nonzero. A fully matching capture can supply the
suite's frozen-bundle commands, including `compare.py --self-test`, the
teacher interposer and the published and graph HTTP width runs. Use
`$run/build/model-reference/out` and `$run/build/model-reference-graphs/out`
as the respective build directories. Retain divergent captures as separate
evidence with their own provenance.

## W6 width-four diagnostic repair

This repair switches the W3-derived W6 donor's width-four FP8 projection
arithmetic in the probe and seeds the device winner for cancellation and reuse.
The corrected scratch arm matched two width-two groups for all 14,899,200
BF16 words and 60 winners at position 131,072 over 15 steps. On one GPU7
lease the original MMA and packed graphs measured 89.2251 and 89.4419 ms
median per step over 42 warm samples each. The published serving graph has
different eligibility at this position. The measurements describe this
diagnostic and its donor.

```bash
build_arm w6-width-four
python3 -IB "$bundle/fixtures.py" w6 "$run/concurrent" "$run/w6-fixtures.bin"
"$(bin w6-width-four schedule-parity)" "$CHECKPOINT" "$run/w6-fixtures.bin" \
  "$run/w6-output" 131072 cancel >"$run/w6.log" 2>&1
python3 -IB "$bundle/diagnostics/compare.py" w6 "$run/w6-output" "$run/w6.log"
```
