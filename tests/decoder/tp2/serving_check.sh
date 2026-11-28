#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 || $# > 6 )); then
    printf '%s\n' 'Usage: serving_check.sh BUILD CHECKPOINT UUID0 UUID1 OUTPUT' \
        '  [all|build|static|tp1|tp2-mixed|tp2-fused|cache|store|restart]' >&2
    exit 2
fi
build=$(realpath -- "$1")
checkpoint=$(realpath -- "$2")
first=$3 second=$4
output=$(realpath -m -- "$5")
phase=${6:-all}
[[ $phase == all || $phase == build || $phase == static || $phase == tp1 ||
   $phase == tp2-mixed || $phase == tp2-fused || $phase == cache ||
   $phase == store || $phase == restart ]]
devices=("$first")
if [[ $phase != tp1 ]]; then
    [[ $first != "$second" ]]
    devices+=("$second")
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
source=$(realpath -- "${SGLANG_TEST_SOURCE:-$root}")
shared=${SGLANG_HOST_ROOT:-/data1/shared/dongwu.chen}
toolkit=${CONDA_PREFIX:?Run in the managed Pixi environment}/targets/x86_64-linux
nccl=$(python -I -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl")')
mkdir -p -- "$output"
cd -- "$root"
if [[ $phase == all || $phase == build || $phase == static || $phase == tp1 ||
      $phase == tp2-mixed || $phase == tp2-fused ]]; then
libraries=("$build"/lib/libsglang*.a)
sha256sum "${libraries[@]}" "$source"/include/sglang/{decoder_mtp.h,decoder_batch.h} \
    tests/decoder/tp2/{serving_check.cpp,serving_probe.cpp,prefill_plan_check.cpp} \
    >"$output/native-inputs.sha256"
draft_wave=$(nm -P --defined-only "${libraries[@]}" | awk '
    /^_ZN6sglang7decoder15MtpDraftRequest10enqueue_tpE[^ ]* T / { print $1 }
')
[[ $draft_wave == _ZN6sglang7decoder15MtpDraftRequest10enqueue_tpE* && $draft_wave != *$'\n'* ]]
"${CXX:?Missing Pixi host compiler}" -std=c++23 -O2 -Wall -Wextra -Werror \
    -I"$source/include" -isystem "$toolkit/include" -isystem "$nccl/include" \
    -DSGLANG_REAL_DRAFT_WAVE="__real_$draft_wave" \
    -DSGLANG_WRAP_DRAFT_WAVE="__wrap_$draft_wave" \
    tests/decoder/tp2/serving_check.cpp tests/decoder/tp2/serving_probe.cpp \
    tests/decoder/tp2/prefill_plan_check.cpp \
    -Wl,--wrap=ncclAllReduce -Wl,--wrap=ncclAllGather -Wl,--wrap=cudaMemcpyAsync \
    -Wl,--wrap="$draft_wave" \
    -Wl,--wrap=_ZN6sglang7decoder15MtpDraftRequest5State9tp_winnerEv \
    -Wl,--wrap=_ZN6sglang7decoder15MtpDraftRequest13read_tp_tokenERS1_S2_ \
    -Wl,--start-group "${libraries[@]}" -Wl,--end-group \
    -L"$toolkit/lib" -L"$toolkit/lib/stubs" -Wl,-rpath,"$toolkit/lib" \
    -lcuda -lcudart "$nccl/lib/libnccl.so.2" \
    -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$nccl/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" \
    -licuuc -licudata -lpthread -ldl \
    -o "$output/serving-check" >"$output/native-build.log" 2>&1
sha256sum --quiet -c "$output/native-inputs.sha256"
sha256sum "$output/serving-check" >"$output/native-binary.sha256"
fi
if [[ $phase == build ]]; then
    exit 0
fi
if [[ $phase == all || $phase == cache || $phase == store || $phase == restart ]]; then
    sha256sum "$build/bin/sglang-server" "$checkpoint"/{config.json,model.safetensors.index.json} \
        tests/decoder/tp2/serving_cache.py tests/prefix/{pool_routes.py,lmcache_relay.py} \
        >"$output/cache-inputs.sha256"
fi

mapfile -t ordered < <(printf '%s\n' "${devices[@]}" | LC_ALL=C sort)
for gpu in "${ordered[@]}"; do
    [[ $gpu =~ ^GPU-[0-9a-fA-F-]{36}$ ]]
    for lock_dir in /tmp/gpu-coord "$shared/environments/vllm-81a8cb063e62"; do
        exec {lock}<"$lock_dir/$gpu.lock"
        flock -w 15 "$lock" || {
            printf 'GPU lease busy for %s in %s\n' "$gpu" "$lock_dir" >&2
            exit 1
        }
    done
done
for gpu in "${devices[@]}"; do
    clients=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
    read -r free utilization < <(nvidia-smi -i "$gpu" \
        --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
    if [[ -n $clients || $free -lt 175000 || $utilization -ne 0 ]]; then
        echo "GPU $gpu is busy" >&2
        exit 1
    fi
    nvidia-smi -i "$gpu" --query-gpu=index,uuid,name,driver_version --format=csv
done
visible=$(IFS=,; printf '%s' "${devices[*]}")
export CUDA_VISIBLE_DEVICES="$visible" CUDA_DEVICE_ORDER=PCI_BUS_ID
compat="$shared/software/cuda-compat-13.4/usr/local/cuda-13.4/compat"
export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
failed=0
if [[ $phase == all || $phase == static || $phase == tp1 ||
      $phase == tp2-mixed || $phase == tp2-fused ]]; then
cases=(tp1 tp2)
if [[ $phase == tp1 ]]; then cases=(tp1 mixed); fi
if [[ $phase == tp2-mixed ]]; then cases=(tp2-mixed); fi
if [[ $phase == tp2-fused ]]; then cases=(tp2-mixed tp2-graph); fi
for test_case in "${cases[@]}"; do
    tp=1
    variant=()
    if [[ $test_case == tp2* ]]; then tp=2; fi
    if [[ $test_case == *mixed ]]; then variant=(mixed); fi
    if [[ $test_case == tp2-graph ]]; then variant=(mixed-graph); fi
    timeout -k 5s 60s "$output/serving-check" "$checkpoint" "$tp" "$output/$test_case.tsv" \
        "${variant[@]}" >"$output/$test_case.log" 2>&1 || failed=1
done
python - "$output" "$phase" "${devices[@]}" -- "${cases[@]}" <<'PY' || failed=1
from pathlib import Path
import json
import re
import sys

root, phase = Path(sys.argv[1]), sys.argv[2]
separator = sys.argv.index("--")
devices, cases = sys.argv[3:separator], sys.argv[separator + 1:]
if phase != "tp1":
    log = (root / f'{cases[-1]}.log').read_text()
    for rank, gpu in enumerate(devices):
        match = re.search(rf'^rank={rank} device={rank} pid=\d+ .* uuid=([a-f0-9]+)$', log, re.M)
        assert match and match[1] == gpu[4:].replace('-', ''), (rank, gpu, log)
golden = json.loads(Path("tests/fixtures/short_fp8.json").read_bytes())["expected_output_ids"]
frames = []
for case in cases:
    rows = {}
    for row in (root / f"{case}.tsv").read_text().splitlines():
        label, lane, *tokens = row.split("\t")
        assert (label, lane) not in rows and len(tokens) == 8
        rows[label, lane] = list(map(int, tokens))
    assert len(rows) == 8, (case, rows)
    assert rows["serial_greedy", "0"] == golden, (case, rows["serial_greedy", "0"], golden)
    for slot in ("0", "1"):
        for mode in ("greedy", "sample"):
            assert rows[f"serial_{mode}", slot] == rows[f"batch_mtp4_{mode}", slot], (case, rows)
    assert rows["serial_greedy", "0"] != rows["serial_greedy", "1"], (case, rows)
    assert any(rows["serial_greedy", i] != rows["serial_sample", i] for i in ("0", "1"))
    frames.append(rows)
if phase == "tp1":
    assert frames[0]["serial_greedy", "1"] != frames[1]["serial_greedy", "1"], frames
    print("TP1 short and mixed MTP4 exactly matched greedy and seeded target-only oracles.")
elif phase == "tp2-mixed":
    print("TP2 mixed prefill and MTP4 exactly matched greedy and seeded target-only oracles.")
elif phase == "tp2-fused":
    assert frames[0] == frames[1], frames
    print("Normal TP2 fused, eager MTP4, and target-only generated identical token IDs.")
else:
    for slot in ("0", "1"):
        for case in ("serial_greedy", "batch_mtp4_greedy"):
            assert frames[0][case, slot] == frames[1][case, slot], (case, slot, frames)
        print("seeded sample", slot, "TP1", frames[0]["serial_sample", slot],
              "TP2", frames[1]["serial_sample", slot])
    print("Greedy IDs match across worlds. Every MTP4 row matches its same-world target-only row.")
PY
sha256sum --quiet -c "$output/native-inputs.sha256"
sha256sum --quiet -c "$output/native-binary.sha256"
fi
if [[ $phase == all || $phase == cache || $phase == store || $phase == restart ]]; then
options=()
if [[ $phase == store ]]; then options+=(--store-only); fi
if [[ $phase == restart ]]; then options+=(--restart-only); fi
timeout -k 35s 200s python tests/decoder/tp2/serving_cache.py "$build/bin/sglang-server" \
    "$checkpoint" "$output/cache" "${options[@]}" >"$output/cache.log" 2>&1 || failed=1
sha256sum --quiet -c "$output/cache-inputs.sha256"
fi
exit "$failed"
