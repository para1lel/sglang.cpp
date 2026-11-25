#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 || $# > 6 )) || [[ $3 == "$4" ]]; then
    echo 'Usage: serving_check.sh BUILD CHECKPOINT UUID0 UUID1 OUTPUT [all|static|cache|store]' >&2
    exit 2
fi
build=$(realpath -- "$1")
checkpoint=$(realpath -- "$2")
first=$3 second=$4
output=$(realpath -m -- "$5")
phase=${6:-all}
[[ $phase == all || $phase == static || $phase == cache || $phase == store ]]
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
shared=${SGLANG_HOST_ROOT:-/data1/shared/dongwu.chen}
toolkit=${CONDA_PREFIX:?Run in the managed Pixi environment}/targets/x86_64-linux
nccl=$(python -I -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl")')
mkdir -p -- "$output"
cd -- "$root"
if [[ $phase == all || $phase == static ]]; then
"${CXX:?Missing Pixi host compiler}" -std=c++23 -O2 -Wall -Wextra -Werror \
    -Iinclude -isystem "$toolkit/include" -isystem "$nccl/include" \
    tests/decoder/tp2/serving_check.cpp tests/decoder/tp2/serving_probe.cpp \
    tests/decoder/tp2/prefill_plan_check.cpp \
    -Wl,--wrap=ncclAllReduce -Wl,--wrap=cudaMemcpyAsync \
    -Wl,--start-group "$build"/lib/libsglang*.a -Wl,--end-group \
    -L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib" -lcudart "$nccl/lib/libnccl.so.2" \
    -Wl,-rpath,"$nccl/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" -lpthread -ldl \
    -o "$output/serving-check"
fi

mapfile -t ordered < <(printf '%s\n' "$first" "$second" | LC_ALL=C sort)
for gpu in "${ordered[@]}"; do
    [[ $gpu =~ ^GPU-[0-9a-fA-F-]{36}$ ]]
    for lock_dir in /tmp/gpu-coord "$shared/environments/vllm-81a8cb063e62"; do
        exec {lock}<"$lock_dir/$gpu.lock"
        flock -w 15 "$lock"
    done
done
for gpu in "$first" "$second"; do
    clients=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
    read -r free utilization < <(nvidia-smi -i "$gpu" \
        --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
    if [[ -n $clients || $free -lt 175000 || $utilization -ne 0 ]]; then
        echo "GPU $gpu is busy" >&2
        exit 1
    fi
    nvidia-smi -i "$gpu" --query-gpu=index,uuid,name,driver_version --format=csv
done
export CUDA_VISIBLE_DEVICES="$first,$second" CUDA_DEVICE_ORDER=PCI_BUS_ID
compat="$shared/software/cuda-compat-13.4/usr/local/cuda-13.4/compat"
export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
failed=0
if [[ $phase == all || $phase == static ]]; then
for tp in 1 2; do
    timeout -k 5s 180s "$output/serving-check" "$checkpoint" "$tp" "$output/tp$tp.tsv" \
        >"$output/tp$tp.log" 2>&1 || failed=1
done
python - "$output" "$first" "$second" <<'PY' || failed=1
from pathlib import Path
import json
import re
import sys
log = (Path(sys.argv[1]) / 'tp2.log').read_text()
for rank, gpu in enumerate(sys.argv[2:]):
    match = re.search(rf'^rank={rank} device={rank} pid=\d+ .* uuid=([a-f0-9]+)$', log, re.M)
    assert match and match[1] == gpu[4:].replace('-', ''), (rank, gpu, log)
golden = json.loads(Path("benchmarks/vllm/small-fp8-result.json").read_bytes())["output_ids"]
frames = []
for tp in (1, 2):
    rows = {}
    for row in (Path(sys.argv[1]) / f"tp{tp}.tsv").read_text().splitlines():
        case, lane, *tokens = row.split("\t")
        assert (case, lane) not in rows and len(tokens) == 8
        rows[case, lane] = list(map(int, tokens))
    assert len(rows) == 8, (tp, rows)
    assert rows["serial_greedy", "0"] == golden, (tp, rows["serial_greedy", "0"], golden)
    for slot in ("0", "1"):
        for mode in ("greedy", "sample"):
            assert rows[f"serial_{mode}", slot] == rows[f"batch_mtp4_{mode}", slot], (tp, rows)
    assert rows["serial_greedy", "0"] != rows["serial_greedy", "1"], (tp, rows)
    assert any(rows["serial_greedy", i] != rows["serial_sample", i] for i in ("0", "1"))
    frames.append(rows)
for slot in ("0", "1"):
    for case in ("serial_greedy", "batch_mtp4_greedy"):
        assert frames[0][case, slot] == frames[1][case, slot], (case, slot, frames)
    print("seeded sample", slot, "TP1", frames[0]["serial_sample", slot],
          "TP2", frames[1]["serial_sample", slot])
print("Greedy IDs match across worlds. Every MTP4 row matches its same-world target-only row.")
PY
fi
if [[ $phase != static ]]; then
options=()
if [[ $phase == store ]]; then options+=(--store-only); fi
timeout -k 35s 200s python tests/decoder/tp2/serving_cache.py "$build/bin/sglang-server" \
    "$checkpoint" "$output/cache" "${options[@]}" >"$output/cache.log" 2>&1 || failed=1
fi
exit "$failed"
