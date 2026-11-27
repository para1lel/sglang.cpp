#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )) || [[ $1 == "$2" || ( $# == 4 && $4 != --require-mnnvl ) ]]; then
    echo "Usage: bash $0 GPU_UUID GPU_UUID OUTPUT [--require-mnnvl]" >&2
    exit 2
fi
first=$1
second=$2
out=$(realpath -m -- "$3")
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
shared=${SGLANG_HOST_ROOT:-/data1/shared/dongwu.chen}
toolkit=${CONDA_PREFIX:?Run in the managed Pixi environment}/targets/x86_64-linux
nccl=$(python -I -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl")')
mkdir -p -- "$out"
cd -- "$root"
vendor_objects=()
vendor_flags=()
options=()
if [[ ${4:-} == --require-mnnvl ]]; then
    baseline="$shared/environments/vllm-81a8cb063e62/venv/lib/python3.12/site-packages"
    flash=${SGLANG_FLASHINFER_ROOT:-"$baseline/flashinfer"}/data
    nvcc -ccbin "${CXX:?Missing Pixi host compiler}" -std=c++17 -O3 -DNDEBUG \
        -gencode arch=compute_100a,code=sm_100a --expt-relaxed-constexpr \
        -Iinclude -I"$toolkit/include" -I"$flash/include" -I"$flash/spdlog/include" \
        -c src/runtime/backend/mnnvl_ar.cu -o "$out/mnnvl_ar.o"
    vendor_objects=("$out/mnnvl_ar.o" -L"$toolkit/lib/stubs" -lcuda -pthread)
    vendor_flags=(-DSGLANG_MNNVL_AR=1)
    options=(--require-mnnvl)
fi
"${CXX:?Missing Pixi host compiler}" -std=c++17 -O2 -ffunction-sections -fdata-sections \
    "${vendor_flags[@]}" \
    -Iinclude -isystem "$toolkit/include" -isystem "$nccl/include" \
    tests/decoder/tp2/collective_check.cpp src/runtime/backend/tensor_parallel.cpp \
    src/runtime/backend/cuda_resources.cpp "${vendor_objects[@]}" -Wl,--gc-sections \
    -L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib" -lcudart \
    "$nccl/lib/libnccl.so.2" -Wl,-rpath,"$nccl/lib" -o "$out/collective-check"

mapfile -t ordered < <(printf '%s\n' "$first" "$second" | LC_ALL=C sort)
for gpu in "${ordered[@]}"; do
    if [[ ! $gpu =~ ^GPU-[0-9a-fA-F-]{36}$ ]]; then
        echo "Invalid physical GPU UUID: $gpu" >&2
        exit 2
    fi
    for lock_dir in /tmp/gpu-coord "$shared/environments/vllm-81a8cb063e62"; do
        exec {lock}<"$lock_dir/$gpu.lock"
        flock -w 15 "$lock"
    done
done
for gpu in "$first" "$second"; do
    clients=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
    read -r free utilization recovery < <(nvidia-smi -i "$gpu" \
        --query-gpu=memory.free,utilization.gpu,gpu_recovery_action \
        --format=csv,noheader,nounits | tr -d ,)
    if [[ -n $clients || $free -lt 175000 || $utilization -ne 0 || $recovery != None ]]; then
        echo "GPU $gpu is busy" >&2
        exit 1
    fi
    nvidia-smi -i "$gpu" --query-gpu=index,uuid,name,driver_version --format=csv
done
export CUDA_VISIBLE_DEVICES="$first,$second" CUDA_DEVICE_ORDER=PCI_BUS_ID
compat="$shared/software/cuda-compat-13.4/usr/local/cuda-13.4/compat"
export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
timeout -k 5s 45s "$out/collective-check" "${options[@]}"
