#!/usr/bin/env bash
set -euo pipefail

mode=${4:-}
if (( $# < 3 || $# > 4 )) || [[ $1 == "$2" ||
    ( $# == 4 && $mode != --require-mnnvl && $mode != --require-nccl ) ]]; then
    echo "Usage: bash $0 GPU_UUID GPU_UUID TP2_COLLECTIVE_CHECK" \
        "[--require-mnnvl|--require-nccl]" >&2
    exit 2
fi
first=$1
second=$2
binary=$(realpath -m -- "$3")
if [[ ! -f "$binary" || ! -x "$binary" ]]; then
    echo "Pass the executable built by the XMake tp2-collective-check target" >&2
    exit 2
fi
shared=${SGLANG_HOST_ROOT:-/data1/shared/dongwu.chen}

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
timeout -k 5s 45s "$binary" "${@:4}"
