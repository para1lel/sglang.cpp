#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
    echo "Usage: bash tests/text/run_logit_bias_check.sh BUILD CHECKPOINT GPU_UUID OUTPUT" >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
build=$(realpath -- "$1")
checkpoint=$(realpath -- "$2")
gpu=$3
output=$(realpath -m -- "$4")
shared=${SGLANG_HOST_ROOT:-/data1/shared/dongwu.chen}
toolkit="${CONDA_PREFIX:?Run in the managed Pixi environment}/targets/x86_64-linux"
mkdir -p "$output/cache"
export CUDA_CACHE_PATH="$output/cache" TMPDIR="$output"

"${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror \
    -I"$root/include" -isystem "$toolkit/include" \
    "$root/tests/text/logit_bias_check.cpp" -Wl,--start-group \
    "$build/lib/libsglang_decoder.a" "$build/lib/libsglang_cuda.a" \
    "$build/lib/libsglang_prefill_fp8_sm100.a" "$build/lib/libsglang_checkpoint.a" \
    "$build/lib/libsglang_tokenizer.a" "$build/lib/libsglang_json.a" -Wl,--end-group \
    -L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" \
    -lcudart -lpthread -ldl -o "$output/logit-bias-check"

exec 8<"/tmp/gpu-coord/$gpu.lock"
exec 9<"$shared/environments/vllm-81a8cb063e62/$gpu.lock"
flock -w 15 8
flock -w 15 9
clients=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ -n $clients || $memory != 0 ]]; then
    echo "GPU $gpu is busy" >&2
    exit 1
fi
nvidia-smi -i "$gpu" --query-gpu=index,name,uuid,compute_cap,driver_version --format=csv
export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID
compat="$shared/software/cuda-compat-13.4/usr/local/cuda-13.4/compat"
export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset SGLANG_PREFILL_ATTN SGLANG_EXPERIMENTAL_TMEM_PREFILL
timeout -k 5s 120s "$output/logit-bias-check" "$checkpoint"
