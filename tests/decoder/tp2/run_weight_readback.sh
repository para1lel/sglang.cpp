#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo 'Usage: bash run_weight_readback.sh CHECKPOINT GPU_UUID OUTPUT_DIRECTORY' >&2
    exit 2
fi
checkpoint=$(realpath -e -- "$1")
gpu=$2
out=$(realpath -m -- "$3")
[[ $gpu =~ ^GPU-[0-9a-fA-F-]{36}$ ]] || { echo 'Expected a physical GPU UUID' >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
shared=${SGLANG_HOST_ROOT:-/data1/shared/dongwu.chen}
kit=${CONDA_PREFIX:?Run in the default managed Pixi environment}/targets/x86_64-linux
mkdir -p -- "$out"
cd -- "$root"
sources=(
    tests/decoder/tp2/weight_readback_check.cpp src/checkpoint.cpp src/json.cpp src/utf8.cpp
    src/runtime/model/{qwen_checkpoint,decoder_weights}.cpp
    src/runtime/backend/{cuda_resources,cuda_backend_policy,retired_request_blocks}.cpp
    src/runtime/execution/decoder_execution_policy.cpp
)
"${CXX:?Missing Pixi C++ compiler}" -std=c++17 -O2 -ffunction-sections -fdata-sections \
    -Iinclude -isystem "$kit/include" "${sources[@]}" -Wl,--gc-sections \
    -L"$kit/lib" -Wl,-rpath,"$kit/lib" -lcudart -pthread -o "$out/weight-readback-check"
sha256sum "${sources[@]}" include/sglang/runtime/model/qwen_{weight_shard,weight_upload,layers}.h \
    "$out/weight-readback-check" "$checkpoint"/{config.json,model.safetensors.index.json}
exec 8<"/tmp/gpu-coord/$gpu.lock"
flock -n 8
exec 9<"$shared/environments/vllm-81a8cb063e62/$gpu.lock"
flock -n 9
clients=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
read -r free busy < <(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ,)
[[ -z $clients && $free -ge 175000 && $busy -eq 0 ]] || { echo 'GPU is busy' >&2; exit 1; }
nvidia-smi -i "$gpu" --query-gpu=index,uuid,name,driver_version,memory.free --format=csv
export CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID
compat="$shared/software/cuda-compat-13.4/usr/local/cuda-13.4/compat"
export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for world_rank in '2 1' '2 0' '1 0'; do
    read -r world rank <<< "$world_rank"
    "$out/weight-readback-check" "$checkpoint" "$world" "$rank"
done
