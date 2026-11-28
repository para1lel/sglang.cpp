#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )); then
    printf '%s\n' 'Usage: run_graph_generalization.sh RELEASE SOURCE BUILD CHECKPOINT OUTPUT' \
        '  [baseline|build|trace [CASE ...]|CASE ...]' >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
release=$(realpath -- "$1")
candidate_source=$(realpath -- "$2")
candidate_build=$(realpath -- "$3")
checkpoint=$(realpath -- "$4")
output=$(realpath -m -- "$5")
shift 5
[[ $release != "$candidate_source" && "$release/build" != "$candidate_build" ]]
[[ -d $release/source/include && -d $candidate_source/include ]]
shared=${SGLANG_HOST_ROOT:-/data1/shared/dongwu.chen}
toolkit=${CONDA_PREFIX:?Activate the managed Pixi environment}/targets/x86_64-linux
cxx=${CXX:?Missing Pixi host compiler}
nccl=$(python -I -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl")')
compat="$shared/software/cuda-compat-13.4/usr/local/cuda-13.4/compat"
mkdir -- "$output"
cd -- "$root"

compile_arm() {
    local name=$1 source=$2 build=$3
    local libraries=("$build"/lib/libsglang*.a)
    sha256sum "${libraries[@]}" "$source/include/sglang/decoder.h" \
        "$source/include/sglang/decoder_batch.h" >"$output/$name-inputs-before.sha256"
    "$cxx" -std=c++23 -O2 -Wall -Wextra -Werror \
        -I"$source/include" -isystem "$toolkit/include" -isystem "$nccl/include" \
        tests/batch/fixtures/graph_generalization.cpp -Wl,--wrap=cudaIpcGetMemHandle \
        -Wl,--wrap=_ZN6sglang7decoder14TensorParallel25record_fused_graph_replayEv \
        -Wl,--wrap=_ZN6sglang7decoder14TensorParallel25record_fused_graph_replayEii \
        -Wl,--wrap=_ZN6sglang7decoder14TensorParallel25record_fused_graph_replayEiii \
        -Wl,--start-group "${libraries[@]}" -Wl,--end-group \
        -L"$toolkit/lib" -L"$toolkit/lib/stubs" -Wl,-rpath,"$toolkit/lib" \
        -lcuda -lcudart "$nccl/lib/libnccl.so.2" -Wl,-rpath,"$nccl/lib" \
        -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" -licuuc -licudata \
        -lpthread -ldl -o "$output/$name"
    sha256sum --quiet -c "$output/$name-inputs-before.sha256"
}

compile_arm baseline "$release/source" "$release/build" >"$output/baseline-build.log" 2>&1
if [[ ${1:-} == baseline ]]; then
    [[ $# == 1 ]]
    exit 0
fi
compile_arm candidate "$candidate_source" "$candidate_build" >"$output/candidate-build.log" 2>&1
"$cxx" -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -isystem "$toolkit/include" \
    tests/batch/graph_activity.cpp -L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib" -lcupti \
    -o "$output/activity.so" >"$output/observer-build.log" 2>&1
sha256sum tests/batch/{graph_activity.cpp,run_graph_generalization.sh} \
    tests/batch/fixtures/{graph_generalization.cpp,check_graph_generalization.py} \
    "$output"/{baseline,candidate,activity.so} "$release/official-release.sha256" \
    "$checkpoint/config.json" "$checkpoint/model.safetensors.index.json" >"$output/inputs.sha256"
if [[ ${1:-} == build ]]; then
    [[ $# == 1 ]]
    exit 0
fi

validator=()
if [[ ${1:-} == trace ]]; then
    validator=(--trace-only)
    shift
fi
first=${GPU_UUID_0:?Set GPU_UUID_0 to the authorized idle B200 UUID}
second=${GPU_UUID_1:-}
if (( !$# )); then
    : "${second:?Set GPU_UUID_1 for the default mixed TP1 and TP2 matrix}"
else
    for selection in "$@"; do
        if [[ $selection == '2 '* ]]; then
            : "${second:?Set GPU_UUID_1 for TP2 cases}"
        fi
    done
fi
[[ $first != "$second" ]]
devices=("$first")
if [[ $second ]]; then devices+=("$second"); fi
mapfile -t ordered < <(printf '%s\n' "${devices[@]}" | LC_ALL=C sort)
for gpu in "${ordered[@]}"; do
    [[ $gpu =~ ^GPU-[0-9a-fA-F-]{36}$ ]]
    for directory in /tmp/gpu-coord "$shared/environments/vllm-81a8cb063e62"; do
        exec {lock_fd}>>"$directory/$gpu.lock"
        flock -n "$lock_fd"
    done
done

check_idle() {
    local gpu=$1 memory utilization clients
    clients=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
    read -r memory utilization < <(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ',')
    if [[ -n $clients ]] || (( memory < 175000 || utilization != 0 )); then
        printf 'GPU %s is occupied (%s MiB free, %s%% utilization)\n' \
            "$gpu" "$memory" "$utilization" >&2
        return 1
    fi
}

snapshot() {
    local fields=uuid,name,driver_version,memory.used,memory.free,utilization.gpu,temperature.gpu
    nvidia-smi -i "${first}${second:+,$second}" --query-gpu="$fields" --format=csv >"$1"
}

check_idle "$first"
if [[ $second ]]; then check_idle "$second"; fi
snapshot "$output/device-before.csv"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

matrix=(
    '1 1 0 greedy edge' '1 2 0 sample short' '1 3 0 sample mixed' '1 4 0 greedy max'
    '1 1 0 greedy split' '1 3 0 sample split'
    '1 7 0 greedy short' '1 8 0 sample short' '1 9 0 greedy short' '1 16 0 sample short'
    '1 32 0 sample mixed-max'
    '1 1 1 greedy edge' '1 2 4 sample mixed' '1 3 4 greedy edge' '1 4 4 sample short'
    '1 7 4 greedy short' '1 8 4 sample short' '1 9 4 greedy short' '1 16 4 sample short'
    '2 1 0 greedy edge' '2 2 0 sample short' '2 1 4 greedy max' '2 2 4 sample mixed'
    '2 3 4 sample edge' '2 4 4 greedy short' '2 7 4 greedy short' '2 8 4 sample short'
    '2 9 4 greedy short' '2 16 4 sample short'
)
if (( $# )); then matrix=("$@"); fi

run_arm() {
    local version=$1 route=$2 kind=$3 iteration=${4:-0} file environment=()
    if [[ $kind == trace ]]; then
        file="$directory/$version-$route-trace"
        environment=("LD_PRELOAD=$output/activity.so"
                     "BATCH_GRAPH_ACTIVITY=$directory/$version-$route-cupti.tsv"
                     BATCH_GRAPH_NODE_ACTIVITY=1)
    else
        file="$directory/$version-$route-timing-$iteration"
    fi
    env -u LD_PRELOAD -u BATCH_GRAPH_ACTIVITY -u BATCH_GRAPH_NODE_ACTIVITY \
        timeout -k 5s 180s env "${environment[@]}" "$output/$version" "$checkpoint" "$world" \
        "$width" "$depth" "$route" "$sampling" "$shape" >"$file.tsv" 2>"$file.log"
}

failed=0
for selection in "${matrix[@]}"; do
    read -r world width depth sampling shape extra <<<"$selection"
    [[ $world =~ ^[12]$ && $width =~ ^[1-9][0-9]?$ && $depth =~ ^[014]$ && ! $extra ]]
    [[ $sampling =~ ^(greedy|sample)$ && $shape =~ ^(short|split|mixed|edge|max|mixed-max)$ ]]
    directory="$output/tp$world-w$width-k$depth-$sampling-$shape"
    mkdir -- "$directory"
    check_idle "$first"
    export CUDA_VISIBLE_DEVICES="$first"
    if [[ $world == 2 ]]; then
        check_idle "$second"
        export CUDA_VISIBLE_DEVICES="$first,$second"
    fi
    snapshot "$directory/device-before.csv"
    for version in baseline candidate; do
        for route in eager graph; do
            run_arm "$version" "$route" trace || failed=1
        done
    done
    if (( !${#validator[@]} )); then
        for repetition in 1 2 3; do
            if (( repetition % 2 )); then
                arms=(baseline:eager candidate:graph baseline:graph candidate:eager)
            else
                arms=(candidate:eager baseline:graph candidate:graph baseline:eager)
            fi
            for arm in "${arms[@]}"; do
                run_arm "${arm%:*}" "${arm#*:}" timing "$repetition" || failed=1
            done
        done
    fi
    snapshot "$directory/device-after.csv"
    python -IB tests/batch/fixtures/check_graph_generalization.py "$directory" "${validator[@]}" \
        >"$directory/report.json" 2>"$directory/rejected.log" || failed=1
done
snapshot "$output/device-after.csv"
sha256sum --quiet -c "$output/inputs.sha256"
sha256sum --quiet -c "$output/baseline-inputs-before.sha256"
sha256sum --quiet -c "$output/candidate-inputs-before.sha256"
exit "$failed"
