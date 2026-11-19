#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$root"
tests="$root/tests/batch"
build=$(realpath -- "${1:?Pass the Release build directory}")
checkpoint=$(realpath -- "${2:?Pass the checkpoint directory}")
out=$(realpath -m -- "${3:?Pass a fresh evidence directory}")
gpu=${GPU_UUID:?Set GPU_UUID to the reserved B200 device UUID}
toolkit="${CONDA_PREFIX:?Activate the managed Pixi default environment}/targets/x86_64-linux"
cxx=${CXX:-g++}
mode=${4:-all}

case "$mode" in
    all)
        test -x "$build/bin/sglang-server"
        test -x "$build/bin/mxfp8-gqa-check"
        ;;
    admission) test -x "$build/bin/sglang-server" ;;
    boundary) ;;
    *) exit 2 ;;
esac
if [[ "$mode" != admission ]]; then
    test -x "$build/bin/batch-graph-lanes"
fi
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
mkdir "$out"
python3 -IB "$tests/check_graph_evidence.py" --check-slot-mask >"$out/slot-mask.log"
export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,memory.free --format=csv \
    >"$out/device-before.csv"

if [[ "$mode" == all ]]; then
    "$build/bin/batch-graph-lanes" "$checkpoint" verify-wide exact >"$out/wide-verify.log" 2>&1
fi
if [[ "$mode" != boundary ]]; then
    python3 -IB "$tests/run_admission.py" --binary "$build/bin/sglang-server" \
        --checkpoint "$checkpoint" --evidence "$out/admission"
fi
if [[ "$mode" == admission ]]; then
    test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
    nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,memory.free --format=csv \
        >"$out/device-after.csv"
    exit
fi

compile_observer() {
    local source=$1
    shift
    "$cxx" -std=c++17 -O2 -fPIC -shared -Iinclude -I"$toolkit/include" \
        "$tests/graph_$source.cpp" "$@" -o "$out/$source.so"
}

compile_observer activity -L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib" -lcupti
compile_observer scores -ldl
compile_observer oom -ldl

run_lanes() {
    local name=$1 role=$2 policy=$3 fault=${4:-} addon=${5:-} scenario=${6:-}
    local preload="$out/activity.so:$out/scores.so"
    if [[ -n "$addon" ]]; then
        preload="$preload:$out/$addon.so"
    fi
    LD_PRELOAD="$preload" BATCH_SCORE_OUTPUT="$out/$name.scores" \
        BATCH_GRAPH_ACTIVITY="$out/$name.cupti.tsv" BATCH_GRAPH_FAILURE="$fault" \
        "$build/bin/batch-graph-lanes" "$checkpoint" "$role" "$policy" ${scenario:+"$scenario"} \
        >"$out/$name.lanes" 2>"$out/$name.log"
}

run_boundary() {
    run_lanes boundary-eager eager tensor-core '' '' boundary
    run_lanes boundary-graph graph tensor-core '' '' boundary
    run_lanes boundary-score-oom score-oom tensor-core score-oom oom boundary
}

if [[ "$mode" == boundary ]]; then
    run_boundary
    test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
    nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,memory.free --format=csv \
        >"$out/device-after.csv"
    python3 -IB "$tests/check_graph_evidence.py" "$out" --boundary-only
    exit
fi

compile_observer failure -ldl
LD_PRELOAD="$out/activity.so" BATCH_GRAPH_ACTIVITY="$out/operators.cupti.tsv" \
    "$build/bin/mxfp8-gqa-check" >"$out/operators.log" 2>&1

for policy in exact tensor-core; do
    for route in eager graph; do
        run_lanes "$policy-$route" "$route" "$policy"
    done
done
for fault in begin instantiate launch scratch-oom instantiate-oom hard; do
    route=graph
    observer=failure
    case "$fault" in
        hard) route=hard ;;
        *-oom) observer=oom ;;
    esac
    run_lanes "fault-$fault" "$route" exact "$fault" "$observer"
done

for policy in exact tensor-core; do
    for route in eager graph; do
        python3 -IB "$tests/test_batch_graph.py" 0 "$out/http-$policy-$route" 4 "$route" \
            "$policy" --binary "$build/bin/sglang-server" --checkpoint "$checkpoint" \
            --activity "$out/activity.so"
    done
done

run_boundary
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,memory.free --format=csv \
    >"$out/device-after.csv"
python3 -IB "$tests/check_graph_evidence.py" "$out"
