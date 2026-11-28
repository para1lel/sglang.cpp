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
    singleton) ;;
    k1) test -x "$build/bin/sglang-server" ;;
    *) exit 2 ;;
esac
if [[ "$mode" != admission ]]; then
    test -x "$build/bin/batch-graph-lanes"
fi
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
mkdir "$out"
python3 -IB "$tests/check_graph_evidence.py" --check-slot-mask >"$out/slot-mask.log"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu}" CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,memory.free --format=csv \
    >"$out/device-before.csv"

if [[ "$mode" == all ]]; then
    "$build/bin/batch-graph-lanes" "$checkpoint" verify-wide >"$out/wide-verify.log" 2>&1
    "$build/bin/batch-graph-lanes" "$checkpoint" graph wide >"$out/wide-eager.log" 2>&1
fi
if [[ "$mode" == all || "$mode" == admission ]]; then
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

run_lanes() {
    local name=$1 role=$2 fault=${3:-} addon=${4:-} scenario=${5:-}
    local preload="$out/activity.so:$out/scores.so"
    if [[ -n "$addon" ]]; then
        preload="$preload:$out/$addon.so"
    fi
    LD_PRELOAD="$preload" BATCH_SCORE_OUTPUT="$out/$name.scores" \
        BATCH_GRAPH_ACTIVITY="$out/$name.cupti.tsv" BATCH_GRAPH_FAILURE="$fault" \
        "$build/bin/batch-graph-lanes" "$checkpoint" "$role" ${scenario:+"$scenario"} \
        >"$out/$name.lanes" 2>"$out/$name.log"
}

run_singleton() {
    run_lanes singleton-eager eager '' '' singleton
    run_lanes singleton-graph graph '' '' singleton
    run_lanes singleton-launch launch launch failure singleton
}

run_k1() {
    for role in eager graph prepare begin instantiate upload launch submitted \
                hard hard-drain cleanup exec-cleanup; do
        case "$role" in
            eager|graph) run_lanes "k1-$role" "$role" '' '' verify ;;
            *) run_lanes "k1-$role" "$role" "$role" failure verify ;;
        esac
    done
    python3 -IB "$tests/../mtp/graph_http.py" --server "$build/bin/sglang-server" \
        --checkpoint "$checkpoint" --evidence "$out/k1-http" --activity "$out/activity.so"
}

finish_device() {
    test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
    nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,memory.free --format=csv \
        >"$out/device-after.csv"
}

if [[ "$mode" == singleton ]]; then
    compile_observer failure -ldl
    run_singleton
    finish_device
    python3 -IB "$tests/check_graph_evidence.py" "$out" --singleton-only
    exit
fi

if [[ "$mode" == k1 ]]; then
    compile_observer failure -ldl
    run_k1
    finish_device
    python3 -IB "$tests/check_graph_evidence.py" "$out" --k1-only
    exit
fi

compile_observer oom -ldl
compile_observer failure -ldl
LD_PRELOAD="$out/activity.so" BATCH_GRAPH_ACTIVITY="$out/operators.cupti.tsv" \
    "$build/bin/mxfp8-gqa-check" >"$out/operators.log" 2>&1 ||
    echo "mxfp8-gqa-check exited $?" | tee "$out/operators.failure" >&2

for route in eager graph; do
    run_lanes "default-$route" "$route"
done
run_lanes fault-eager eager '' '' fault
for fault in begin instantiate launch scratch-oom instantiate-oom hard; do
    route=graph
    observer=failure
    case "$fault" in
        hard) route=hard ;;
        *-oom) observer=oom ;;
    esac
    run_lanes "fault-$fault" "$route" "$fault" "$observer" fault
done

for route in eager graph; do
    python3 -IB "$tests/test_batch_graph.py" 0 "$out/http-default-$route" 4 "$route" \
        --binary "$build/bin/sglang-server" --checkpoint "$checkpoint" \
        --activity "$out/activity.so"
done

run_singleton
run_k1
finish_device
python3 -IB "$tests/check_graph_evidence.py" "$out"
