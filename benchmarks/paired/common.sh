#!/usr/bin/env bash

paired_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project=$(cd -- "$paired_dir/../.." && pwd -P)
protocol=$(realpath -- "${PAIRED_PROTOCOL_ROOT:-$project}")
host_root=${PAIRED_HOST_ROOT:-/data1/shared/dongwu.chen}
checkpoint=${PAIRED_CHECKPOINT:-/data0/shared/Qwen3.8-27B-FP8}
vllm_root=${PAIRED_VLLM_ROOT:-$host_root/environments/vllm-81a8cb063e62}
compat=${PAIRED_CUDA_COMPAT:-$host_root/software/cuda-compat-13.4/usr/local/cuda-13.4/compat}
client_python=${PAIRED_CLIENT_PYTHON:-/usr/bin/python3}
warmups=${PAIRED_WARMUPS:-1}
measured=${PAIRED_MEASURED:-1}
token_budget=${PAIRED_MAX_NUM_BATCHED_TOKENS-8192}
server_cpus=${PAIRED_SERVER_CPUS:-24-33}
client_cpus=${PAIRED_CLIENT_CPUS:-34-35}
port=${PAIRED_PORT:-17393}
url=http://127.0.0.1:$port
gpu_fields=timestamp,index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.current.sm
gpu_fields+=,clocks.current.memory,power.draw
server_pid=
monitor_pid=

validate_run_settings() {
    if [[ $warmups != 1 || $measured != 1 ]]; then
        echo "Use exactly one warmup and one measured cohort." >&2
        return 2
    fi
    if [[ ! $token_budget =~ ^[1-9][0-9]{0,4}$ ]] || (( token_budget > 32768 )); then
        echo "Use PAIRED_MAX_NUM_BATCHED_TOKENS in [1, 32768]." >&2
        return 2
    fi
}

write_command() {
    local destination=$1
    shift
    printf '%q ' "$@" >"$destination"
    printf '\n' >>"$destination"
}

record_compute() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader >"$1"
}

check_idle() {
    local active
    active=$(nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid --format=csv,noheader)
    if [[ -n $active ]]; then
        echo "GPU $gpu_uuid is occupied by $active" >&2
        return 75
    fi
}

comparison_identity() {
    PYTHONPATH=$protocol "$client_python" -B "$paired_dir/identity.py" "$protocol" "$output" "$@"
}

plan_comparison() {
    local mode=$1
    shift
    local manifest=${PAIRED_CHECKPOINT_MANIFEST:-}
    local options=()
    if [[ $mode == matched ]]; then
        local saved
        saved=$(realpath -- "$PAIRED_VLLM_RESULTS")
        options+=(--saved "$saved")
        if [[ -z $manifest && -f $saved/checkpoint.sha256 ]]; then
            manifest=$saved/checkpoint.sha256
        elif [[ -z $manifest ]]; then
            manifest=$saved/meta/checkpoint.sha256
        fi
        if [[ ${PAIRED_ALLOW_COUNT_MISMATCH:-0} == 1 ]]; then
            options+=(--allow-count-mismatch)
        fi
    fi
    if [[ ! -f $manifest ]]; then
        echo "Provide the published model file list as PAIRED_CHECKPOINT_MANIFEST." >&2
        return 2
    fi
    cp -- "$manifest" "$output/checkpoint.sha256"
    local plan
    plan=$(comparison_identity plan "$mode" "$checkpoint" "$output/checkpoint.sha256" \
        "${fixtures:-$output}" "$warmups" "$measured" "$@" \
        --max-num-batched-tokens "$token_budget" "${options[@]}")
    comparison_plan_sha256=$(printf '%s\n' "$plan" | sha256sum)
    comparison_plan_sha256=${comparison_plan_sha256%% *}
    printf '%s\n' "$plan" >"$output/comparison-plan.tmp"
    mv -- "$output/comparison-plan.tmp" "$output/comparison-plan.json"
}

prepare_run() {
    local index=$1
    local reservation_dir=$2
    [[ $index =~ ^[0-9]+$ ]]
    gpu_uuid=$(nvidia-smi -i "$index" --query-gpu=uuid --format=csv,noheader)
    [[ $gpu_uuid =~ ^GPU-[0-9a-fA-F-]{36}$ ]]
    exec 9>>"$vllm_root/$gpu_uuid.lock"
    flock -n 9 || { echo "GPU reservation busy for $gpu_uuid" >&2; return 75; }
    check_idle
    mkdir -- "$reservation_dir"
    output=$(realpath -- "$reservation_dir")
    date -u +%FT%TZ >"$output/started-utc.txt"
    printf '%s\n' "$index $gpu_uuid" >"$output/gpu.txt"
    printf 'server=%s\nclient=%s\nmemory=localalloc\n' "$server_cpus" "$client_cpus" \
        >"$output/cpus.txt"
    uname -a >"$output/host.txt"
    LC_ALL=C lscpu >>"$output/host.txt"
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,power.limit \
        --format=csv >"$output/devices.csv"
    nvidia-smi topo -m >"$output/topology.txt"
    sha256sum "$paired_dir"/{common.sh,measure_concurrent.py,makespan.py} \
        "$protocol"/benchmarks/streaming/{client.py,W1.json,W2.json} \
        "$protocol"/benchmarks/concurrent/{client.py,reference.py,freeze_prompts.py} \
        "$protocol"/benchmarks/concurrent/{fixture-manifest.json,gpu2-reference.json} \
        >"$output/protocol.sha256"
}

prepare_fixtures() {
    if [[ -n ${PAIRED_FIXTURES:-} ]]; then
        fixtures=$(realpath -- "$PAIRED_FIXTURES")
    else
        fixtures=$output/fixtures
        (
            cd -- "$protocol"
            taskset --cpu-list "$client_cpus" "$vllm_root/venv/bin/python" -B \
                -m benchmarks.concurrent.freeze_prompts "$checkpoint/tokenizer.json" "$fixtures"
        )
    fi
}

stop_group() {
    if [[ -n $monitor_pid ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=
    fi
    if [[ -n $server_pid ]]; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for ((attempt = 0; attempt < 40; ++attempt)); do
            kill -0 -- "-$server_pid" 2>/dev/null || break
            sleep .25
        done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
        server_pid=
    fi
    if [[ -n ${server_launcher_pid:-} ]]; then
        wait "$server_launcher_pid" 2>/dev/null || true
        server_launcher_pid=
    fi
}

start_group() {
    local run=$1
    shift
    if [[ -z ${lmcache_pid:-} ]]; then
        check_idle || return
    fi
    record_compute "$run/compute-before.csv" || return
    if [[ -n ${lmcache_pid:-} ]]; then
        check_compute_owners "$run/compute-before.csv" "$lmcache_pid" || return
    fi
    if curl --noproxy '*' -fsS --max-time 1 "$url/health" >/dev/null 2>&1; then
        echo "A server already responds on $url" >&2
        return 1
    fi
    local command=(numactl --localalloc taskset --cpu-list "$server_cpus" "$@")
    write_command "$run/server-command.sh" "${command[@]}"
    nvidia-smi --query-gpu="$gpu_fields" --format=csv -l 5 >"$run/gpu-samples.csv" 9<&- &
    monitor_pid=$!
    date -u +%FT%TZ >"$run/server-started-utc.txt"
    "$client_python" -IB -c '
import subprocess
import sys
from pathlib import Path

server = subprocess.Popen(sys.argv[2:], start_new_session=True)
Path(sys.argv[1]).write_text(f"{server.pid}\n")
sys.exit(server.wait())
' "$run/server-pid.txt" "${command[@]}" >"$run/server.log" 2>&1 9<&- &
    server_launcher_pid=$!
    for ((attempt = 0; attempt < 100; ++attempt)); do
        [[ ! -s $run/server-pid.txt ]] || break
        kill -0 "$server_launcher_pid" 2>/dev/null || { tail -40 "$run/server.log" >&2; return 1; }
        sleep .05
    done
    read -r server_pid <"$run/server-pid.txt" || return
    for ((attempt = 0; attempt < 480; ++attempt)); do
        kill -0 "$server_pid" 2>/dev/null || { tail -40 "$run/server.log" >&2; return 1; }
        if curl --noproxy '*' -fsS --max-time 1 "$url/health" >"$run/health.json" 2>/dev/null; then
            record_compute "$run/compute-ready.csv" || return
            date -u +%FT%TZ >"$run/server-ready-utc.txt"
            return 0
        fi
        sleep 1
    done
    echo "Server health timed out; see $run/server.log" >&2
    return 1
}

measure_group() {
    local group=$1
    local run=$2
    local arm=${3:-vllm}
    local collector=makespan.py
    if [[ ${mode:-} == frozen ]]; then
        collector=measure_concurrent.py
    fi
    local command=(env "PYTHONPATH=$protocol" numactl --localalloc
        taskset --cpu-list "$client_cpus" "$client_python" -B)
    if [[ $group == W1-W2 ]]; then
        command+=("$protocol/benchmarks/streaming/client.py" --url "$url"
            --output "$run/measurement")
        if [[ $arm == vllm ]]; then
            command+=(--freeze-baseline)
        fi
    else
        command+=("$paired_dir/$collector" "$group" "$url" "$fixtures" "$run/measurement"
            --warmups "$warmups" --measured "$measured")
    fi
    write_command "$run/client-command.sh" "${command[@]}"
    date -u +%FT%TZ >"$run/measurement-started-utc.txt"
    local status=0
    "${command[@]}" >"$run/client.log" 2>&1 9<&- || status=$?
    date -u +%FT%TZ >"$run/measurement-finished-utc.txt"
    printf '%s\n' "$status" >"$run/client-exit-code.txt"
    record_compute "$run/compute-measurement-end.csv" || return
    return "$status"
}
