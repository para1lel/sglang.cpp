#!/usr/bin/env bash

paired_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project=$(cd -- "$paired_dir/../.." && pwd -P)
protocol=$(realpath -- "${PAIRED_PROTOCOL_ROOT:-$project}")
host_root=${PAIRED_HOST_ROOT:-/data1/shared/dongwu.chen}
checkpoint=${PAIRED_CHECKPOINT:-/data0/shared/Qwen3.8-27B-FP8}
vllm_root=${PAIRED_VLLM_ROOT:-$host_root/environments/vllm-81a8cb063e62}
compat=${PAIRED_CUDA_COMPAT:-$host_root/software/cuda-compat-13.4/usr/local/cuda-13.4/compat}
client_python=${PAIRED_CLIENT_PYTHON:-/usr/bin/python3}
warmups=${PAIRED_WARMUPS:-2}
measured=${PAIRED_MEASURED:-5}
server_cpus=${PAIRED_SERVER_CPUS:-24-33}
client_cpus=${PAIRED_CLIENT_CPUS:-34-35}
port=${PAIRED_PORT:-17393}
url=http://127.0.0.1:$port
gpu_fields=timestamp,index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.current.sm
gpu_fields+=,clocks.current.memory,power.draw
server_pid=
monitor_pid=

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

check_saved_counts() {
    local group=$1
    local summary=$PAIRED_VLLM_RESULTS/vllm-$group/$group/measurement/summary.json
    local expected="$warmups $measured"
    local actual
    if [[ $group == W1-W2 ]]; then
        expected='3 9'
        actual=$(jq -er 'select(.workloads.W1.valid and .workloads.W2.valid) |
            select(.workloads.W1.warmups_successful == .workloads.W2.warmups_successful
                and .workloads.W1.successful == .workloads.W2.successful) |
            [.workloads.W1.warmups_successful,.workloads.W1.successful] | join(" ")' \
            "$summary") || actual=
    else
        actual=$(jq -er 'select(.status == "complete") |
            [.request_outcomes[].tag] |
            [(map(select(startswith("warmup-"))) | length),
             (map(select(startswith("measured-"))) | length)] | join(" ")' \
            "$summary") || actual=
    fi
    if [[ -z $actual ]]; then
        echo "No complete vLLM evidence for $group in $summary." >&2
        return 1
    fi
    if [[ $actual != "$expected" ]]; then
        echo "$group saved vLLM warmup/measured counts: $actual. Requested: $expected." >&2
        if [[ ${PAIRED_ALLOW_COUNT_MISMATCH:-0} != 1 ]]; then
            return 2
        fi
        echo "Using the explicit PAIRED_ALLOW_COUNT_MISMATCH override." >&2
    fi
}

prepare_run() {
    local index=$1
    local directory=$2
    [[ $index =~ ^[0-9]+$ ]]
    gpu_uuid=$(nvidia-smi -i "$index" --query-gpu=uuid --format=csv,noheader)
    [[ $gpu_uuid =~ ^GPU-[0-9a-fA-F-]{36}$ ]]
    exec 9>>"$vllm_root/$gpu_uuid.lock"
    flock -n 9 || { echo "GPU reservation busy for $gpu_uuid" >&2; return 75; }
    check_idle
    mkdir -- "$directory"
    output=$(realpath -- "$directory")
    date -u +%FT%TZ >"$output/started-utc.txt"
    printf '%s\n' "$index $gpu_uuid" >"$output/gpu.txt"
    printf 'server=%s\nclient=%s\nmemory=localalloc\n' "$server_cpus" "$client_cpus" \
        >"$output/cpus.txt"
    uname -a >"$output/host.txt"
    LC_ALL=C lscpu >>"$output/host.txt"
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,power.limit \
        --format=csv >"$output/devices.csv"
    nvidia-smi topo -m >"$output/topology.txt"
    sha256sum "$paired_dir"/{common.sh,measure_concurrent.py} \
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
        wait "$server_pid" 2>/dev/null || true
        server_pid=
    fi
}

start_group() {
    local run=$1
    shift
    check_idle || return
    record_compute "$run/compute-before.csv" || return
    if curl --noproxy '*' -fsS --max-time 1 "$url/health" >/dev/null 2>&1; then
        echo "A server already responds on $url" >&2
        return 1
    fi
    local command=(numactl --localalloc taskset --cpu-list "$server_cpus" "$@")
    write_command "$run/server-command.sh" "${command[@]}"
    nvidia-smi --query-gpu="$gpu_fields" --format=csv -l 5 >"$run/gpu-samples.csv" 9<&- &
    monitor_pid=$!
    date -u +%FT%TZ >"$run/server-started-utc.txt"
    setsid "${command[@]}" >"$run/server.log" 2>&1 9<&- &
    server_pid=$!
    printf '%s\n' "$server_pid" >"$run/server-pid.txt"
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
    local command=(env "PYTHONPATH=$protocol" numactl --localalloc
        taskset --cpu-list "$client_cpus" "$client_python" -B)
    if [[ $group == W1-W2 ]]; then
        command+=("$protocol/benchmarks/streaming/client.py" --url "$url"
            --output "$run/measurement")
        if [[ $arm == vllm ]]; then
            command+=(--freeze-baseline)
        fi
    else
        command+=("$paired_dir/measure_concurrent.py" "$group" "$url" "$fixtures" "$run/measurement"
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
