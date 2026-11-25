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

# Run in the background. The shell's $! stays the session leader across exec.
launch_session() {
    exec "$client_python" -IB -c '
import os
import sys
from pathlib import Path

os.setsid()
Path(sys.argv[1]).write_text(f"{os.getpid()}\n")
os.execvp(sys.argv[2], sys.argv[2:])
' "$@"
}

stop_session() {
    local pid=$1 attempt
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for ((attempt = 0; attempt < 40; ++attempt)); do
        kill -0 -- "-$pid" 2>/dev/null || break
        sleep .25
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

stop_group() {
    if [[ -n $monitor_pid ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=
    fi
    if [[ -n $server_pid ]]; then
        stop_session "$server_pid"
        server_pid=
    fi
}

server_preflight() {
    local run=$1
    if [[ -z ${lmcache_pid:-} ]]; then
        check_idle || return
    fi
    record_compute "$run/compute-before.csv" || return
    if [[ -n ${lmcache_pid:-} ]]; then
        check_compute_owners "$run/compute-before.csv" "$lmcache_pid" || return
    fi
}

start_group() {
    local run=$1
    shift
    server_preflight "$run" || return
    if curl --noproxy '*' -fsS --max-time 1 "$url/health" >/dev/null 2>&1; then
        echo "A server already responds on $url" >&2
        return 1
    fi
    local command=(numactl --localalloc taskset --cpu-list "$server_cpus" "$@")
    write_command "$run/server-command.sh" "${command[@]}"
    nvidia-smi --query-gpu="$gpu_fields" --format=csv -l 5 >"$run/gpu-samples.csv" 9<&- &
    monitor_pid=$!
    date -u +%FT%TZ >"$run/server-started-utc.txt"
    launch_session "$run/server-pid.txt" "${command[@]}" >"$run/server.log" 2>&1 9<&- &
    server_pid=$!
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

start_vllm() {
    local run=$1
    shift
    mkdir "$run"/{cache,config,tmp}
    cp -aL --reflink=auto "$seed_cache/." "$run/cache/"
    "${client[@]}" cubins "$run/cache" "$run"
    local -x VLLM_HAS_FLASHINFER_CUBIN=1 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1
    local -x OMP_NUM_THREADS=2 MAX_JOBS=2
    local -x XDG_CACHE_HOME="$run/cache" XDG_CONFIG_HOME="$run/config" TMPDIR="$run/tmp"
    local -x VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm" VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
    local -x HF_HOME="$XDG_CACHE_HOME/huggingface" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
    local -x TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME"
    local -x VLLM_RPC_BASE_PATH=tmp VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_SERVER_DEV_MODE=1
    local -x CUDA_HOME=/usr/local/cuda-13.2
    local -x PATH="$vllm_root/venv/bin:$CUDA_HOME/bin:$PATH"
    local -x LD_LIBRARY_PATH="$compat" CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONDONTWRITEBYTECODE=1
    local -x PYTHONPATH="$paired_dir:$project"
    local env_prefixes='CUDA_|VLLM_|FLASHINFER_|HF_|TRITON_|OMP_NUM_THREADS=|MAX_JOBS=|XDG_'
    env_prefixes+='|LD_LIBRARY_PATH=|CC=|CXX=|TMPDIR=|PYTHONPATH='
    local environment=(env -u PYTHONHOME -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS)
    if [[ $native_defaults == true ]]; then
        local names=(CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER CUDA_HOME CUDA_CACHE_PATH
            VLLM_{HAS_FLASHINFER_CUBIN,NO_USAGE_STATS,CACHE_ROOT,CONFIG_ROOT,RPC_BASE_PATH}
            VLLM_{WORKER_MULTIPROC_METHOD,SERVER_DEV_MODE} HF_{HUB_OFFLINE,HOME}
            OMP_NUM_THREADS MAX_JOBS XDG_{CACHE_HOME,CONFIG_HOME} TMPDIR TRITON_CACHE_DIR
            FLASHINFER_WORKSPACE_BASE LD_LIBRARY_PATH CC CXX PYTHONDONTWRITEBYTECODE PYTHONPATH)
        local name
        mkdir "$run"/{home,tools}
        ln -s "$uv" "$run/tools/uv"
        environment=(env -i "PATH=$vllm_root/venv/bin:$CUDA_HOME/bin:$run/tools:/usr/bin:/bin"
            "HOME=$run/home" LANG=C.UTF-8 "VIRTUAL_ENV=$vllm_root/venv"
            UV_NO_CACHE=1 UV_OFFLINE=1)
        for name in "${names[@]}"; do
            environment+=("$name=${!name}")
        done
        printf '%s\n' "${environment[@]:2}" >"$run/effective-environment.txt"
    else
        env | LC_ALL=C sort | sed -nE "/^($env_prefixes)/p" >"$run/effective-environment.txt"
    fi
    local attention_config='{"backend":"FLASHINFER","disable_flashinfer_q_quantization":true,'
    attention_config+='"use_trtllm_attention":true}'
    local cache_flags=(--no-enable-prefix-caching)
    if [[ $cache_on == true ]]; then
        cache_flags=(--enable-prefix-caching)
    fi
    start_group "$run" "${environment[@]}" "$python" -B -m vllm.entrypoints.openai.api_server \
        --host 127.0.0.1 --port "$port" --model "$checkpoint" --served-model-name benchmark \
        "$@" --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --mamba-cache-dtype bfloat16 \
        --mamba-ssm-cache-dtype float32 \
        --attention-config "$attention_config" \
        --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_AND_PIECEWISE"}' \
        --language-model-only --skip-tokenizer-init --generation-config vllm --seed 0 \
        "${cache_flags[@]}" --enable-prompt-tokens-details --enable-chunked-prefill \
        --max-model-len "$context" --max-num-batched-tokens "$token_budget" \
        --max-num-seqs "$max_seqs" \
        --max-num-scheduled-tokens "$token_budget" --async-scheduling \
        --kv-cache-memory-bytes "$kv_bytes" --stream-interval 1 \
        --worker-extension-cls vllm_worker.PairedBaselineObserver \
        --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
}
