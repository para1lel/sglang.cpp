#!/usr/bin/env bash
# Source for a paired run, or execute for an untimed pinned-vLLM readiness check.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../paired/common.sh"

pair_idle() {
    local device active
    for device in "${devices[@]}"; do
        active=$(nvidia-smi -i "$device" --query-compute-apps=pid --format=csv,noheader)
        [[ -z $active ]] || { echo "$device occupied by $active" >&2; return 75; }
    done
}

reserve_pair() {
    local first=$1 second=$2 base=$3
    local index uuid name free utilization sample device
    devices=()
    for index in "$first" "$second"; do
        [[ $index =~ ^[0-9]+$ ]]
        IFS=, read -r uuid name < <(nvidia-smi -i "$index" --query-gpu=uuid,name \
            --format=csv,noheader)
        [[ $uuid =~ ^GPU-[a-f0-9-]{36}$ && $name == *B200* ]]
        devices+=("$uuid")
    done
    [[ ${devices[0]} != "${devices[1]}" ]]
    exec 5<"/tmp/gpu-coord/${devices[0]}.lock" 6<"/tmp/gpu-coord/${devices[1]}.lock"
    exec 7>>"$vllm_root/${devices[1]}.lock" 8>>"$vllm_root/loopback-$port.lock"
    for descriptor in 5 6 7 8; do flock -n -E 75 "$descriptor"; done
    prepare_run "$first" "$base/reservation"
    printf '%s %s\n' "$second" "${devices[1]}" >>"$base/reservation/gpu.txt"
    for sample in 1 2 3; do
        pair_idle
        for device in "${devices[@]}"; do
            read -r free utilization < <(nvidia-smi -i "$device" \
                --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
            printf '%s %s %s %s\n' "$sample" "$device" "$free" "$utilization" >>"$base/idle.txt"
            [[ $free -ge 175000 && $utilization -eq 0 ]]
        done
        [[ $sample == 3 ]] || sleep 1
    done
    export CUDA_VISIBLE_DEVICES="${devices[0]},${devices[1]}" CUDA_DEVICE_ORDER=PCI_BUS_ID
    [[ -z $(ss -Hltn "sport = :$port") ]]
}

server_receipts() {
    local run=$1
    cat "/proc/$server_pid/cmdline" >"$run/live-argv.bin"
    cat "/proc/$server_pid/environ" >"$run/live-environment.bin"
    awk '/^Cpus_allowed_list:/ {print $2}' "/proc/$server_pid/status" >"$run/live-cpus.txt"
    ss -Hlnpt "sport = :$port" >"$run/listener.txt"
    awk -v pid="pid=$server_pid," -v addr="127.0.0.1:$port" '
        $4 == addr && index($0,pid) {owned++} END {exit !(NR == 1 && owned == 1)}' \
        "$run/listener.txt"
}

start_reference() {
    local run=$1 row=$2
    local native_defaults=true cache_on=true context=132096 kv_bytes=137438953472
    if [[ $row == W1 || $row == W2 ]]; then context=2560; kv_bytes=10737418240; fi
    local seed_cache=${PAIRED_VLLM_SEED_CACHE:-$host_root/cache/vllm-81a8cb063e62}
    local python=$vllm_root/venv/bin/python uv token_budget=32768 max_seqs=16
    uv=$(realpath -e -- "$(type -P uv)")
    local client=(env "PYTHONPATH=$project" "$client_python" -B "$paired_dir/k4.py")
    local pin
    pin=$(jq -er '.vllm_pin' "$project/benchmarks/concurrent/gpu2-reference.json")
    [[ $(git -C "$vllm_root/source" rev-parse HEAD) == "$pin" ]]
    local site
    site=$("$python" -IB -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
    (cd "$vllm_root/wheels" && sha256sum -c --strict "$project/benchmarks/vllm/wheel.sha256") \
        >"$run/wheel-check.txt"
    (cd "$site" && sha256sum -c --strict "$vllm_root/native.sha256") >"$run/native-check.txt"
    pair_idle
    start_vllm "$run" --tensor-parallel-size 2 5<&- 6<&- 7>&- 8>&- 9>&-
    server_receipts "$run"
    local fetch=(curl --noproxy '*' -fsS --max-time 120)
    "${fetch[@]}" "$url/server_info?config_format=json" >"$run/server-info.json"
    reference_snapshot "$run/effective-worker.json"
    "${fetch[@]}" "$url/metrics" >"$run/metrics-before.txt"
}

reference_snapshot() {
    curl --noproxy '*' -fsS --max-time 120 -H 'Content-Type: application/json' \
        -d '{"method":"tp2_snapshot","timeout":90}' "$url/collective_rpc" >"$1"
}

verify_reference() {
    local run=$1 row=$2 reservation=$3
    shift 3
    PYTHONPATH="$project" "$client_python" -B "$paired_dir/tp2_vllm.py" verify "$row" \
        "$run" "$reservation" "$checkpoint" "$vllm_root" "${devices[@]}" "$@"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    if [[ $# != 4 || $3 != W[1-5] ]]; then
        echo 'Usage: bash benchmarks/tp2/session.sh GPU0 GPU1 ROW NEW_OUTPUT_DIR' >&2
        exit 2
    fi
    mkdir -- "$4"
    base=$(realpath "$4")
    run=$base/vllm-k4
    reserve_pair "$1" "$2" "$base"
    trap 'stop_group' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    mkdir "$run"
    cd "$run"
    start_reference "$run" "$3"
    verify_reference "$run" "$3" "$base/reservation" --ready-only >"$run/ready.json"
    stop_group
    pair_idle
    record_compute "$run/compute-after.csv"
fi
