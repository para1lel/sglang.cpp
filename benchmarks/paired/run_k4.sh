#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash benchmarks/paired/run_k4.sh NATIVE_BINARY GPU_INDEX CAMPAIGN_DIR ROW"
    echo "Current rows are W1, W2, W3, W4, and W5. GPU_INDEX selects the single participating B200."
    echo "Set PAIRED_CHECKPOINT_MANIFEST, PAIRED_NATIVE_SOURCE and PAIRED_NATIVE_SHA256."
    echo "For direct LMCache, set PAIRED_LMCACHE_PYTHON," \
        "PAIRED_LMCACHE_PYTHONPATH and PAIRED_LMCACHE_WHEEL."
    echo "The default profile, server-defaults, uses matched 32768-token cold-cache K4."
    echo "Native uses direct GPU LMCache 0.5.8. Pinned vLLM uses local APC without KV offload."
    echo "W4 can start a fresh campaign. W5 requires a passing W4 in the same campaign."
    echo "Every live timing arm runs one warmup and one measured cohort. Controls remain untimed."
    echo "Each row passes when the measured native makespan is at most 110% of pinned vLLM."
    echo "Historical PAIRED_K4_NATIVE_PROFILE values require compatible older releases:"
    echo "  server-defaults-cache-on-v5, server-defaults-cache-on-v3,"
    echo "  server-defaults-cache-on-v2, server-defaults-cache-off,"
    echo "  exact, ordered-tc-int8,"
    echo "  ordered-tc-int8-split-gqa, ordered-tc-int8-long-split-gqa."
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# != 4 )); then
    usage >&2
    exit 2
fi

binary=$(realpath -e -- "$1")
index=$2
target=$3
row=$4
native_source=${PAIRED_NATIVE_SOURCE:-}
native_sha=${PAIRED_NATIVE_SHA256:-}
manifest=${PAIRED_CHECKPOINT_MANIFEST:-}
native_profile=${PAIRED_K4_NATIVE_PROFILE-server-defaults}
native_fp8=exact
native_split_mode=
native_defaults=false
cache_on=false
current=false
wide_budget=false
lmcache=false
max_seqs=32
case "$native_profile" in
    exact) ;;
    server-defaults)
        native_defaults=true; cache_on=true; current=true; wide_budget=true; lmcache=true ;;
    server-defaults-cache-on-v5)
        native_defaults=true; cache_on=true; current=true; wide_budget=true ;;
    server-defaults-cache-on-v3) native_defaults=true; cache_on=true; current=true ;;
    server-defaults-cache-on-v2) native_defaults=true; cache_on=true ;;
    server-defaults-cache-off) native_defaults=true ;;
    ordered-tc-int8) ;;
    ordered-tc-int8-split-gqa) native_split_mode=1 ;;
    ordered-tc-int8-long-split-gqa) native_split_mode=long ;;
    *) echo "Unknown PAIRED_K4_NATIVE_PROFILE: $native_profile" >&2; exit 2 ;;
esac
if [[ $native_profile != exact && $native_defaults == false ]]; then
    native_fp8=tensor-core-mtp-target
    overrides=$(compgen -e | sed -nE '/^SGLANG_(EXPERIMENTAL_|PREFILL_|DECODE_|SERIAL_)/p')
    if [[ -n $overrides ]]; then
        printf 'Unset native dispatch overrides for %s:\n%s\n' "$native_profile" "$overrides" >&2
        exit 2
    fi
fi
unset SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA
if ! [[ -x $binary && $native_source =~ ^[a-f0-9]{40}$ &&
        $native_sha =~ ^[a-f0-9]{64}$ && -f $manifest ]]; then
    usage >&2
    exit 2
fi
if [[ ! $index =~ ^[0-9]+$ ]]; then
    echo "Specify exactly one participating GPU index." >&2
    exit 2
fi
if [[ $current == true && $row == W6 ]]; then
    echo "W6 is an archived row. The current suite ends at W5." >&2
    exit 2
fi
case "$index" in
    0) export PAIRED_SERVER_CPUS=0-9 PAIRED_CLIENT_CPUS=10-11 ;;
    1|2|3) export PAIRED_SERVER_CPUS=24-33 PAIRED_CLIENT_CPUS=34-35 ;;
    5|6) export PAIRED_SERVER_CPUS=72-81 PAIRED_CLIENT_CPUS=82-83 ;;
    7) export PAIRED_SERVER_CPUS=84-93 PAIRED_CLIENT_CPUS=94-95 ;;
    *) usage >&2; exit 2 ;;
esac
case "$row" in
    W1|W3|W5) arms=(vllm-k4 native-k4) ;;
    W2|W4|W6) arms=(native-k4 vllm-k4) ;;
    *) usage >&2; exit 2 ;;
esac
export PAIRED_PROTOCOL_ROOT
PAIRED_PROTOCOL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
export PAIRED_MAX_NUM_BATCHED_TOKENS=8192
if [[ $wide_budget == true ]]; then
    export PAIRED_MAX_NUM_BATCHED_TOKENS=32768
fi
export PAIRED_PORT=${PAIRED_PORT:-17424}
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
validate_run_settings
if [[ $lmcache == true ]]; then
    source "$paired_dir/lmcache_arm.sh"
    lmcache_pid= lmcache_runtime= lmcache_socket_id=
    lmcache_launcher=$(realpath -e -- "$project/tools/lmcache/serve.sh")
    lmcache_sources=(
        "$project/tools/lmcache/"{serve.sh,sglang_lmcache.py}
        "$project/tools/lmcache/"{sglang_gpu_transfer.py,sglang_socket.py,requirements.txt}
    )
    lmcache_python=$(realpath -es -- "${PAIRED_LMCACHE_PYTHON:?Set the LMCache 0.5.8 interpreter}")
    lmcache_pythonpath=$(resolve_lmcache_pythonpath "${PAIRED_LMCACHE_PYTHONPATH:-}")
    lmcache_wheel=$(realpath -e -- "${PAIRED_LMCACHE_WHEEL:?Set the official LMCache 0.5.8 wheel}")
    [[ -x $lmcache_python ]]
    lmcache_port=${PAIRED_LMCACHE_PORT:-17524}
    if ! [[ $lmcache_port =~ ^[1-9][0-9]{0,4}$ ]] ||
       (( lmcache_port > 65535 || lmcache_port == port )); then
        echo "Choose a separate valid LMCache loopback port with PAIRED_LMCACHE_PORT." >&2
        exit 2
    fi
fi

[[ $port =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 ))
manifest=$(realpath -- "$manifest")
checkpoint=$(realpath -e -- "$checkpoint")
vllm_root=$(realpath -e -- "$vllm_root")
python=$vllm_root/venv/bin/python
seed_cache=$(realpath -e -- "${PAIRED_VLLM_SEED_CACHE:-$host_root/cache/vllm-81a8cb063e62}")
if [[ $native_defaults == true ]]; then
    uv=$(realpath -e -- "$(type -P uv)")
fi
pin=$(jq -er '.vllm_pin' "$project/benchmarks/concurrent/gpu2-reference.json")
model_sha=$(jq -er '.checkpoint_manifest_sha256' \
    "$project/benchmarks/concurrent/gpu2-reference.json")
check_digest() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --check --strict --status
}
check_digest "$native_sha" "$binary"
check_digest "$model_sha" "$manifest"
[[ $(git --no-replace-objects -C "$vllm_root/source" rev-parse HEAD) == "$pin" ]]

uuid=$(nvidia-smi -i "$index" --query-gpu=uuid --format=csv,noheader)
[[ $uuid =~ ^GPU-[0-9a-fA-F-]{36}$ ]]
if [[ $index == 2 ]]; then
    [[ $uuid == GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c ]]
fi
[[ $(nvidia-smi -i "$uuid" --query-gpu=name --format=csv,noheader) == *B200* ]]
mkdir -p -- "$target"
root=$(realpath -- "$target")
exec 7>>"$root/campaign.lock"
flock -n -E 75 7
protocol_hash=$(cd -- "$project" && sha256sum \
    benchmarks/paired/{run_k4.sh,k4.py,common.sh,default_evidence.py,saved_evidence.py} \
    benchmarks/paired/{identity.py,makespan.py} \
    benchmarks/paired/{measure_concurrent.py,vllm_worker.py} \
    benchmarks/streaming/{client.py,check_vllm.py,vllm_observer.py,W1.json,W2.json} \
    benchmarks/concurrent/{client.py,reference.py,freeze_prompts.py,fixture-manifest.json} \
    benchmarks/concurrent/gpu2-reference.json \
    benchmarks/vllm/wheel.sha256)
if [[ $current == true ]]; then
    max_seqs=$(PYTHONPATH="$project:$paired_dir" "$client_python" -B -c \
        'from workload_suite import serving_capacity; print(serving_capacity(1))')
    workload_protocol_hash=$(cd -- "$project" && sha256sum benchmarks/paired/workload_suite.py)
    protocol_hash+=$'\n'"$workload_protocol_hash"
fi
if [[ $cache_on == true ]]; then
    cache_protocol_hash=$(cd -- "$project" && sha256sum benchmarks/paired/cold_cache.py)
    protocol_hash+=$'\n'"$cache_protocol_hash"
fi
if [[ $lmcache == true ]]; then
    lmcache_protocol_hash=$(cd -- "$project" && sha256sum \
        benchmarks/paired/{lmcache_arm.sh,lmcache_evidence.py} "${lmcache_sources[@]#"$project/"}")
    protocol_hash+=$'\n'"$lmcache_protocol_hash"
fi
protocol_sha=$(printf '%s\n' "$protocol_hash" | sha256sum)
identity=$(jq -nc --arg source "$native_source" --arg binary "$native_sha" \
    --arg gpu "$uuid" --arg checkpoint "$model_sha" --arg vllm "$pin" \
    --arg protocol_sha "${protocol_sha%% *}" \
    --arg server "$server_cpus" --arg client "$client_cpus" \
    '{protocol:"matched-k4-v1",sampling:"single-round-v1",replay_receipts:2,
      protocol_sha256:$protocol_sha,
      makespan_criterion:"native-median-at-most-110-percent",
      native_source:$source,native_binary_sha256:$binary,gpu_uuid:$gpu,
      checkpoint_sha256:$checkpoint,vllm_commit:$vllm,
      server_cpus:$server,client_cpus:$client,memory:"localalloc"}')
if [[ $native_profile != exact ]]; then
    if [[ -e $root/identity.json ]]; then
        campaign_id=$(jq -er '.campaign_id | strings' "$root/identity.json")
    else
        read -r campaign_id </proc/sys/kernel/random/uuid
    fi
    [[ $campaign_id =~ ^[0-9a-f-]{36}$ ]]
    if [[ $native_defaults == true ]]; then
        identity=$(jq -c --arg model "$checkpoint" --arg campaign_id "$campaign_id" \
            --argjson port "$port" --argjson cache_on "$cache_on" \
            --argjson current "$current" --argjson max_seqs "$max_seqs" \
            --argjson wide_budget "$wide_budget" --argjson token_budget "$token_budget" \
            --argjson lmcache "$lmcache" --arg lmcache_port "${lmcache_port:-}" \
            --arg lmcache_python "${lmcache_python:-}" --argjson lmcache_uid "$EUID" \
            --arg lmcache_pythonpath "${lmcache_pythonpath:-}" \
            '. + {protocol:"matched-k4-native-defaults-v1",campaign_id:$campaign_id,
                  native_profile:"server-defaults",checkpoint_path:$model,port:$port}
             | if $cache_on then . + {protocol:"matched-k4-native-defaults-cache-on-v2",
                 cache_policy:"reset-before-every-cohort"} else . end
             | if $current then . + {protocol:"matched-k4-native-defaults-cache-on-v3",
                 participating_gpus:[.gpu_uuid],max_running_requests:$max_seqs} else . end
             | if $wide_budget then . + {protocol:"matched-k4-native-defaults-cache-on-v5",
                 max_num_batched_tokens:$token_budget} else . end
             | if $lmcache then . + {protocol:"matched-k4-native-defaults-lmcache-slc3-drain",
                 lmcache_version:"0.5.8",lmcache_wire:"SLC3",
                 lmcache_ready_version:"0.5.8+native-v3",lmcache_port:($lmcache_port | tonumber),
                 lmcache_capacity_bytes:68719476736,lmcache_python:$lmcache_python,
                 lmcache_uid:$lmcache_uid,lmcache_pythonpath:$lmcache_pythonpath} else . end' \
            <<<"$identity")
    else
        identity=$(jq -c --arg profile "$native_profile" --arg fp8 "$native_fp8" \
        --arg campaign_id "$campaign_id" --arg split_mode "$native_split_mode" \
        '. + {protocol:"matched-k4-opt-in-v1",campaign_id:$campaign_id,native_profile:$profile,
              native_config:{fp8_prefill:$fp8,mtp_steps:4,mtp_int8_head:true}}
         | if $split_mode == "1" then .native_config.experimental_target_split_gqa = true
           elif $split_mode == "long" then .native_config.experimental_long_target_split_gqa = true
           else . end' \
            <<<"$identity")
    fi
fi
if [[ ! -e $root/identity.json ]] &&
   [[ $row == W1 || ( $wide_budget == true && $row == W4 ) ]]; then
    printf '%s\n' "$identity" >"$root/identity.json"
fi
if [[ ! -f $root/identity.json || $(cat "$root/identity.json") != "$identity" ]]; then
    echo "Campaign identity differs. Use a fresh campaign directory." >&2
    exit 2
fi
PYTHONPATH="$project" "$client_python" -B "$paired_dir/k4.py" check-prior "$row" "$root"

directory=$root/$row
mkdir -- "$directory"
printf '%s\n' "$identity" >"$directory/identity.json"
printf '%s\n' "$protocol_hash" >"$directory/protocol.sha256"
printf '%s\n' "${arms[@]}" >"$directory/arm-order.txt"
printf '%s\n' "$seed_cache" >"$directory/cache-seed.txt"
lease_held=0
finish() {
    local status=$?
    trap - EXIT
    stop_group
    if [[ $lmcache == true ]]; then
        stop_lmcache || status=2
    fi
    if (( lease_held )); then
        record_compute "$directory/compute-after.csv" || status=2
        check_idle || status=2
        port_free || status=2
        if [[ $lmcache == true && ${lmcache_port_held:-0} == 1 ]]; then
            lmcache_port_free || status=2
        fi
    fi
    date -u +%FT%TZ >"$directory/finished-utc.txt"
    printf '%s\n' "$status" >"$directory/exit-code.txt"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cp -- "$manifest" "$directory/checkpoint.sha256"
(cd -- "$checkpoint" && sha256sum --check --strict "$directory/checkpoint.sha256") \
    >"$directory/checkpoint-check.txt"
site=$("$python" -IB -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
cp -- "$vllm_root"/{artifact.txt,freeze.txt,native.sha256} \
    "$project/benchmarks/vllm/wheel.sha256" "$directory/"
(cd -- "$vllm_root/wheels" && sha256sum --check --strict "$directory/wheel.sha256") \
    >"$directory/wheel-check.txt"
(cd -- "$site" && sha256sum --check --strict "$directory/native.sha256") \
    >"$directory/vllm-native-check.txt"
if [[ $lmcache == true ]]; then
    PYTHONPATH="$project" "$client_python" -B "$paired_dir/lmcache_evidence.py" verify-wheel \
        "$lmcache_wheel" "$lmcache_python" "$lmcache_pythonpath" >"$directory/lmcache-wheel.json"
fi

output=$directory
if [[ $row == W1 || $row == W2 ]]; then
    fixtures=$directory
    context=2560
    kv_bytes=10737418240
else
    prepare_fixtures
    context=132096
    kv_bytes=137438953472
fi

port_free() {
    local listeners
    listeners=$(ss -Hltn "sport = :$port") || return
    [[ -z $listeners ]] || { echo "Port $port already has a listener." >&2; return 75; }
}
exec 6>>"$vllm_root/loopback-$port.lock"
flock -n -E 75 6
exec 8<"/tmp/gpu-coord/$uuid.lock"
flock -n -E 75 8
prepare_run "$index" "$directory/reservation"
[[ $gpu_uuid == "$uuid" ]]
lease_held=1
port_free
if [[ $lmcache == true ]]; then
    exec 5>>"$vllm_root/loopback-$lmcache_port.lock"
    flock -n -E 75 5
    lmcache_port_held=1
    lmcache_port_free
fi
numactl --localalloc taskset --cpu-list "$client_cpus" /usr/bin/true
for observation in 1 2 3; do
    check_idle
    read -r free utilization < <(nvidia-smi -i "$gpu_uuid" \
        --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
    printf '%s %s %s\n' "$observation" "$free" "$utilization" >>"$directory/idle-samples.txt"
    [[ $free -ge 175000 && $utilization -eq 0 ]]
    [[ $observation == 3 ]] || sleep 1
done
export CUDA_VISIBLE_DEVICES=$gpu_uuid CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
env | LC_ALL=C sort |
    sed -nE '/^(CUDA_|LD_LIBRARY_PATH=|SGLANG_(EXPERIMENTAL_|PREFILL_|DECODE_|SERIAL_))/p' \
    >"$directory/native-environment.txt"
if [[ $native_defaults == true ]]; then
    env -i PATH=/usr/bin:/bin "LD_LIBRARY_PATH=$compat" ldd "$binary" \
        >"$directory/native-ldd.txt"
else
    ldd "$binary" >"$directory/native-ldd.txt"
fi
if grep -q 'not found' "$directory/native-ldd.txt"; then
    echo "Native shared libraries are missing. See $directory/native-ldd.txt." >&2
    exit 2
fi
client=(env "PYTHONPATH=$project" numactl --localalloc taskset --cpu-list "$client_cpus"
    "$client_python" -B "$paired_dir/k4.py")
fetch=(curl --noproxy '*' -fsS --max-time 120)

start_native_defaults() {
    local run=$1 arm=$2
    local args=("$binary" --port "$port")
    if [[ $arm == native-k4-trace ]]; then
        args+=(--trace-batches)
    fi
    mkdir "$run"/{home,cache,config,tmp}
    local environment=()
    if [[ $lmcache == true ]]; then
        start_lmcache "$run" || return
        environment=("XDG_RUNTIME_DIR=$lmcache_runtime")
    fi
    start_group "$run" env -i PATH=/usr/bin:/bin LANG=C.UTF-8 "${environment[@]}" \
        "HOME=$run/home" "TMPDIR=$run/tmp" "XDG_CACHE_HOME=$run/cache" \
        "XDG_CONFIG_HOME=$run/config" "CUDA_CACHE_PATH=$run/cuda-cache" \
        "CUDA_VISIBLE_DEVICES=$gpu_uuid" CUDA_DEVICE_ORDER=PCI_BUS_ID \
        "LD_LIBRARY_PATH=$compat" "${args[@]}"
}

check_native_info() {
    local depth=$1 traced=$2 fp8=$3 int8=$4 split_mode=$5
    jq -e --argjson depth "$depth" --argjson traced "$traced" --arg path "$checkpoint" \
        --arg profile "$native_profile" --arg fp8 "$fp8" --argjson int8 "$int8" \
        --arg split_mode "$split_mode" \
        '.mtp_steps == $depth and .model_path == $path and .max_running_requests == 32
         and .max_num_batched_tokens == 8192 and .prefix_cache_entries == 0
         and .batch_cuda_graphs == true and .eager == false and .trace_batches == $traced
         and .fp8_prefill == $fp8 and .host == "127.0.0.1" and .device == 0
         and (if has("experimental_target_split_gqa") then .experimental_target_split_gqa
              else false end) == ($split_mode == "1")
         and (if has("experimental_long_target_split_gqa") then .experimental_long_target_split_gqa
              else false end) == ($split_mode == "long")
         and (if $profile == "exact" then (.mtp_int8_head // false) == false
              else .mtp_int8_head == $int8 end)' \
        server-info.json >server-check.txt
}

finish_arm() {
    local arm=$1 run=$directory/$1
    record_compute "$run/compute-measurement-end.csv"
    if [[ $arm == vllm-k4 ]]; then
        if [[ $native_defaults == true ]]; then
            cat "/proc/$server_pid/cmdline" | cmp -s -- live-argv.bin -
            cat "/proc/$server_pid/environ" >live-environment-end.bin
            cmp -s live-environment.bin live-environment-end.bin
        fi
        "${fetch[@]}" "$url/metrics" >metrics-after.txt
        "${client[@]}" worker "$row" "$run" "$directory/reservation" "$checkpoint"
        "${client[@]}" activity "$run" vllm
        if [[ $lmcache == true ]]; then
            local engine_pid
            engine_pid=$(jq -er '.worker_pid' worker-identity.json)
            check_compute_owners compute-measurement-end.csv "$engine_pid"
        fi
    else
        cat "/proc/$server_pid/cmdline" | cmp -s -- live-argv.bin -
        if [[ $native_defaults == true ]]; then
            native_environment >live-environment-end.bin
            cmp -s live-environment.bin live-environment-end.bin
        else
            "${client[@]}" native-dispatch "$server_pid" >live-dispatch-end.bin
            cmp -s -- live-dispatch.bin live-dispatch-end.bin
        fi
        if [[ $lmcache == true ]]; then
            check_compute_owners compute-measurement-end.csv "$server_pid" "$lmcache_pid"
            record_lmcache_peer "$run" -end
            cmp -s lmcache-ready.json lmcache-ready-end.json
            cmp -s lmcache-live-argv.bin lmcache-live-argv-end.bin
        else
            for receipt in compute-ready.csv compute-measurement-end.csv; do
                awk -F', *' -v uuid="$gpu_uuid" -v pid="$server_pid" \
                    '$1 == uuid {count++; if ($2 == pid) owned++}
                     END {exit !(count == 1 && owned == 1)}' "$receipt"
            done
        fi
        if [[ $arm == native-k4-trace ]]; then
            "${client[@]}" activity "$run" native
        fi
    fi
    stop_group
    if [[ $lmcache == true && $arm != vllm-k4 ]]; then
        local endpoint=$lmcache_runtime/sglang-lmcache.sock
        stop_lmcache
        [[ ! -e $endpoint && ! -e ${endpoint%/*} ]]
        jq -nc --arg socket "$endpoint" \
            '{socket:$socket,runtime_absent:true,socket_absent:true}' >lmcache-retired.json
    fi
    check_idle
    port_free
    record_compute "$run/compute-after.csv"
    if [[ $lmcache == true ]]; then
        check_compute_owners "$run/compute-after.csv"
        ss -Hltn "sport = :$port" >listener-after.txt
        ss -Hltn "sport = :$lmcache_port" >lmcache-listener-after.txt
        [[ ! -s listener-after.txt && ! -s lmcache-listener-after.txt ]]
    fi
}

native_environment() {
    local options=()
    if [[ $lmcache == true ]]; then
        options=(--lmcache)
    fi
    "${client[@]}" native-default-environment "$server_pid" "${options[@]}"
}

run_arm() {
    local arm=$1 run=$directory/$1 depth=4 traced=false int8=false client_status=0
    local fp8=$native_fp8
    local split_mode=$native_split_mode
    local flags=() options=() dispatch=()
    mkdir -- "$run"
    cd -- "$run"
    port_free
    if [[ $lmcache == true ]]; then lmcache_port_free; fi
    if [[ $arm == vllm-k4 ]]; then
        start_vllm "$run" 5>&- 6>&- 7>&- 8<&-
        if [[ $native_defaults == true ]]; then
            cat "/proc/$server_pid/cmdline" >live-argv.bin
            cat "/proc/$server_pid/environ" >live-environment.bin
            awk '/^Cpus_allowed_list:/ {print $2}' "/proc/$server_pid/status" >live-cpus.txt
        fi
        "${fetch[@]}" "$url/server_info?config_format=json" >server-info.json
        "${fetch[@]}" -H 'Content-Type: application/json' \
            -d '{"method":"baseline_snapshot","timeout":90}' \
            "$url/collective_rpc" >effective-worker.json
        "${fetch[@]}" "$url/metrics" >metrics-before.txt
        if [[ $native_defaults == true ]]; then
            if [[ $lmcache == true ]]; then
                check_compute_owners "$run/compute-before.csv"
            fi
            "${client[@]}" worker "$row" "$run" "$directory/reservation" "$checkpoint" \
                --before-measurement
        fi
    elif [[ $native_defaults == true ]]; then
        [[ $arm != native-k4-trace ]] || traced=true
        start_native_defaults "$run" "$arm" 5>&- 6>&- 7>&- 8<&-
        check_digest "$native_sha" "/proc/$server_pid/exe"
        sha256sum "/proc/$server_pid/exe" >running-binary.sha256
        cat "/proc/$server_pid/cmdline" >live-argv.bin
        awk '/^Cpus_allowed_list:/ {print $2}' "/proc/$server_pid/status" >live-cpus.txt
        native_environment >live-environment.bin
        "${fetch[@]}" "$url/server_info" >server-info.json
        "${client[@]}" native-default-check "$run" >server-check.txt
    else
        case "$arm" in
            native-k0) depth=0; flags+=(--mtp-steps 0) ;;
            native-exact-k0)
                depth=0
                fp8=exact
                split_mode=
                flags+=(--mtp-steps 0)
                options+=(--native-control "$directory/native-k0/measurement")
                ;;
            native-k4-trace) traced=true; flags+=(--trace-batches) ;;
        esac
        if (( depth == 4 )); then
            options+=(--native-control "$directory/native-k0/measurement")
            if [[ $native_profile != exact ]]; then
                flags+=(--mtp-steps 4 --mtp-int8-head)
                int8=true
            fi
        fi
        if [[ -n $split_mode ]]; then
            dispatch+=("SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA=$split_mode")
        fi
        mkdir cuda-cache
        start_group "$run" env "CUDA_CACHE_PATH=$run/cuda-cache" "${dispatch[@]}" \
            "$binary" --checkpoint "$checkpoint" --device 0 --port "$port" \
            --batch-cuda-graphs --prefix-cache-entries 0 \
            --fp8-prefill "$fp8" \
            --max-active 32 --max-num-batched-tokens "$token_budget" "${flags[@]}" \
            5>&- 6>&- 7>&- 8<&-
        check_digest "$native_sha" "/proc/$server_pid/exe"
        sha256sum "/proc/$server_pid/exe" >running-binary.sha256
        cat "/proc/$server_pid/cmdline" >live-argv.bin
        "${client[@]}" native-dispatch "$server_pid" >live-dispatch.bin
        "${fetch[@]}" "$url/server_info" >server-info.json
        check_native_info "$depth" "$traced" "$fp8" "$int8" "$split_mode"
    fi
    ss -Hlnpt "sport = :$port" >listener.txt
    awk -v pid="pid=$server_pid," -v address="127.0.0.1:$port" \
        '$4 == address && index($0, pid) {owned++} END {exit !(NR == 1 && owned == 1)}' listener.txt
    local capture=("${client[@]}" capture "$row" "$arm" "$url" "$fixtures" "$run/measurement"
        "${options[@]}")
    write_command "$run/client-command.sh" "${capture[@]}"
    date -u +%FT%TZ >measurement-started-utc.txt
    "${capture[@]}" >client.log 2>&1 5>&- 6>&- 7>&- 8<&- 9<&- || client_status=$?
    printf '%s\n' "$client_status" >client-exit-code.txt
    date -u +%FT%TZ >measurement-finished-utc.txt
    finish_arm "$arm"
    if (( client_status != 0 )); then
        cat client.log >&2
        return 2
    fi
}

if [[ $native_defaults == false ]]; then
    run_arm native-k0
    if [[ $native_profile != exact ]]; then
        run_arm native-exact-k0
    fi
fi
if [[ $row == W1 ]]; then
    run_arm native-k4-trace
fi
for arm in "${arms[@]}"; do
    run_arm "$arm"
done
"${client[@]}" assess "$row" "$fixtures" "$directory"
