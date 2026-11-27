#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/session.sh"
source "$paired_dir/lmcache_arm.sh"
tp2_dir=$project/benchmarks/tp2
export PYTHONPATH=$project:$paired_dir
[[ $# == 6 && ( $1 == qualify || $1 == measure ) && ( $4 == W[1-5] || $4 == all ) ]] || {
    echo 'Usage: run.sh qualify|measure GPU0 GPU1 W1..W5|all NATIVE_BINARY CAMPAIGN_DIR' >&2
    exit 2
}
mode=$1
binary=$(realpath -e "$5")
fixtures=$(realpath -e "${PAIRED_FIXTURES:?Provide the frozen TP2 corpus}")
expected_sha=${PAIRED_NATIVE_SHA256:?Provide the fixed executable SHA-256}
native_source=${PAIRED_NATIVE_SOURCE:?Provide the fixed native source revision}
[[ -x $binary && $expected_sha =~ ^[0-9a-f]{64}$ && $native_source =~ ^[0-9a-f]{40}$ ]]
printf '%s  %s\n' "$expected_sha" "$binary" | sha256sum -c --strict --status
lmcache_python=$(realpath -es "${PAIRED_LMCACHE_PYTHON:?Provide the LMCache interpreter}")
lmcache_pythonpath=$(resolve_lmcache_pythonpath "${PAIRED_LMCACHE_PYTHONPATH:-}")
wheel=$(realpath -e "${PAIRED_LMCACHE_WHEEL:?Provide the official LMCache wheel}")
lmcache_launcher=$project/tools/lmcache/serve.sh
lmcache_sources=("$project"/tools/lmcache/{serve.sh,sglang_lmcache.py,sglang_gpu_transfer.py}
                 "$project"/tools/lmcache/{sglang_socket.py,requirements.txt})
lmcache_port=${PAIRED_LMCACHE_PORT:-17655}
lmcache_memory_gib=128
[[ $lmcache_port =~ ^[1-9][0-9]{0,4}$ && $port =~ ^[1-9][0-9]{0,4}$ ]]
(( lmcache_port <= 65535 && port <= 65535 && lmcache_port != port ))
mkdir -p -- "$6"
base=$(realpath "$6")
exec {campaign_lock}<"$base"
flock -n -E 75 "$campaign_lock"
session=$base
if [[ -e $base/identity.json ]]; then
    if [[ $(jq -er .protocol "$base/identity.json") != tp2-paired-v2 ]]; then
        echo 'Use a new campaign root for the single-daemon protocol.' >&2
        exit 1
    fi
    [[ $mode == measure && $4 != all && ! -e $base/$4 ]]
    session=$base/session-$4
    mkdir "$session"
else
    [[ -z $(ls -A "$base") ]]
fi
reserve_pair "$2" "$3" "$session"
exec 3>>"$vllm_root/loopback-$lmcache_port.lock"
flock -n -E 75 3
lmcache_port_free
"$client_python" -B "$paired_dir/tp2_corpus.py" verify "$fixtures" >"$session/corpus.json"
lmcache_evidence verify-wheel "$wheel" "$lmcache_python" "$lmcache_pythonpath" \
    >"$session/lmcache-wheel.json"
(cd "$project" && sha256sum \
    benchmarks/tp2/{run.sh,session.sh,native.py,cohorts.py,lifecycle.py,cache_activity.py} \
    benchmarks/paired/{common.sh,lmcache_arm.sh,lmcache_evidence.py,tp2_vllm.py} \
    benchmarks/paired/{tp2_corpus.py,tp2-corpus-v1.json,k4.py,vllm_worker.py,default_evidence.py} \
    benchmarks/paired/{saved_evidence.py,cold_cache.py,makespan.py,workload_suite.py} \
    benchmarks/paired/{measure_concurrent.py,identity.py,driver_health.py} \
    benchmarks/streaming/{client.py,vllm_observer.py,check_vllm.py,gpu3-reference.json} \
    benchmarks/concurrent/{client.py,reference.py,freeze_prompts.py} \
    benchmarks/concurrent/{gpu2-reference.json,fixture-manifest.json} \
    benchmarks/vllm/wheel.sha256 \
    "${lmcache_sources[@]#"$project/"}") >"$session/protocol.sha256"
campaign_id=$(cat /proc/sys/kernel/random/uuid)
if [[ $session != "$base" ]]; then campaign_id=$(jq -er .campaign_id "$base/identity.json"); fi
jq -n --arg mode "$mode" --arg id "$campaign_id" --arg rev "$native_source" \
    --arg binary "$binary" --arg sha "$expected_sha" --arg model "$checkpoint" \
    --arg root "$vllm_root" --arg python "$client_python" \
    --arg protocol "$(sha256sum "$session/protocol.sha256" | cut -d' ' -f1)" \
    --arg wire "$(jq -r .manifest_sha256 "$session/corpus.json")" --argjson port "$port" \
    --arg server "$server_cpus" --arg client "$client_cpus" \
    --arg dev0 "${devices[0]}" --arg dev1 "${devices[1]}" \
    --argjson cache_port "$lmcache_port" --argjson uid "$EUID" \
    --argjson capacity "$((lmcache_memory_gib << 30))" \
    --slurpfile lm "$session/lmcache-wheel.json" \
    --slurpfile ref "$project/benchmarks/concurrent/gpu2-reference.json" \
    '{protocol:"tp2-paired-v2",protocol_sha256:$protocol,corpus_sha256:$wire,
      campaign_id:$id,mode:$mode,
      native_source:$rev,binary:$binary,binary_sha256:$sha,checkpoint_path:$model,
      checkpoint_sha256:$ref[0].checkpoint_manifest_sha256,vllm_pin:$ref[0].vllm_pin,
      vllm_root:$root,
      client_python:$python,port:$port,
      server_cpus:$server,client_cpus:$client,devices:[$dev0,$dev1],
      daemon_port:$cache_port,lmcache_capacity_bytes:$capacity,lmcache_uid:$uid,
      lmcache_wire:"SLC3",lmcache_ready_version:"0.5.8+native-v3",lmcache_python:$lm[0].python,
      lmcache_pythonpath:$lm[0].pythonpath}' >"$session/identity.json"
if [[ $session != "$base" ]]; then
    receipts=(identity.json protocol.sha256 corpus.json lmcache-wheel.json
              reservation/{gpu.txt,cpus.txt})
    for receipt in "${receipts[@]}"; do
        cmp -- "$session/$receipt" "$base/$receipt"
    done
fi

lmcache_runtime= lmcache_pid= lmcache_socket_id=
cleanup_arm() {
    stop_group
    stop_lmcache
}
trap cleanup_arm EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

server_preflight() {
    record_compute "$1/compute-before.csv"
    if [[ -z $lmcache_pid ]]; then pair_idle; return; fi
    local gpu
    for gpu in "${devices[@]}"; do
        lmcache_evidence owners "$1/compute-before.csv" "$gpu" "$lmcache_pid"
    done
}

launch_native() {
    local run=$1 directory=$1/daemon gpu_uuid=$CUDA_VISIBLE_DEVICES
    local args=("$binary" --port "$port" --tp 2 --max-active 16)
    [[ $run != */native-k4-trace ]] || args+=(--trace-batches)
    pair_idle
    record_compute "$run/compute-before-daemons.csv"
    lmcache_runtime=$(mktemp -d /tmp/sglang-tp2.XXXXXXXX)
    mkdir -p "$directory"/{home,cache,tmp}
    launch_lmcache "$directory" 3>&- 5<&- 6<&- 7>&- 8>&- 9>&-
    mkdir "$run"/{home,cache,config,tmp}
    start_group "$run" env -i PATH=/usr/bin:/bin LANG=C.UTF-8 "XDG_RUNTIME_DIR=$lmcache_runtime" \
        "HOME=$run/home" "TMPDIR=$run/tmp" "XDG_CACHE_HOME=$run/cache" \
        "XDG_CONFIG_HOME=$run/config" "CUDA_CACHE_PATH=$run/cuda-cache" \
        "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" CUDA_DEVICE_ORDER=PCI_BUS_ID \
        "LD_LIBRARY_PATH=$compat" "${args[@]}" 3>&- 5<&- 6<&- 7>&- 8>&- 9>&-
    server_receipts "$run"
    sha256sum "/proc/$server_pid/exe" >"$run/running-binary.sha256"
    curl --noproxy '*' -fsS "$url/server_info" >"$run/server-info.json"
    "$client_python" -B "$tp2_dir/native.py" "$run" >"$run/ready.json"
}

run_arm() {
    local row=$1 arm=$2
    run=$base/$row/$arm
    mkdir -p "$run"
    cd "$run"
    if [[ $arm == vllm-k4 ]]; then
        start_reference "$run" "$row" 3>&-
        verify_reference "$run" "$row" "$base/reservation" --ready-only >"$run/ready.json"
    else
        launch_native "$run"
    fi
    local cohort=("$client_python" -B "$tp2_dir/cohorts.py")
    if [[ $mode == measure || $arm == native-k4 ]]; then
        local action=capture
        [[ $mode != qualify ]] || action=probe
        local capture=(numactl --localalloc taskset --cpu-list "$client_cpus" "${cohort[@]}"
                       "$action" "$run" "$fixtures" --endpoint "$url")
        write_command "$run/client-command.sh" "${capture[@]}"
        "${capture[@]}" >"$run/$action.json"
    fi
    record_compute "$run/compute-measurement-end.csv"
    cat "/proc/$server_pid/cmdline" >"$run/live-argv-end.bin"
    cmp -- "$run/live-argv.bin" "$run/live-argv-end.bin"
    cat "/proc/$server_pid/environ" >"$run/live-environment-end.bin"
    if [[ $arm == vllm-k4 ]]; then
        curl --noproxy '*' -fsS "$url/metrics" >"$run/metrics-after.txt"
        reference_snapshot "$run/effective-worker-end.json"
        verify_reference "$run" "$row" "$base/reservation" >"$run/verified-end.json"
    else
        curl --noproxy '*' -fsS "$url/server_info" >"$run/server-info-end.json"
        record_lmcache_peer "$run/daemon" -end
        "$client_python" -B "$tp2_dir/native.py" "$run" --completed >"$run/verified-end.json"
    fi
    cleanup_arm
    pair_idle
    lmcache_port_free
    for number in "$port" "$lmcache_port"; do
        ss -Hltn "sport = :$number"
    done >"$run/listeners-after.txt"
    [[ ! -s $run/listeners-after.txt ]]
    record_compute "$run/compute-after.csv"
    if [[ $mode == qualify && $arm == native-k4 ]]; then
        "${cohort[@]}" replay-probe "$run" "$fixtures" | cmp - "$run/probe.json"
    fi
    "${cohort[@]}" binding "$run" "$fixtures" | jq '{binding:.,retired:true}' >"$run/completed.json"
}

rows=("$4")
[[ $4 != all ]] || rows=(W1 W2 W3 W4 W5)
require_predecessor() {
    local prior=$1
    local check=("$client_python" -B "$tp2_dir/cohorts.py" assess "$base/$prior" "$fixtures")
    [[ -s $base/$prior/comparison.json ]]
    "${check[@]}" | cmp - "$base/$prior/comparison.json"
    jq -e '.passed' "$base/$prior/comparison.json"
}
if [[ $mode == measure && $session != "$base" ]]; then
    for prior in W{1..5}; do
        [[ ! -e $base/$prior ]] || require_predecessor "$prior"
    done
fi
for row in "${rows[@]}"; do
    if [[ $mode == measure ]]; then
        case $row in
            W2) [[ -e $base/W1/comparison.json ]] ;;
            W3) [[ -e $base/W1/comparison.json && -e $base/W2/comparison.json ]] ;;
            W5) [[ -e $base/W4/comparison.json ]] ;;
        esac
    fi
    arms=(vllm-k4 native-k4)
    if [[ $row == W2 || $row == W4 ]]; then arms=(native-k4 vllm-k4); fi
    for arm in "${arms[@]}"; do run_arm "$row" "$arm"; done
    if [[ $row == W1 && $mode == measure ]]; then run_arm W1 native-k4-trace; fi
    if [[ $mode == measure ]]; then
        "$client_python" -B "$tp2_dir/cohorts.py" assess "$base/$row" "$fixtures" \
            >"$base/$row/comparison.json"
        jq -e '.passed' "$base/$row/comparison.json"
    fi
done
comparisons=("$base"/W[1-5]/comparison.json)
if [[ $mode == measure && ${#comparisons[@]} == 5 ]]; then
    "$client_python" -B "$tp2_dir/cohorts.py" compare "$base" "$fixtures" >"$base/comparison.json"
    jq -e '.passed' "$base/comparison.json"
fi
