#!/usr/bin/env bash
# Usage: run.sh qualify|measure GPU0 GPU1 W1|W2|W3|W4|W5|all NATIVE_BINARY CAMPAIGN_DIR
# Require PAIRED_NATIVE_SOURCE, PAIRED_NATIVE_SHA256, PAIRED_FIXTURES,
# PAIRED_LMCACHE_PYTHON, PAIRED_LMCACHE_PYTHONPATH and PAIRED_LMCACHE_WHEEL.
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
first_port=${PAIRED_LMCACHE_PORT:-17655}
[[ $first_port =~ ^[1-9][0-9]{0,4}$ && $port =~ ^[1-9][0-9]{0,4}$ ]]
(( first_port < 65535 && port <= 65535 ))
daemon_ports=("$first_port" "$((first_port+1))")
[[ $port != "${daemon_ports[0]}" && $port != "${daemon_ports[1]}" ]]
mkdir -p -- "$6"
base=$(realpath "$6")
exec {campaign_lock}<"$base"
flock -n -E 75 "$campaign_lock"
session=$base
if [[ -e $base/identity.json ]]; then
    [[ $mode == measure && $4 != all && ! -e $base/$4 ]]
    session=$base/session-$4
    mkdir "$session"
else
    [[ -z $(ls -A "$base") ]]
fi
reserve_pair "$2" "$3" "$session"
exec 3>>"$vllm_root/loopback-${daemon_ports[0]}.lock"
exec 4>>"$vllm_root/loopback-${daemon_ports[1]}.lock"
flock -n -E 75 3
flock -n -E 75 4
for lmcache_port in "${daemon_ports[@]}"; do lmcache_port_free; done
"$client_python" -B "$paired_dir/tp2_corpus.py" verify "$fixtures" >"$session/corpus.json"
lmcache_evidence verify-wheel "$wheel" "$lmcache_python" "$lmcache_pythonpath" \
    >"$session/lmcache-wheel.json"
(cd "$project" && sha256sum \
    benchmarks/tp2/{run.sh,session.sh,native.py,cohorts.py,lifecycle.py,cache_activity.py} \
    benchmarks/paired/{common.sh,lmcache_arm.sh,lmcache_evidence.py,tp2_vllm.py} \
    benchmarks/paired/{tp2_corpus.py,tp2-corpus-v1.json,k4.py,vllm_worker.py,default_evidence.py} \
    benchmarks/paired/{saved_evidence.py,cold_cache.py,makespan.py,workload_suite.py} \
    benchmarks/paired/{measure_concurrent.py,identity.py} \
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
    --argjson p0 "${daemon_ports[0]}" --argjson p1 "${daemon_ports[1]}" --argjson uid "$EUID" \
    --slurpfile lm "$session/lmcache-wheel.json" \
    --slurpfile ref "$project/benchmarks/concurrent/gpu2-reference.json" \
    '{protocol:"tp2-paired-v1",protocol_sha256:$protocol,corpus_sha256:$wire,
      campaign_id:$id,mode:$mode,
      native_source:$rev,binary:$binary,binary_sha256:$sha,checkpoint_path:$model,
      checkpoint_sha256:$ref[0].checkpoint_manifest_sha256,vllm_pin:$ref[0].vllm_pin,
      vllm_root:$root,
      client_python:$python,port:$port,
      server_cpus:$server,client_cpus:$client,devices:[$dev0,$dev1],
      daemon_ports:[$p0,$p1],lmcache_uid:$uid,
      lmcache_wire:"SLC3",lmcache_ready_version:"0.5.8+native-v3",lmcache_python:$lm[0].python,
      lmcache_pythonpath:$lm[0].pythonpath}' >"$session/identity.json"
if [[ $session != "$base" ]]; then
    receipts=(identity.json protocol.sha256 corpus.json lmcache-wheel.json
              reservation/{gpu.txt,cpus.txt})
    for receipt in "${receipts[@]}"; do
        cmp -- "$session/$receipt" "$base/$receipt"
    done
fi

daemon_pids=()
lmcache_runtime= lmcache_pid=
cleanup_arm() {
    stop_group
    local pid rank endpoint socket_id
    if [[ -n $lmcache_pid ]]; then stop_session "$lmcache_pid"; fi
    for pid in "${daemon_pids[@]}"; do
        [[ $pid == "$lmcache_pid" ]] || stop_session "$pid"
    done
    if [[ -n $lmcache_runtime ]]; then
        for rank in 0 1; do
            endpoint=$lmcache_runtime/sglang-lmcache-r$rank.sock
            if [[ -S $endpoint && -s $run/daemon-$rank/lmcache-ready.json ]]; then
                socket_id=$(jq -er '.socket_id' "$run/daemon-$rank/lmcache-ready.json")
                "$client_python" -IB "$project/tools/lmcache/sglang_socket.py" \
                    "$endpoint" "$socket_id"
            fi
        done
        local locks=$lmcache_runtime/.sglang-lmcache-locks-$EUID
        if [[ -d $locks && ! -L $locks && -O $locks ]]; then
            find "$locks" -maxdepth 1 -type f -name 'sglang-lmcache-r[01].sock.lock' \
                -uid "$EUID" -perm 0600 -links 1 -delete
            rmdir -- "$locks"
        fi
        rmdir -- "$lmcache_runtime"
    fi
    daemon_pids=(); lmcache_pid=; lmcache_runtime=
}
trap cleanup_arm EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

server_preflight() {
    record_compute "$1/compute-before.csv"
    if [[ ${#daemon_pids[@]} == 0 ]]; then pair_idle; return; fi
    local rank
    for rank in 0 1; do
        lmcache_evidence owners "$1/compute-before.csv" "${devices[rank]}" "${daemon_pids[rank]}"
    done
}

launch_native() {
    local run=$1 rank directory gpu_uuid lmcache_port lmcache_endpoint lmcache_namespace
    local args=("$binary" --port "$port" --tp 2 --max-active 16)
    [[ $run != */native-k4-trace ]] || args+=(--trace-batches)
    pair_idle
    record_compute "$run/compute-before-daemons.csv"
    lmcache_runtime=$(mktemp -d /tmp/sglang-tp2.XXXXXXXX)
    for rank in 0 1; do
        directory=$run/daemon-$rank
        mkdir -p "$directory"/{home,cache,tmp}
        gpu_uuid=${devices[rank]} lmcache_port=${daemon_ports[rank]}
        lmcache_endpoint=$lmcache_runtime/sglang-lmcache-r$rank.sock lmcache_namespace=sglang-r$rank
        launch_lmcache "$directory" --rank "$rank" 3>&- 4>&- 5<&- 6<&- 7>&- 8>&- 9>&-
        daemon_pids+=("$lmcache_pid")
    done
    mkdir "$run"/{home,cache,config,tmp}
    start_group "$run" env -i PATH=/usr/bin:/bin LANG=C.UTF-8 "XDG_RUNTIME_DIR=$lmcache_runtime" \
        "HOME=$run/home" "TMPDIR=$run/tmp" "XDG_CACHE_HOME=$run/cache" \
        "XDG_CONFIG_HOME=$run/config" "CUDA_CACHE_PATH=$run/cuda-cache" \
        "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" CUDA_DEVICE_ORDER=PCI_BUS_ID \
        "LD_LIBRARY_PATH=$compat" "${args[@]}" 3>&- 4>&- 5<&- 6<&- 7>&- 8>&- 9>&-
    server_receipts "$run"
    sha256sum "/proc/$server_pid/exe" >"$run/running-binary.sha256"
    curl --noproxy '*' -fsS "$url/server_info" >"$run/server-info.json"
    "$client_python" -B "$tp2_dir/native.py" "$run" >"$run/ready.json"
}

run_arm() {
    local row=$1 arm=$2 rank lmcache_namespace lmcache_endpoint
    run=$base/$row/$arm
    mkdir -p "$run"
    cd "$run"
    if [[ $arm == vllm-k4 ]]; then
        start_reference "$run" "$row" 3>&- 4>&-
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
        for rank in 0 1; do
            lmcache_pid=${daemon_pids[rank]} lmcache_namespace=sglang-r$rank
            lmcache_endpoint=$lmcache_runtime/sglang-lmcache-r$rank.sock
            record_lmcache_peer "$run/daemon-$rank" -end
        done
        "$client_python" -B "$tp2_dir/native.py" "$run" --completed >"$run/verified-end.json"
    fi
    cleanup_arm
    pair_idle
    for lmcache_port in "${daemon_ports[@]}"; do lmcache_port_free; done
    for number in "$port" "${daemon_ports[@]}"; do
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
