#!/usr/bin/env bash

# The daemon is owned by the native arm and has its own process group.
lmcache_evidence() {
    PYTHONPATH="$project" "$client_python" -B "$paired_dir/lmcache_evidence.py" "$@"
}

check_compute_owners() {
    lmcache_evidence owners "$1" "$gpu_uuid" "${@:2}"
}

lmcache_port_free() {
    local listeners
    listeners=$(ss -Hltn "sport = :$lmcache_port") || return
    [[ -z $listeners ]] || { echo "Port $lmcache_port already has a listener." >&2; return 75; }
}

resolve_lmcache_pythonpath() {
    "$client_python" -IB -c '
import os
import sys
path = sys.argv[1]
print(os.pathsep.join(os.path.realpath(p) for p in path.split(os.pathsep)) if path else "")
' "$1"
}

record_lmcache_peer() {
    local run=$1 suffix=${2:-} options=()
    if [[ -n $suffix ]]; then options=(--once); fi
    lmcache_evidence ready "$lmcache_runtime/sglang-lmcache.sock" "$lmcache_pid" \
        --wire SLC3 "${options[@]}" >"$run/lmcache-ready$suffix.json" || return
    cat "/proc/$lmcache_pid/cmdline" >"$run/lmcache-live-argv$suffix.bin"
}

start_lmcache() {
    local run=$1
    record_compute "$run/compute-before-daemon.csv" || return
    check_compute_owners "$run/compute-before-daemon.csv" || return
    lmcache_runtime=$(mktemp -d /tmp/sglang-paired.XXXXXXXX)
    local -a daemon=(env -i PATH=/usr/bin:/bin LANG=C.UTF-8
        "HOME=$run/home" "TMPDIR=$run/tmp" "XDG_CACHE_HOME=$run/cache"
        "LMCACHE_PYTHON=$lmcache_python" "PYTHONPATH=$lmcache_pythonpath"
        PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1
        "LD_LIBRARY_PATH=$compat" "CUDA_VISIBLE_DEVICES=$gpu_uuid" CUDA_DEVICE_ORDER=PCI_BUS_ID
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        numactl --localalloc taskset --cpu-list "$client_cpus"
        bash "$lmcache_launcher" --socket "$lmcache_runtime/sglang-lmcache.sock"
        --namespace sglang --memory-gib 64 --port "$lmcache_port")
    write_command "$run/lmcache-command.sh" "${daemon[@]}"
    sha256sum "${lmcache_sources[@]}" >"$run/lmcache-source.sha256" || return
    launch_session "$run/lmcache-pid.txt" "${daemon[@]}" >"$run/lmcache.log" 2>&1 9<&- &
    lmcache_pid=$!
    record_lmcache_peer "$run" || return
    lmcache_socket_id=$(jq -er '.socket_id' "$run/lmcache-ready.json") || return
    for ((attempt = 0; attempt < 100; ++attempt)); do
        ss -Hlnpt "sport = :$lmcache_port" >"$run/lmcache-listener.txt" || return 1
        [[ ! -s $run/lmcache-listener.txt ]] || break
        kill -0 "$lmcache_pid" 2>/dev/null || return 1
        sleep .05
    done
    awk -v pid="pid=$lmcache_pid," -v address="127.0.0.1:$lmcache_port" '
        $4 == address && index($0, pid) {owned++} END {exit !(NR == 1 && owned == 1)}' \
        "$run/lmcache-listener.txt" || return 1
    tr '\0' '\n' <"/proc/$lmcache_pid/environ" |
        LC_ALL=C grep -E '^CUDA_(DEVICE_ORDER|VISIBLE_DEVICES)=' | LC_ALL=C sort \
        >"$run/lmcache-cuda-environment.txt" || return
    tr '\0' '\n' <"/proc/$lmcache_pid/environ" |
        LC_ALL=C grep -E '^(LMCACHE_PYTHON|PYTHONPATH)=' | LC_ALL=C sort \
        >"$run/lmcache-python-environment.txt"
}

stop_lmcache() {
    if [[ -n ${lmcache_pid:-} ]]; then
        stop_session "$lmcache_pid"
        if kill -0 -- "-$lmcache_pid" 2>/dev/null; then
            echo "LMCache process group $lmcache_pid did not stop." >&2
            return 2
        fi
        lmcache_pid=
    fi
    if [[ -n ${lmcache_runtime:-} ]]; then
        if [[ -n ${lmcache_socket_id:-} ]]; then
            "$client_python" -IB "$project/tools/lmcache/sglang_socket.py" \
                "$lmcache_runtime/sglang-lmcache.sock" "$lmcache_socket_id" || return 2
            lmcache_socket_id=
        fi
        rmdir -- "$lmcache_runtime" || return 2
        lmcache_runtime=
    fi
}
