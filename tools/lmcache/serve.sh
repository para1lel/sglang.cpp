#!/usr/bin/env bash
set -euo pipefail

case ${XDG_RUNTIME_DIR:-} in
    /*) endpoint=${XDG_RUNTIME_DIR%/}/sglang-lmcache.sock ;;
    *) endpoint=/tmp/sglang-lmcache-$EUID.sock ;;
esac
namespace=sglang
memory_gib=16
port=17655
rank=-1
socket_given=0
namespace_given=0
port_given=0
python=${LMCACHE_PYTHON:-python3}

usage() {
    echo 'Usage: serve.sh [--socket PATH] [--namespace NAME] [--memory-gib N] [--port N] [--rank N]'
    echo 'Set LMCACHE_PYTHON to the interpreter with LMCache 0.5.8 installed.'
    echo 'Expose the native server GPUs to one daemon using CUDA_VISIBLE_DEVICES.'
    echo '--rank N selects legacy single-device naming: socket <default>-rN.sock,'
    echo 'namespace sglang-rN and port 17655+N unless overridden explicitly.'
}

while (($#)); do
    case $1 in
        -h|--help) usage; exit 0 ;;
        --socket|--namespace|--memory-gib|--port|--rank)
            if (($# < 2)); then usage >&2; exit 2; fi
            case $1 in
                --socket) endpoint=$2; socket_given=1 ;;
                --namespace) namespace=$2; namespace_given=1 ;;
                --memory-gib) memory_gib=$2 ;;
                --port) port=$2; port_given=1 ;;
                --rank)
                    case $2 in
                        0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15) rank=$2 ;;
                        *) echo "Invalid LMCache rank: $2 (expected 0-15)" >&2
                           usage >&2; exit 2 ;;
                    esac ;;
            esac
            shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done

if (( rank >= 0 )); then
    if (( !socket_given )); then
        case $endpoint in
            *.sock) endpoint=${endpoint%.sock}-r$rank.sock ;;
            *) endpoint=$endpoint-r$rank ;;
        esac
    fi
    (( namespace_given )) || namespace=sglang-r$rank
    (( port_given )) || port=$((port + rank))
fi

module_dir=$(cd -- "$(dirname -- "$0")" && pwd)
module_config=$("$python" - "$endpoint" "$namespace" "$memory_gib" "$rank" <<'PY'
import json
import math
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version

path, namespace, memory, rank = sys.argv[1:]
try:
    installed = version("lmcache")
except PackageNotFoundError:
    installed = "missing"
if installed != "0.5.8":
    sys.exit(f"LMCache 0.5.8 is required, found {installed}")
if not os.path.isabs(path) or len(os.fsencode(path)) >= 108:
    sys.exit("LMCache socket must be an absolute path shorter than 108 bytes")
if rank != "-1" and not (rank.isdigit() and int(rank) < 16):
    sys.exit("Invalid LMCache rank")
if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", namespace):
    sys.exit("Invalid LMCache namespace")
try:
    size = float(memory)
except ValueError:
    sys.exit("Invalid LMCache capacity")
if not math.isfinite(size) or size <= 0:
    sys.exit("LMCache capacity must be finite and positive")
config = {"socket": path, "namespace": namespace}
if rank != "-1":
    config["single_device"] = True
print(json.dumps({"module_path": "sglang_lmcache", "config": config}))
PY
)

umask 077
export PYTHONPATH="$module_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
unset LMCACHE_ISOLATED_IPC LMCACHE_USE_VMM_API
export LMCACHE_TRACK_USAGE=false
unset LMCACHE_API_KEY_FILE LMCACHE_HTTP_API_KEY LMCACHE_HTTP_API_KEY_FILE
LMCACHE_API_KEY=$("$python" -c 'import secrets; print(secrets.token_urlsafe(32))')
export LMCACHE_API_KEY
exec "$python" -m lmcache.v1.multiprocess.server --host 127.0.0.1 --port "$port" \
    --supported-transfer-mode lmcache_driven --no-run-script-api-enabled --disable-metrics \
    --l1-size-gb "$memory_gib" --l1-init-size-gb "$memory_gib" --eviction-policy LRU \
    --l1-read-ttl-seconds 60 --l1-write-ttl-seconds 60 --server-module "$module_config"
