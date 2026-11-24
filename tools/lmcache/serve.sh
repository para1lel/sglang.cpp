#!/usr/bin/env bash
set -euo pipefail

case ${XDG_RUNTIME_DIR:-} in
    /*) endpoint=${XDG_RUNTIME_DIR%/}/sglang-lmcache.sock ;;
    *) endpoint=/tmp/sglang-lmcache-$EUID.sock ;;
esac
namespace=sglang
memory_gib=16
port=17655
python=${LMCACHE_PYTHON:-python3}

usage() {
    echo 'Usage: serve.sh [--socket PATH] [--namespace NAME] [--memory-gib N] [--port N]'
    echo 'Set LMCACHE_PYTHON to the interpreter with LMCache 0.5.8 installed.'
}

while (($#)); do
    case $1 in
        -h|--help) usage; exit 0 ;;
        --socket|--namespace|--memory-gib|--port)
            if (($# < 2)); then usage >&2; exit 2; fi
            case $1 in
                --socket) endpoint=$2 ;;
                --namespace) namespace=$2 ;;
                --memory-gib) memory_gib=$2 ;;
                --port) port=$2 ;;
            esac
            shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done

module_dir=$(cd -- "$(dirname -- "$0")" && pwd)
module_config=$("$python" - "$endpoint" "$namespace" "$memory_gib" <<'PY'
import json
import math
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version

path, namespace, memory = sys.argv[1:]
try:
    installed = version("lmcache")
except PackageNotFoundError:
    installed = "missing"
if installed != "0.5.8":
    sys.exit(f"LMCache 0.5.8 is required, found {installed}")
if not os.path.isabs(path) or len(os.fsencode(path)) >= 108:
    sys.exit("LMCache socket must be an absolute path shorter than 108 bytes")
if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", namespace):
    sys.exit("Invalid LMCache namespace")
try:
    size = float(memory)
except ValueError:
    sys.exit("Invalid LMCache capacity")
if not math.isfinite(size) or size <= 0:
    sys.exit("LMCache capacity must be finite and positive")
print(json.dumps({"module_path": "sglang_lmcache", "config": {
    "socket": path, "namespace": namespace,
}}))
PY
)

umask 077
export PYTHONPATH="$module_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=
export LMCACHE_TRACK_USAGE=false
unset LMCACHE_API_KEY_FILE LMCACHE_HTTP_API_KEY LMCACHE_HTTP_API_KEY_FILE
LMCACHE_API_KEY=$("$python" -c 'import secrets; print(secrets.token_urlsafe(32))')
export LMCACHE_API_KEY
exec "$python" -m lmcache.v1.multiprocess.server --host 127.0.0.1 --port "$port" \
    --supported-transfer-mode lmcache_driven --no-run-script-api-enabled --disable-metrics \
    --l1-size-gb "$memory_gib" --l1-init-size-gb 0 --eviction-policy LRU \
    --l1-read-ttl-seconds 60 --l1-write-ttl-seconds 60 --server-module "$module_config"
