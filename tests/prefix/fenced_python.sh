#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == -m && ${2:-} == lmcache.v1.multiprocess.server ]]; then
    shift 2
    exec "${SGLANG_TEST_PYTHON:?}" "$(dirname -- "$0")/lmcache_relay.py" "$@"
fi
exec "${SGLANG_TEST_PYTHON:?}" "$@"
