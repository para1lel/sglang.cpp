#!/usr/bin/env bash
set -euo pipefail

tests=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source=$(realpath -- "${1:?Pass the service source directory}")
output=$(realpath -m -- "${2:?Pass the output executable path}")
toolkit="${CONDA_PREFIX:?Activate the managed Pixi default environment}/targets/x86_64-linux"

"${CXX:-c++}" -std=c++17 -O2 -pthread -Wall -Wextra -Wpedantic -Werror \
    -I"$source/include" -isystem "$toolkit/include" \
    "$tests/service_mtp_mailbox.cpp" "$source/apps/server/batch_service.cpp" \
    "$source/src/runtime/server/single_device_batch_executor.cpp" \
    "$source/src/utf8.cpp" -o "$output"
CUDA_VISIBLE_DEVICES= timeout 30s "$output" "${@:3}"
