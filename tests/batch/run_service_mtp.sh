#!/usr/bin/env bash
set -euo pipefail

tests=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source=$(realpath -- "${1:?Pass the service source directory}")
output=$(realpath -m -- "${2:?Pass the output executable path}")

"${CXX:-c++}" -std=c++17 -O2 -pthread -Wall -Wextra -Werror \
    -I"$source/include" \
    "$tests/service_mtp.cpp" "$source/apps/server/batch_service.cpp" \
    "$source/src/runtime/server/single_device_batch_executor.cpp" \
    "$source"/src/tokenizer/*.cpp "$source/src/json.cpp" "$source/src/utf8.cpp" -o "$output"
CUDA_VISIBLE_DEVICES= "$output"
