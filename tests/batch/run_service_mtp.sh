#!/usr/bin/env bash
set -euo pipefail

tests=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source=$(realpath -- "${1:?Pass the service source directory}")
output=$(realpath -m -- "${2:?Pass the output executable path}")

sdk=${CUDA_HOME:-${CONDA_PREFIX:?Use the managed Pixi default environment}}
source "$tests/service_sources.sh"
"${CXX:-c++}" -std=c++17 -O2 -pthread -Wall -Wextra -Werror \
    -I"$source/include" -isystem "$sdk/targets/x86_64-linux/include" \
    "$tests/service_mtp.cpp" "${service_sources[@]}" \
    "$source"/src/tokenizer/*.cpp "$source/src/json.cpp" -o "$output"
CUDA_VISIBLE_DEVICES= "$output"
