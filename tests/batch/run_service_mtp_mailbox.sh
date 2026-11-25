#!/usr/bin/env bash
set -euo pipefail

tests=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source=$(realpath -- "${1:?Pass the service source directory}")
output=$(realpath -m -- "${2:?Pass the output executable path}")

"${CXX:-c++}" -std=c++17 -O2 -pthread -Wall -Wextra -Wpedantic -Werror \
    -I"$source/include" \
    "$tests/service_mtp_mailbox.cpp" "$source/apps/server/batch_service.cpp" \
    "$source"/apps/server/{batch_cache,batch_service_cache}.cpp \
    "$source"/src/runtime/cache/{lmcache_client,lmcache_prefix,prefix_payload}.cpp \
    "$source/src/media/multimodal_prompt.cpp" "$source/src/tokenizer/digest.cpp" \
    "$source/src/runtime/server/single_device_batch_executor.cpp" \
    "$source/src/utf8.cpp" -o "$output"
CUDA_VISIBLE_DEVICES= timeout 30s "$output" "${@:3}"
