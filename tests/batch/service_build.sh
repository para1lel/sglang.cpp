#!/usr/bin/env bash

tests=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source=$(realpath -- "${1:?Pass the service source directory}")
output=$(realpath -m -- "${2:?Pass the output executable path}")
sdk=${CUDA_HOME:-${CONDA_PREFIX:?Use the managed Pixi default environment}}

service_sources=(
    "$tests/service_decoder.cpp"
    "$source"/apps/server/{batch_service,batch_cache}.cpp
    "$source"/apps/server/detail/{batch_service_cache,batch_state,batch_trace,text_stop}.cpp
    "$source"/src/runtime/cache/{lmcache_client,lmcache_group,lmcache_prefix}.cpp
    "$source"/src/runtime/cache/{prefix_cache_layout,prefix_payload}.cpp
    "$source"/src/runtime/execution/{batch_admission_order,decoder_options}.cpp
    "$source/src/runtime/model/qwen_text_model.cpp"
    "$source"/src/runtime/server/{batch_executor,batch_restore_join,cache_identity}.cpp
    "$source/src/runtime/server/single_device_batch_executor.cpp"
    "$source/src/media/multimodal_prompt.cpp" "$source/src/utf8.cpp"
)

build_service_fixture() {
    local fixture=$1
    shift
    "${CXX:-c++}" -std=c++17 -O2 -pthread -Wall -Wextra -Werror \
        -I"$source/include" -isystem "$sdk/targets/x86_64-linux/include" \
        "$tests/$fixture.cpp" "${service_sources[@]}" "$@" -o "$output"
}
