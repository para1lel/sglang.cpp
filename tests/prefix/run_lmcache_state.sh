#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
build=$(realpath -- "${1:?Pass the release build directory}")
checkpoint=$(realpath -- "${2:?Pass the checkpoint directory}")
output=$(realpath -m -- "${3:?Pass a fresh evidence directory}")
gpu=${GPU_UUID:?Set the reserved B200 UUID}
prefix=${CONDA_PREFIX:?Use the managed Pixi default environment}
toolkit=$prefix/targets/x86_64-linux
export LMCACHE_SOCKET=${LMCACHE_SOCKET:?Start the LMCache daemon on the same reserved GPU}
mkdir "$output"

append=_ZN6sglang15mxfp8_kv_appendENS_12Mxfp8KvCacheENS_14Mxfp8PageTableEii
append+=PK13__nv_bfloat16S4_P11CUstream_st
for name in state graph; do
    wrap=()
    if [[ $name == state ]]; then
        wrap=(-Wl,--wrap="$append" -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap=cudaFreeAsync
              -Wl,--wrap=cudaFree -Wl,--wrap=cudaMalloc -Wl,--wrap=cudaEventRecord)
    fi
    "${CXX:-c++}" -std=c++17 -O3 -pthread -Iinclude -I"$toolkit/include" \
        "tests/prefix/lmcache_$name.cpp" src/runtime/cache/lmcache_client.cpp \
        src/runtime/cache/lmcache_prefix.cpp \
        -L"$build/lib" -L"$toolkit/lib" -L"$prefix/lib64" \
        -Wl,-rpath,"$prefix/lib" -Wl,-rpath,"$toolkit/lib" -Wl,-rpath,"$prefix/lib64" \
        "${wrap[@]}" -lsglang_decoder -lsglang_checkpoint -lsglang_cuda \
        -lsglang_prefill_fp8_sm100 -lsglang_tokenizer -lsglang_json \
        -lcudart -lcudadevrt -lcudart_static -lrt -ldl -o "$output/lmcache-$name"
    CUDA_VISIBLE_DEVICES="$gpu" "$output/lmcache-$name" "$checkpoint" | tee "$output/$name.log"
done
