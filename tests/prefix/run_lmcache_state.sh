#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
build=$(realpath -- "${1:?Pass the release build directory}")
checkpoint=$(realpath -- "${2:?Pass the checkpoint directory}")
output=$(realpath -m -- "${3:?Pass a fresh evidence directory}")
gpu=${GPU_UUID:?Set the reserved B200 UUID}
mode=${4:-}
if [[ -n $mode && $mode != --serving-release-only ]]; then
    printf 'Unknown state probe mode: %s\n' "$mode" >&2
    exit 2
fi
prefix=${CONDA_PREFIX:?Use the managed Pixi default environment}
toolkit=$prefix/targets/x86_64-linux
nccl=$(python -I -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl")')
if [[ -z $mode ]]; then
    export LMCACHE_SOCKET=${LMCACHE_SOCKET:?Start the LMCache daemon on the same reserved GPU}
fi
mkdir "$output"

append=_ZN6sglang15mxfp8_kv_appendENS_12Mxfp8KvCacheENS_14Mxfp8PageTableEii
append+=PK13__nv_bfloat16S4_P11CUstream_st
names=(state graph release)
if [[ $mode == --serving-release-only ]]; then
    names=(release)
fi
for name in "${names[@]}"; do
    wrap=()
    extra_sources=()
    args=("$checkpoint")
    if [[ $name == state ]]; then
        wrap=(-Wl,--wrap="$append" -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap=cudaFree
              -Wl,--wrap=cudaMalloc -Wl,--wrap=cudaEventRecord)
        extra_sources=(apps/server/batch_cache.cpp
                       src/runtime/server/{batch_restore_join,cache_identity}.cpp)
    elif [[ $name == release ]]; then
        wrap=(-Wl,--wrap=cudaEventRecord -Wl,--wrap=cudaEventQuery -Wl,--wrap=cudaFree
              -Wl,--wrap=cudaIpcGetMemHandle)
        args=()
        if [[ $mode == --serving-release-only ]]; then
            args=("$checkpoint")
        fi
    else
        extra_sources=(apps/server/detail/batch_state.cpp)
    fi
    "${CXX:-c++}" -std=c++17 -O3 -pthread -ffunction-sections -fdata-sections \
        -Iinclude -I"$toolkit/include" \
        "tests/prefix/lmcache_$name.cpp" tests/prefix/prefix_fixture.cpp \
        src/runtime/cache/{lmcache_client,lmcache_group,lmcache_prefix}.cpp "${extra_sources[@]}" \
        -L"$build/lib" -L"$toolkit/lib" -L"$toolkit/lib/stubs" -L"$prefix/lib64" \
        -Wl,-rpath,"$prefix/lib" -Wl,-rpath,"$toolkit/lib" -Wl,-rpath,"$prefix/lib64" \
        "${wrap[@]}" -Wl,--start-group -lsglang_decoder -lsglang_checkpoint -lsglang_cuda \
        -lsglang_prefill_fp8_sm100 -lsglang_tokenizer -lsglang_json -lsglang_mnnvl_ar \
        -lsglang_cuda_support -Wl,--end-group -Wl,--gc-sections \
        -Wl,-rpath,"$nccl/lib" "$nccl/lib/libnccl.so.2" \
        -lcuda -lcudart -lcudadevrt -lcudart_static -lrt -ldl -o "$output/lmcache-$name"
    CUDA_VISIBLE_DEVICES="$gpu" timeout --verbose --kill-after=5s "${LMCACHE_TEST_TIMEOUT:-5m}" \
        "$output/lmcache-$name" "${args[@]}" 2>&1 | tee "$output/$name.log"
done
