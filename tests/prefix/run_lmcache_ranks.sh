#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=$(realpath -m -- "${1:?Pass a directory for the CPU client binary}")
mkdir -p "$output"
toolkit=${CUDA_HOME:-${CONDA_PREFIX:?Use the managed Pixi default environment}}
"${CXX:-c++}" -std=c++17 -O1 -pthread -Iinclude -I"$toolkit/targets/x86_64-linux/include" \
    tests/prefix/lmcache_rank_client.cpp src/runtime/cache/{lmcache_client,lmcache_group}.cpp \
    src/runtime/cache/{lmcache_prefix,prefix_payload}.cpp src/media/multimodal_prompt.cpp \
    src/tokenizer/digest.cpp -o "$output/lmcache-rank-client"
CUDA_VISIBLE_DEVICES='' "${LMCACHE_PYTHON:-python3}" -B tests/prefix/lmcache_rank_roundtrip.py \
    "$output/lmcache-rank-client"
