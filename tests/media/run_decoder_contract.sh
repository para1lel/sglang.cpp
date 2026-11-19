#!/bin/bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
build=$(realpath "${1:?pass an existing release build directory}")
output=$(realpath -m "${2:?pass a disposable output executable path}")
checkpoint=${3:?pass the Qwen3.8-27B-FP8 checkpoint}
toolkit=${CUDA_TOOLKIT_ROOT:-}
if [[ -z $toolkit ]]; then
    toolkit=${CONDA_PREFIX:?run in the project Pixi environment}/targets/x86_64-linux
fi
lib=$build/lib
symbols=$(nm --defined-only "$lib/libsglang_cuda.a")
wrap=(-Wl,--wrap=cudaMallocFromPoolAsync -Wl,--wrap=cudaStreamBeginCapture)
names=(mrope rope append attend decode)
patterns=('mrope_gqa_preprocessE' 'text_gqa_preprocessE' 'mxfp8_kv_appendE'
          'mxfp8_gqa_attentionE' 'exact_decode_gqa_attentionENS_12Mxfp8KvCache.*TableEi')
for index in "${!names[@]}"; do
    symbol=$(awk -v pattern="${patterns[index]}" \
        '$2 == "T" && $3 ~ pattern {print $3}' <<<"$symbols")
    [[ -n $symbol && $symbol != *$'\n'* ]]
    name=${names[index]}
    wrap+=("-Wl,--wrap=$symbol" "-Dreal_$name=__real_$symbol" "-Dtraced_$name=__wrap_$symbol")
done
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -I"$repo/include" -I"$toolkit/include" \
    "$repo/tests/media/decoder_contract.cpp" "$repo/tests/media/decoder_trace.cpp" \
    -L"$lib" -L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib" "${wrap[@]}" \
    -lsglang_decoder -lsglang_checkpoint -lsglang_json -lsglang_cuda \
    -lsglang_prefill_fp8_sm100 -lsglang_tokenizer -lcudart -o "$output"
if [[ ${MEDIA_CONTRACT_BUILD_ONLY:-0} != 1 ]]; then
    "$output" "$checkpoint"
fi
