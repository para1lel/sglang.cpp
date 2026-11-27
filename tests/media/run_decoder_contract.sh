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
nccl=${NCCL_ROOT:-}
if [[ -z $nccl ]]; then
    shopt -s nullglob
    prefix=${CONDA_PREFIX:?set NCCL_ROOT or use the Pixi environment that built Release}
    candidates=("$prefix"/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so.2)
    if (( ${#candidates[@]} == 1 )); then
        nccl=${candidates[0]%/lib/libnccl.so.2}
    fi
fi
if [[ -z $nccl || ! -f $nccl/lib/libnccl.so.2 ]]; then
    echo "Set NCCL_ROOT to the Release NCCL package directory containing lib/libnccl.so.2" >&2
    exit 2
fi
lib=$build/lib
symbols=$(nm --defined-only "$lib/libsglang_cuda.a")
wrap=(-Wl,--wrap=cudaMallocFromPoolAsync -Wl,--wrap=cudaStreamBeginCapture)
names=(mrope mrope_model rope append attend attend_tensor decode decode_vector seed_keys seed_norm)
patterns=('mrope_gqa_preprocessE.*P11CUstream_st$'
          'mrope_gqa_preprocessE.*GqaGeometryE$' 'text_gqa_preprocessE.*GqaGeometryE$'
          'mxfp8_kv_appendE.*GqaGeometryE$' 'mxfp8_gqa_attentionE.*GqaGeometryE$'
          'tensor_core_multirow_prefill_gqaENS_12Mxfp8KvCache.*PairedPrefillScratch'
          'exact_decode_gqa_attentionENS_12Mxfp8KvCache.*TableEi.*P11CUstream_stmb$'
          'vector_decode_gqa_attentionENS_12Mxfp8KvCache.*TableEi'
          'mrope_gqa_preprocess_keysE.*P11CUstream_st$' 'text_rms_norm_pair_rowsE')
for index in "${!names[@]}"; do
    symbol=$(awk -v pattern="${patterns[index]}" \
        '$2 == "T" && $3 ~ pattern {print $3}' <<<"$symbols")
    if [[ -z $symbol || $symbol == *$'\n'* ]]; then
        echo "Expected one Release symbol for ${names[index]}" >&2
        exit 2
    fi
    name=${names[index]}
    wrap+=("-Wl,--wrap=$symbol" "-Dreal_$name=__real_$symbol" "-Dtraced_$name=__wrap_$symbol")
done
"${CXX:-c++}" -std=c++17 -O2 -pthread -Wall -Wextra -Werror -I"$repo/include" -I"$toolkit/include" \
    "$repo/tests/media/decoder_contract.cpp" "$repo/tests/media/decoder_trace.cpp" \
    "$repo/tests/media/decoder_fixture.cpp" \
    -L"$lib" -L"$toolkit/lib" -L"$toolkit/lib/stubs" -Wl,-rpath,"$toolkit/lib" "${wrap[@]}" \
    -Wl,--start-group -lsglang_decoder -lsglang_checkpoint -lsglang_json -lsglang_cuda \
    -lsglang_prefill_fp8_sm100 -lsglang_tokenizer -lsglang_mnnvl_ar \
    -lsglang_cuda_support -Wl,--end-group \
    -Wl,-rpath,"$nccl/lib" "$nccl/lib/libnccl.so.2" -lcuda -lcudart -o "$output"
if [[ ${MEDIA_CONTRACT_BUILD_ONLY:-0} != 1 ]]; then
    "$output" "$checkpoint"
fi
