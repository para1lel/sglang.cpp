#!/usr/bin/env bash
set -euo pipefail

tests=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source=$(realpath -- "${1:?Pass the native loader source directory}")
build=${2:?Pass the matching Release build directory or - for a source build}
output=$(realpath -m -- "${3:?Pass the output executable path}")

sources=("$source/apps/sglang-server.cpp")
for name in batch_service completions flush_cache http text_request text_completion chat_template \
    chat_media chat_output chat_tool_constraints native_generate ollama_discovery ollama_generate \
    parse_function_call prefix_controls anthropic_count score tokenization \
    tokenization_validation tool_literal tool_union; do
    file="$source/apps/server/$name.cpp"
    if [[ -f "$file" ]]; then
        sources+=("$file")
    fi
done

if [[ -f "$source/apps/server/chat_media.cpp" ]]; then
    sources+=("$source/src/media_preprocess.cpp" "$source/src/media_resize.cpp"
             "$source/src/media/multimodal_prompt.cpp")
fi

libraries=()
if [[ "$build" == - ]]; then
    sources+=("$source"/src/tokenizer/*.cpp "$source/src/json.cpp" "$source/src/utf8.cpp")
else
    build=$(realpath -- "$build")
    libraries=("$build/lib/libsglang_tokenizer.a" "$build/lib/libsglang_json.a")
fi

cuda_headers=()
if [[ -n ${CONDA_PREFIX:-} ]]; then
    cuda_headers=(-isystem "$CONDA_PREFIX/targets/x86_64-linux/include")
fi

media_api=()
if [[ $(<"$source/include/sglang/decoder_batch.h") == *admit_multimodal* ]]; then
    media_api=(-DSGLANG_TEST_BATCH_MEDIA)
fi
vision_header="$source/include/sglang/server/vision_completion.h"
if [[ -f "$vision_header" && $(<"$vision_header") == *'generate(const Decoder&'* ]]; then
    media_api+=(-DSGLANG_TEST_SERIAL_MEDIA)
fi

"${CXX:-c++}" -std=c++17 -O2 -pthread \
    -Wall -Wextra -Wpedantic -Werror -ffunction-sections -fdata-sections \
    -I"$source/include" "${cuda_headers[@]}" "${media_api[@]}" "${sources[@]}" \
    "$tests/chat_cpu_decoder.cpp" "${libraries[@]}" \
    -Wl,--wrap=_ZNSt6chrono3_V212steady_clock3nowEv \
    -Wl,--gc-sections,--print-gc-sections,--no-undefined,-Map,"$output.map" \
    -o "$output"
