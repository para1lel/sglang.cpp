#!/usr/bin/env bash
set -euo pipefail

tests=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source=$(realpath -- "${1:?Pass the native loader source directory}")
build=$(realpath -- "${2:?Pass the matching Release build directory}")
output=$(realpath -m -- "${3:?Pass the output executable path}")
prefix=${CONDA_PREFIX:?Activate the managed Pixi default environment}

sources=("$source/apps/sglang-server.cpp")
for name in batch_service completions http text_request text_completion chat_template \
    chat_media prefix_controls; do
    file="$source/apps/server/$name.cpp"
    if [[ -f "$file" ]]; then
        sources+=("$file")
    fi
done

sources+=("$source/src/media_preprocess.cpp" "$source/src/media_resize.cpp"
         "$source/src/media/multimodal_prompt.cpp")

"${CXX:-c++}" -std=c++17 -O2 -pthread \
    -Wall -Wextra -Wpedantic -Werror -ffunction-sections -fdata-sections \
    -I"$source/include" -isystem "$prefix/targets/x86_64-linux/include" "${sources[@]}" \
    "$tests/chat_cpu_decoder.cpp" "$build/lib/libsglang_tokenizer.a" \
    "$build/lib/libsglang_json.a" \
    -Wl,--gc-sections,--print-gc-sections,--no-undefined,-Map,"$output.map" \
    -o "$output"
