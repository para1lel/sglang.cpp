#!/usr/bin/env bash
set -euo pipefail

probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_dir=$(realpath -- "${1:?Pass the native source directory}")
build_dir=$(realpath -- "${2:?Pass the matching Release build directory}")
output=$(realpath -m -- "${3:?Pass the output executable path}")
prefix=${CONDA_PREFIX:?Run in the managed Pixi environment}

sources=()
server_units=(text_request text_completion chat_template chat_output chat_media
              chat_tool_constraints http detail/text_stop)
for name in "${server_units[@]}"; do
    path="$source_dir/apps/server/$name.cpp"
    if [[ -f "$path" ]]; then
        sources+=("$path")
    fi
done

qwen="$source_dir/apps/server/qwen"
sources+=("$qwen"/{qwen_chat_template,qwen_chat_output,qwen_tool_constraints}.cpp
          "$qwen"/{qwen_serving_profile,qwen_tool_syntax}.cpp)

"${CXX:-c++}" -std=c++17 -O2 -pthread -Wall -Wextra -Wpedantic -Werror \
    -ffunction-sections -fdata-sections -I"$source_dir/include" \
    "$probe_dir/tool_probe.cpp" "${sources[@]}" "$source_dir/src/media/multimodal_prompt.cpp" \
    "$source_dir/src/runtime/execution/decoder_options.cpp" \
    "$build_dir/lib/libsglang_media_preprocess.a" "$build_dir/lib/libsglang_tokenizer.a" \
    "$build_dir/lib/libsglang_json.a" -L"$prefix/lib" -Wl,-rpath,"$prefix/lib" \
    -licuuc -licudata -Wl,--gc-sections,--no-undefined -o "$output"
