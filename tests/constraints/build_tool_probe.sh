#!/usr/bin/env bash
set -euo pipefail

probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_dir=$(realpath -- "${1:?Pass the native source directory}")
build_dir=$(realpath -- "${2:?Pass the matching Release build directory}")
output=$(realpath -m -- "${3:?Pass the output executable path}")

sources=()
server_units=(text_request text_completion chat_template chat_output chat_media
              chat_tool_constraints)
for name in "${server_units[@]}"; do
    path="$source_dir/apps/server/$name.cpp"
    if [[ -f "$path" ]]; then
        sources+=("$path")
    fi
done

for name in qwen_chat_template qwen_chat_output qwen_tool_constraints qwen_serving_profile; do
    path="$source_dir/src/runtime/model/$name.cpp"
    if [[ -f "$path" ]]; then
        sources+=("$path")
    fi
done

source_options=()
profile_header="$source_dir/include/sglang/runtime/model/serving_profile.h"
if [[ -f "$profile_header" ]]; then
    source_options=(-DSGLANG_TOOL_PROBE_SERVING_PROFILE)
elif [[ -n ${CONDA_PREFIX:-} ]]; then
    source_options=(-isystem "$CONDA_PREFIX/targets/x86_64-linux/include")
fi

"${CXX:-c++}" -std=c++17 -O2 -pthread -Wall -Wextra -Wpedantic -Werror \
    -ffunction-sections -fdata-sections -I"$source_dir/include" "${source_options[@]}" \
    "$probe_dir/tool_probe.cpp" "${sources[@]}" "$source_dir/src/media/multimodal_prompt.cpp" \
    "$build_dir/lib/libsglang_media_preprocess.a" "$build_dir/lib/libsglang_tokenizer.a" \
    "$build_dir/lib/libsglang_json.a" -Wl,--gc-sections,--no-undefined -o "$output"
