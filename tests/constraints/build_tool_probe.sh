#!/usr/bin/env bash
set -euo pipefail

probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_dir=$(realpath -- "${1:?Pass the native source directory}")
build_dir=$(realpath -- "${2:?Pass the matching Release build directory}")
output=$(realpath -m -- "${3:?Pass the output executable path}")

sources=()
for name in text_request chat_template chat_output chat_media chat_tool_constraints; do
    path="$source_dir/apps/server/$name.cpp"
    if [[ -f "$path" ]]; then
        sources+=("$path")
    fi
done

cuda_headers=()
if [[ -n ${CONDA_PREFIX:-} ]]; then
    cuda_headers=(-isystem "$CONDA_PREFIX/targets/x86_64-linux/include")
fi

"${CXX:-c++}" -std=c++17 -O2 -pthread -Wall -Wextra -Wpedantic -Werror \
    -ffunction-sections -fdata-sections -I"$source_dir/include" "${cuda_headers[@]}" \
    "$probe_dir/tool_probe.cpp" "${sources[@]}" "$source_dir/src/media/multimodal_prompt.cpp" \
    "$build_dir/lib/libsglang_media_preprocess.a" "$build_dir/lib/libsglang_tokenizer.a" \
    "$build_dir/lib/libsglang_json.a" -Wl,--gc-sections,--no-undefined -o "$output"
