#!/usr/bin/env bash
set -euo pipefail

tests=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source=$(realpath -- "${1:?Pass the candidate or Unit16 source directory}")
build=$(realpath -- "${2:?Pass the matching Release build directory}")
output=$(realpath -m -- "${3:?Pass the output executable path}")
prefix=${CONDA_PREFIX:?Activate the managed Pixi default environment}
llvm="$prefix/bin/llvm-config"
llvm_include=$("$llvm" --includedir)
llvm_lib=$("$llvm" --libdir)
read -ra llvm_links <<< "$("$llvm" --link-shared --libs support)"
read -ra system_links <<< "$("$llvm" --link-shared --system-libs)"

sources=("$source/apps/sglang-server.cpp")
for name in batch_service completions http text_request text_completion chat_template; do
    file="$source/apps/server/$name.cpp"
    if [[ -f "$file" ]]; then
        sources+=("$file")
    fi
done

"${CXX:?Activate the managed Pixi default environment}" -std=c++17 -O2 -pthread \
    -Wall -Wextra -Wpedantic -Werror -ffunction-sections -fdata-sections \
    -I"$source/include" -isystem "$llvm_include" "${sources[@]}" \
    "$tests/chat_cpu_decoder.cpp" "$build/lib/libsglang_tokenizer.a" \
    "$build/lib/libsglang_json.a" -L"$llvm_lib" -Wl,-rpath,"$llvm_lib" \
    -Wl,--gc-sections,--print-gc-sections,--no-undefined,-Map,"$output.map" \
    "${llvm_links[@]}" "${system_links[@]}" -o "$output"
