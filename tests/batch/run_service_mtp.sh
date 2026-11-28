#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/service_build.sh"
prefix=${CONDA_PREFIX:?Run in the managed Pixi environment}
build_service_fixture service_mtp -isystem "$prefix/include" \
    "$source"/src/tokenizer/*.cpp "$source/src/json.cpp" \
    -L"$prefix/lib" -Wl,-rpath,"$prefix/lib" -licuuc -licudata
CUDA_VISIBLE_DEVICES= "$output"
