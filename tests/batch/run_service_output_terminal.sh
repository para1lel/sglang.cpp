#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/service_build.sh"
build_service_fixture service_output_terminal -Wpedantic \
    "$source"/src/tokenizer/*.cpp "$source/src/json.cpp"
CUDA_VISIBLE_DEVICES= timeout 25s "$output"
