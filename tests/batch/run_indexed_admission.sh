#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/service_build.sh"
build_service_fixture indexed_admission -Wpedantic "$source/src/tokenizer/digest.cpp"
CUDA_VISIBLE_DEVICES= timeout 30s "$output"
