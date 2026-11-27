#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/service_build.sh"
build_service_fixture service_mtp "$source"/src/tokenizer/*.cpp "$source/src/json.cpp"
CUDA_VISIBLE_DEVICES= "$output"
