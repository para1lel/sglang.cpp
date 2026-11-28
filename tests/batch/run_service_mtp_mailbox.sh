#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/service_build.sh"
build_service_fixture service_mtp_mailbox -Wpedantic \
    "$tests/fixtures/service_mtp_progress.cpp" \
    "$source/src/tokenizer/digest.cpp"
CUDA_VISIBLE_DEVICES= timeout 30s "$output" "${@:3}"
