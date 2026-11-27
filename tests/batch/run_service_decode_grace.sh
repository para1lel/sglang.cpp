#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/service_build.sh"
build_service_fixture service_decode_grace -Wpedantic "$source/src/tokenizer/digest.cpp" \
    -Wl,--wrap=_ZNSt6chrono3_V212steady_clock3nowEv \
    -Wl,--wrap=_ZNSt6chrono3_V212system_clock3nowEv \
    -Wl,--wrap=_ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE \
    -Wl,--wrap=pthread_cond_timedwait
CUDA_VISIBLE_DEVICES= timeout 15s "$output"
