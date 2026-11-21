#!/usr/bin/env bash
set -euo pipefail

tests=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source=$(realpath -- "${1:?Pass the service source directory}")
output=$(realpath -m -- "${2:?Pass the output executable path}")
toolkit="${CONDA_PREFIX:?Activate the managed Pixi default environment}/targets/x86_64-linux"

"${CXX:-c++}" -std=c++17 -O2 -pthread -Wall -Wextra -Wpedantic -Werror \
    -I"$source/include" -isystem "$toolkit/include" \
    "$tests/service_decode_grace.cpp" "$source/apps/server/batch_service.cpp" \
    -Wl,--wrap=_ZNSt6chrono3_V212steady_clock3nowEv \
    -Wl,--wrap=_ZNSt6chrono3_V212system_clock3nowEv \
    -Wl,--wrap=_ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE \
    -Wl,--wrap=pthread_cond_timedwait \
    -o "$output"
CUDA_VISIBLE_DEVICES= timeout 15s "$output"
