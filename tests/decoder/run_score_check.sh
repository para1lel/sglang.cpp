#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: bash tests/decoder/run_score_check.sh TOKENIZER_JSON OUTPUT" >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
tokenizer=$(realpath -- "$1")
output=$(realpath -m -- "$2")
toolkit="${CUDA_HOME:-${CONDA_PREFIX:?Run in the managed Pixi environment}}/targets/x86_64-linux"
mkdir -p "$output"
cd "$root"
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -ffunction-sections -fdata-sections \
    -Iinclude -isystem "$toolkit/include" tests/decoder/score_check.cpp apps/server/score.cpp \
    src/text/label_logprobs.cpp src/runtime/execution/decoder_options.cpp \
    src/json.cpp src/utf8.cpp src/tokenizer/*.cpp -Wl,--gc-sections -o "$output/score-check"
"$output/score-check" "$tokenizer"
