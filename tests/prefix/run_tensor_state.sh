#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
build=$(realpath -- "${1:?Pass the release build directory}")
checkpoint=$(realpath -- "${2:?Pass the checkpoint directory}")
fixtures=$(realpath -- "${3:?Pass the frozen W3 variant fixtures directory}")
output=$(realpath -m -- "${4:?Pass a fresh evidence directory}")
gpu=${GPU_UUID:?Set the reserved B200 UUID}
toolkit="${CONDA_PREFIX:?Use the managed Pixi default environment}/targets/x86_64-linux"
mkdir "$output"

python -B - "$fixtures" "$output" <<'PY'
from pathlib import Path
import sys

from benchmarks.variants.fixtures import load_requests
from benchmarks.variants.verify import verify_fixtures

fixtures, output = map(Path, sys.argv[1:])
verify_fixtures(fixtures)
for name, mode in [("lane0", "prefix"), ("prime", "prime")]:
    _, _, case = load_requests(fixtures, mode)[0]
    (output / f"{name}.ids").write_text(" ".join(map(str, case["prompt_token_ids"])) + "\n")
PY

"${CXX:-c++}" -std=c++17 -O3 -pthread -fno-access-control -Iinclude -Isrc \
    -I"$toolkit/include" tests/prefix/tensor_state.cpp -L"$build/lib" -L"$toolkit/lib" \
    -Wl,-rpath,"$toolkit/lib" -lsglang_decoder -lsglang_checkpoint -lsglang_json \
    -lsglang_cuda -lsglang_prefill_fp8_sm100 -lsglang_tokenizer -lcudart \
    -o "$output/tensor-state"
CUDA_VISIBLE_DEVICES="$gpu" "$output/tensor-state" "$checkpoint" "$output/lane0.ids" \
    "$output/prime.ids" "$output/logits.bf16" | tee "$output/state.log"
