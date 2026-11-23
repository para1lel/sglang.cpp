#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
build=$(realpath -- "${1:?Pass the release build directory}")
checkpoint=$(realpath -- "${2:?Pass the checkpoint directory}")
long_ids=$(realpath -- "${3:?Pass the frozen 8192-token ID file}")
output=$(realpath -m -- "${4:?Pass a fresh evidence directory}")
gpu=${GPU_UUID:?Set the reserved B200 UUID}
toolkit="${CONDA_PREFIX:?Use the managed Pixi default environment}/targets/x86_64-linux"
mkdir "$output"

python -B - "$output" <<'PY'
import json
from pathlib import Path
import sys

workload = json.loads(Path("benchmarks/streaming/W1.json").read_text())
ids = workload["prompt_token_ids"]
assert len(ids) == workload["output_tokens"] == 128
(Path(sys.argv[1]) / "w1.ids").write_text(" ".join(map(str, ids)) + "\n")
PY

"${CXX:-c++}" -std=c++17 -O3 -pthread -fno-access-control -Iinclude -Isrc \
    -I"$toolkit/include" tests/prefix/tensor_state.cpp -L"$build/lib" -L"$toolkit/lib" \
    -Wl,-rpath,"$toolkit/lib" -lsglang_decoder -lsglang_checkpoint -lsglang_json \
    -lsglang_cuda -lsglang_prefill_fp8_sm100 -lsglang_tokenizer -lcudart \
    -o "$output/tensor-state"
CUDA_VISIBLE_DEVICES="$gpu" "$output/tensor-state" "$checkpoint" "$long_ids" \
    "$output/w1.ids" "$output/logits.bf16" | tee "$output/state.log"
