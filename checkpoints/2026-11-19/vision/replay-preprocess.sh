#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    printf 'Usage: bash %s PROJECT_REPO CHECKPOINT VLLM_ENV SGLANG_ARCHIVE NEW_RUN_DIR\n' "$0" >&2
    exit 2
fi

bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
project=$(realpath "$1")
checkpoint=$(realpath "$2")
vllm=$(realpath "$3")
upstream=$(realpath "$4")
mkdir "$5"
run=$(realpath "$5")
python3 "$bundle/verify.py" --repo "$project" --output "$run/trees" vision
vision="$run/trees/vision/vision"
mkdir -p "$run"/{oracle/sources,fixtures,tmp}

revision=f04d886bf79aa3d6237cbb12a3311838eb9f9bd8
prefix="sglang-$revision/python/sglang/srt"
for source in \
    multimodal/processors/base_processor.py \
    managers/mm_utils.py \
    layers/rotary_embedding/mrope.py \
    multimodal/mrope_rope_index.py \
    models/qwen3_5.py \
    models/qwen3_vl.py \
    multimodal/processors/qwen_vl.py \
    multimodal/qwen_vl_media_processing.py
do
    tar -xOzf "$upstream" "$prefix/$source" > "$run/oracle/sources/${source##*/}"
done
cp "$vision/preprocess/oracle.py" "$run/oracle/"

export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR="$run/tmp"
"${CXX:-c++}" -std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Werror \
    "$vision/preprocess/preprocess.cpp" -o "$run/preprocess"

python="$vllm/venv/bin/python"
"$python" -B -c '
import runpy
import sys
from pathlib import Path

script, project, *args = sys.argv[1:]
main = runpy.run_path(script)["main"]
main.__globals__["ROOT"] = Path(project)
sys.argv = [script, *args]
main()
' "$run/oracle/oracle.py" "$project" "$run/fixtures" "$run/preprocess" "$checkpoint" "$vllm" \
    > "$run/preprocess.jsonl"
"$python" -B "$vision/encoder/prepare.py" "$checkpoint" "$run/fixtures" "$run/manifest"
cd "$run"
"$python" -B -c '
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
for path, digest in manifest["vision_cpu_replay"]["generated_sha256"].items():
    print(f"{digest}  {path}")
' "$bundle/manifest.json" | sha256sum --strict -c -
