#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash benchmarks/paired/run_vllm.sh GPU_INDEX NEW_OUTPUT_DIR [GROUP ...]"
    echo "Groups: W1-W2 W3 W4 W5 W6 (default: all in this order)"
    echo "Set PAIRED_CHECKPOINT_MANIFEST to the published model file list."
    echo "PAIRED_MAX_NUM_BATCHED_TOKENS defaults to 8192."
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# < 2 )); then
    usage >&2
    exit 2
fi
index=$1
target=$2
shift 2
groups=("$@")
if (( ${#groups[@]} == 0 )); then
    groups=(W1-W2 W3 W4 W5 W6)
fi
declare -A selected_groups=()
for group in "${groups[@]}"; do
    case "$group" in
        W1-W2|W3|W4|W5|W6) ;;
        *) usage >&2; exit 2 ;;
    esac
    if [[ ${selected_groups[$group]+present} ]]; then
        echo "Duplicate benchmark group: $group" >&2
        exit 2
    fi
    selected_groups[$group]=1
done

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
validate_run_settings
if (( token_budget < 32 )); then
    echo "vLLM requires at least 32 target rows for 32 configured sequences." >&2
    exit 2
fi
manifest=${PAIRED_CHECKPOINT_MANIFEST:-}
if [[ ! -f $manifest ]]; then
    usage >&2
    exit 2
fi
checkpoint=$(realpath -- "$checkpoint")
vllm_root=$(realpath -- "$vllm_root")
manifest=$(realpath -- "$manifest")
pin=81a8cb063e6247fe254d7caf7e98a6691d71022a
test "$(git --no-replace-objects -C "$vllm_root/source" rev-parse HEAD)" = "$pin"
python=$vllm_root/venv/bin/python
site=$("$python" -IB -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
(cd -- "$checkpoint" && sha256sum --check --strict --quiet "$manifest")

run_control() {
    local group=$1 context kv_bytes run seed_cache exit_code
    trap stop_group EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    prepare_run "$index" "$target/vllm-$group"
    case "$group" in
        W1-W2) context=2560; kv_bytes=10737418240 ;;
        *) context=132096; kv_bytes=137438953472; prepare_fixtures ;;
    esac
    printf '%s\n' "$pin" >"$output/vllm-source-commit.txt"
    cp -- "$vllm_root"/{artifact.txt,freeze.txt,native.sha256} \
        "$protocol/benchmarks/vllm/wheel.sha256" "$output/"
    (cd -- "$vllm_root/wheels" && sha256sum --check "$output/wheel.sha256") \
        >"$output/wheel-check.txt"
    (cd -- "$site" && sha256sum --check "$output/native.sha256") >"$output/native-check.txt"
    sha256sum "$paired_dir"/{run_vllm.sh,vllm_worker.py} \
        "$protocol/benchmarks/streaming/vllm_observer.py" >"$output/vllm-launcher.sha256"

    cd -- "$output"
    mkdir cache config tmp
    seed_cache=${PAIRED_VLLM_SEED_CACHE:-$host_root/cache/vllm-81a8cb063e62}
    printf '%s\n' "$seed_cache" >cache-seed.txt
    cp -aL --reflink=auto "$seed_cache/." cache/
    export CUDA_VISIBLE_DEVICES=$gpu_uuid CUDA_DEVICE_ORDER=PCI_BUS_ID
    export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MAX_JOBS=2
    export XDG_CACHE_HOME="$output/cache" XDG_CONFIG_HOME="$output/config" TMPDIR="$output/tmp"
    export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm" VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
    export HF_HOME="$XDG_CACHE_HOME/huggingface" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
    export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME"
    export VLLM_RPC_BASE_PATH=tmp VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_SERVER_DEV_MODE=1
    export CUDA_HOME=/usr/local/cuda-13.2 PATH="$vllm_root/venv/bin:/usr/local/cuda-13.2/bin:$PATH"
    export LD_LIBRARY_PATH="$compat" CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$paired_dir:$protocol"
    unset PYTHONHOME CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
    local keys='CUDA_|VLLM_|FLASHINFER_|HF_|TRITON_|OMP_NUM_THREADS=|MAX_JOBS='
    keys+='|XDG_CACHE_HOME=|XDG_CONFIG_HOME=|LD_LIBRARY_PATH=|CC=|CXX=|TMPDIR=|PYTHONPATH='
    env | LC_ALL=C sort | sed -nE "/^($keys)/p" >effective-environment.txt

    run=$output/$group
    mkdir "$run"
    start_group "$run" "$python" -B -m vllm.entrypoints.openai.api_server \
        --host 127.0.0.1 --port "$port" --model "$checkpoint" --served-model-name benchmark \
        --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --mamba-cache-dtype bfloat16 \
        --mamba-ssm-cache-dtype float32 \
        --attention-config '{"backend":"FLASHINFER","disable_flashinfer_q_quantization":true}' \
        --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_AND_PIECEWISE"}' \
        --language-model-only --skip-tokenizer-init --generation-config vllm --seed 0 \
        --no-enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill \
        --max-model-len "$context" --max-num-batched-tokens "$token_budget" --max-num-seqs 32 \
        --max-num-scheduled-tokens "$token_budget" --async-scheduling \
        --kv-cache-memory-bytes "$kv_bytes" --stream-interval 1 \
        --worker-extension-cls vllm_worker.PairedBaselineObserver
    curl --noproxy '*' -fsS --max-time 120 -H 'Content-Type: application/json' \
        -d '{"method":"baseline_snapshot","timeout":90}' "$url/collective_rpc" \
        >"$run/effective-worker.json"
    "$client_python" -B "$protocol/benchmarks/streaming/check_vllm.py" \
        audit "$run/effective-worker.json"
    "$client_python" -B - "$run" "$group" "$server_cpus" <<'PY'
import sys
from pathlib import Path

from benchmarks.concurrent.reference import reference
from saved_evidence import command, require, vllm_settings

run = Path(sys.argv[1])
placement = {"memory": "localalloc", "server": sys.argv[3]}
worker = vllm_settings(run, command(run, "server", placement), sys.argv[2])
pinned = reference()["observed_vllm_settings"]
require(all(worker[key] == pinned[key] for key in ("vllm", "torch", "torch_cuda", "compilation")),
        f"{run}: unexpected vLLM runtime or graph identity")
PY
    curl --noproxy '*' -fsS --max-time 10 "$url/metrics" >"$run/metrics-before.txt"
    exit_code=0
    measure_group "$group" "$run" || exit_code=$?
    curl --noproxy '*' -fsS --max-time 10 "$url/metrics" >"$run/metrics-after.txt" || true
    stop_group
    record_compute "$run/compute-after.csv"
    date -u +%FT%TZ >finished-utc.txt
    exit "$exit_code"
}

mkdir -- "$target"
target=$(realpath -- "$target")
cp -- "$manifest" "$target/checkpoint.sha256"
exit_code=0
for group in "${groups[@]}"; do
    result=0
    (set -e; run_control "$group") &
    control_pid=$!
    trap 'kill -TERM "$control_pid" 2>/dev/null || true; wait "$control_pid" || true; exit 130' INT
    trap 'kill -TERM "$control_pid" 2>/dev/null || true; wait "$control_pid" || true; exit 143' TERM
    wait "$control_pid" || result=$?
    if (( result != 0 )); then
        exit_code=$result
    fi
done
exit "$exit_code"
