#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash benchmarks/paired/run_native.sh" \
        "BUILD_DIR GPU_INDEX NEW_OUTPUT_DIR [GROUP ...]"
    echo "Groups: W1-W2 W3 W4 W5 W6 (default: all in this order)"
    echo "Set PAIRED_VLLM_RESULTS to check counts against saved vLLM runs."
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# < 3 )); then
    usage >&2
    exit 2
fi

binary=$(realpath -- "$1/bin/sglang-server")
[[ -x $binary ]]
index=$2
target=$3
shift 3
groups=("$@")
if (( ${#groups[@]} == 0 )); then
    groups=(W1-W2 W3 W4 W5 W6)
fi
for group in "${groups[@]}"; do
    case "$group" in
        W1-W2|W3|W4|W5|W6) ;;
        *) usage >&2; exit 2 ;;
    esac
done

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ ! $warmups =~ ^[1-9][0-9]*$ || ! $measured =~ ^([2-9]|[1-9][0-9]+)$ ]]; then
    echo "Use at least one warmup and two measured concurrent cohorts." >&2
    exit 2
fi
if [[ -n ${PAIRED_VLLM_RESULTS:-} ]]; then
    for group in "${groups[@]}"; do
        check_saved_counts "$group"
    done
fi
trap stop_group EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
prepare_run "$index" "$target"
printf 'warmups=%s\nmeasured=%s\nsaved_vllm_results=%s\nallow_count_mismatch=%s\n' \
    "$warmups" "$measured" "${PAIRED_VLLM_RESULTS:-}" "${PAIRED_ALLOW_COUNT_MISMATCH:-0}" \
    >"$output/comparison-inputs.txt"
sha256sum "$binary" "$paired_dir/run_native.sh" >"$output/native.sha256"
export CUDA_VISIBLE_DEVICES=$gpu_uuid CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir "$output/cuda-cache"
export CUDA_CACHE_PATH="$output/cuda-cache"
ldd "$binary" >"$output/native-ldd.txt"
env | LC_ALL=C sort |
    sed -nE '/^(CUDA_|LD_LIBRARY_PATH=|SGLANG_(EXPERIMENTAL_|PREFILL_|DECODE_|SERIAL_))/p' \
        >"$output/native-environment.txt"

if [[ ${groups[*]} != W1-W2 ]]; then
    prepare_fixtures
fi
exit_code=0
for group in "${groups[@]}"; do
    run=$output/$group
    mkdir "$run"
    echo "$group on $gpu_uuid"
    if start_group "$run" "$binary" --checkpoint "$checkpoint" --device 0 --port "$port" \
        --fp8-prefill tensor-core --batch-cuda-graphs --mtp-steps 0 --prefix-cache-entries 0 \
        --max-active 32 --max-num-batched-tokens 32768 &&
        jq -e '.max_active == 32 and .max_num_batched_tokens == 32768' "$run/health.json" >/dev/null
    then
        measure_group "$group" "$run" native || exit_code=1
    else
        exit_code=1
    fi
    stop_group
    record_compute "$run/compute-after.csv"
done
date -u +%FT%TZ >"$output/finished-utc.txt"
exit "$exit_code"
