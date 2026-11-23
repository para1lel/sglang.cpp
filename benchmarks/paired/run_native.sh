#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash benchmarks/paired/run_native.sh" \
        "BUILD_DIR GPU_INDEX NEW_OUTPUT_DIR [GROUP ...]"
    echo "Groups: W1-W2 W3 W4 W5 W6 (default: all in this order)"
    echo "Native K0 uses --mtp-steps 0 with the Release's automatic operator policy."
    echo "PAIRED_MAX_NUM_BATCHED_TOKENS defaults to 8192."
    echo "Set PAIRED_VLLM_RESULTS to validate a saved same-device control."
    echo "Set PAIRED_COMPARISON_MODE=frozen for the published GPU2 W3-W6 reference."
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
validate_run_settings
mode=${PAIRED_COMPARISON_MODE:-}
if [[ -n ${PAIRED_VLLM_RESULTS:-} && -z $mode ]]; then
    mode=matched
fi
case "$mode:${PAIRED_VLLM_RESULTS:+saved}" in
    :|matched:saved|frozen:) ;;
    *) echo "Use matched mode with saved vLLM results or frozen mode without them." >&2; exit 2 ;;
esac
if [[ $mode == frozen && " ${groups[*]} " == *" W1-W2 "* ]]; then
    echo "Select W3, W4, W5 or W6 for the published frozen reference." >&2
    exit 2
fi
trap stop_group EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
prepare_run "$index" "$target"
printf '%s\n' "warmups=$warmups" "measured=$measured" \
    "saved_vllm_results=${PAIRED_VLLM_RESULTS:-}" \
    "allow_count_mismatch=${PAIRED_ALLOW_COUNT_MISMATCH:-0}" \
    "mode=${mode:-unpaired}" "max_num_batched_tokens=$token_budget" \
    >"$output/comparison-inputs.txt"
sha256sum "$binary" "$paired_dir"/{run_native.sh,identity.py} >"$output/native.sha256"
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
if [[ -n $mode ]]; then
    plan_comparison "$mode" "${groups[@]}"
fi
exit_code=0
for group in "${groups[@]}"; do
    run=$output/$group
    mkdir "$run"
    echo "$group on $gpu_uuid"
    comparison_status=0
    if start_group "$run" "$binary" --checkpoint "$checkpoint" --device 0 --port "$port" \
        --batch-cuda-graphs --mtp-steps 0 --prefix-cache-entries 0 \
        --max-active 32 --max-num-batched-tokens "$token_budget" &&
        jq -e --argjson budget "$token_budget" \
        '.max_active == 32 and .max_num_batched_tokens == $budget' "$run/health.json" >/dev/null &&
        curl --noproxy '*' -fsS --max-time 5 "$url/server_info" >"$run/server-info.json" &&
        comparison_identity native-k0 "$group" "$checkpoint" "$token_budget" "$port" \
        >"$run/native-k0-policy.json"
    then
        if ! measure_group "$group" "$run" native; then
            exit_code=1
        elif [[ -n $mode ]]; then
            comparison_identity finish "$mode" "$group" "$comparison_plan_sha256" \
                >"$run/comparison.tmp" || comparison_status=$?
            if (( comparison_status == 0 || comparison_status == 3 )); then
                mv -- "$run/comparison.tmp" "$run/comparison.json"
            else
                rm -- "$run/comparison.tmp"
            fi
            if (( comparison_status != 0 )); then
                exit_code=1
            fi
        fi
    else
        exit_code=1
    fi
    stop_group
    record_compute "$run/compute-after.csv"
    if (( exit_code != 0 )); then
        break
    fi
done
date -u +%FT%TZ >"$output/finished-utc.txt"
exit "$exit_code"
