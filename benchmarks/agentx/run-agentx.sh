#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 || ( $1 != 1 && $1 != 2 ) || $2 != http*://* || -z $3 || -z $4 ]]; then
    printf 'Usage: bash %s <1|2> <server-base-url> <served-model-name> <new-output-dir>\n' "$0" >&2
    exit 2
fi

client_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
cache=/data1/shared/dongwu.chen/cache/agentx-256k-client
tokenizer=/data0/shared/Qwen3.8-27B-FP8
trace_sha256=e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef
trace_revision=8fecd2fc56694469f758f0afbbb6335ad3043740
trace_dataset=semianalysisai/cc-traces-weka-062126-256k
concurrency=$(($1 * 8))
url=${2%/}
model=$3
output=$(realpath -m -- "$4")

work_dir="$cache/replay"
local_dataset="$work_dir/$trace_dataset"
mkdir -p "$local_dataset"
ln -sfn "$cache/traces.jsonl" "$local_dataset/traces.jsonl"
printf '%s  %s\n' "$trace_sha256" "$local_dataset/traces.jsonl" | sha256sum --check --status

export PIXI_HOME="$cache/pixi-home" PIXI_CACHE_DIR="$cache/pixi"
export RATTLER_CACHE_DIR="$cache/rattler" UV_CACHE_DIR="$cache/uv"
export XDG_CACHE_HOME="$cache/xdg" HF_HOME="$cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
tokenizer_digest=$(sha256sum "$tokenizer/tokenizer.json")
mmap_dir="$cache/mmap/$trace_sha256/${tokenizer_digest%% *}"
export AIPERF_DATASET_MMAP_CACHE_DIR="$mmap_dir"
export CUDA_VISIBLE_DEVICES='' NO_PROXY='*' no_proxy='*'

mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
command=(
    pixi run --no-config --locked --manifest-path "$client_dir/pixi.toml"
    agentx profile --scenario agentx
    --endpoint-type chat --url "$url" --model "$model"
    --tokenizer "$tokenizer" --public-dataset semianalysis_cc_traces_weka_062126_256k
    --benchmark-duration 1800 --concurrency "$concurrency"
    --use-server-token-count --export-level raw --no-export-outputs-json
    --ui none --output-artifact-dir "$output"
)
{
    printf 'trace_dataset=%s\ntrace_revision=%s\ntrace_sha256=%s\n' \
        "$trace_dataset" "$trace_revision" "$trace_sha256"
    sha256sum "$client_dir/pixi.lock" "$tokenizer/tokenizer.json" \
        "$tokenizer/tokenizer_config.json" "$tokenizer/chat_template.jinja"
    printf 'cwd=%q\ncommand=' "$work_dir"
    printf '%q ' "${command[@]}"
    printf '\n'
} > "$output/client-invocation.txt"

cd "$work_dir"
printf 'client_start_utc=%s\n' "$(date -u +%FT%TZ)" > "$output/client-status.txt"
set +e
"${command[@]}" 2>&1 | tee "$output/client-console.log"
status=("${PIPESTATUS[@]}")
set -e
printf 'client_end_utc=%s\nclient_exit_code=%s\ntee_exit_code=%s\n' \
    "$(date -u +%FT%TZ)" "${status[0]}" "${status[1]}" >> "$output/client-status.txt"
mmap_event='(?:Populated mmap cache entry [^ ]+/|'
mmap_event+='Memory-mapped dataset cache HIT(?: under lock)? \(key=)'
mapfile -t cache_keys < <(
    rg --no-filename --only-matching --replace '$1' "${mmap_event}([0-9a-f]{32})" \
        "$output/client-console.log" "$output/logs" 2>/dev/null | sort -u
)
if (( ${#cache_keys[@]} == 1 )) && [[ -f "$mmap_dir/${cache_keys[0]}/manifest.json" ]]; then
    cp -- "$mmap_dir/${cache_keys[0]}/manifest.json" "$output/dataset-manifest.json"
    printf 'dataset_mmap_key=%s\n' "${cache_keys[0]}" >> "$output/client-status.txt"
fi
if (( status[0] != 0 )); then
    exit "${status[0]}"
fi
exit "${status[1]}"
