#!/usr/bin/env bash
set -euo pipefail

: "${VLLM_ENV:?set the pinned vLLM environment directory}"
: "${CHECKPOINT:?set the checkpoint directory}"
: "${GPU_UUID:?set the reserved B200 UUID}"
: "${GPU_LOCK_FILE:?set the existing advisory lock for this UUID}"
: "${RUN_DIR:?set a new run directory with a short absolute path}"

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
media="$repo/benchmarks/media"
environment=$(cd "$VLLM_ENV" && pwd)
model=$(cd "$CHECKPOINT" && pwd)
python="$environment/venv/bin/python"
pin=81a8cb063e6247fe254d7caf7e98a6691d71022a
manifest=b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303
test "$(git --no-replace-objects -C "$environment/source" rev-parse HEAD)" = "$pin"

exec 9<"$GPU_LOCK_FILE"
flock -n -x 9
test "$(nvidia-smi -i "$GPU_UUID" --query-gpu=name --format=csv,noheader)" = "NVIDIA B200"
compute=$(nvidia-smi -i "$GPU_UUID" --query-compute-apps=pid --format=csv,noheader)
test -z "$compute"
IFS=, read -r free utilization < <(
    nvidia-smi -i "$GPU_UUID" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits
)
((free >= 65000 && utilization == 0))

(cd "$environment/wheels" && sha256sum -c "$repo/benchmarks/vllm/wheel.sha256")
site=$("$python" -IB -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
(cd "$site" && sha256sum -c "$media/native.sha256")
mkdir "$RUN_DIR"
cd "$RUN_DIR"
run=$PWD
(export LC_ALL=C; cd "$model"; sha256sum *.safetensors *.json *.txt *.jinja) >checkpoint.sha256
printf '%s  checkpoint.sha256\n' "$manifest" | sha256sum -c
mkdir cache config tmp
if [[ -n ${CACHE_SEED:-} ]]; then
    cp -aL --reflink=auto "$CACHE_SEED/." cache/
fi

export CUDA_VISIBLE_DEVICES="$GPU_UUID" CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2 MAX_JOBS=2 VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$run/cache" XDG_CONFIG_HOME="$run/config" TMPDIR="$run/tmp"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm" VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
export HF_HOME="$XDG_CACHE_HOME/huggingface" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME"
export VLLM_RPC_BASE_PATH=tmp VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_SERVER_DEV_MODE=1
export VLLM_VIDEO_LOADER_BACKEND=opencv PYTHONPATH="$repo" PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="$environment/venv/bin:$CUDA_HOME/bin:$PATH"
export CC="${CC:-/usr/bin/gcc}" CXX="${CXX:-/usr/bin/g++}"
unset PYTHONHOME CFLAGS CXXFLAGS CPPFLAGS LDFLAGS

limits='{"image":{"count":1,"width":256,"height":256},'
limits+='"video":{"count":1,"num_frames":4,"width":128,"height":128}}'
exec "$python" -B -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port "${PORT:-17462}" --model "$model" --served-model-name benchmark \
    --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --mamba-cache-dtype bfloat16 \
    --mamba-ssm-cache-dtype float32 \
    --attention-config '{"backend":"FLASHINFER","disable_flashinfer_q_quantization":true}' \
    --generation-config vllm --seed 0 --no-enable-prefix-caching --enable-prompt-tokens-details \
    --trust-request-mm-kwargs --limit-mm-per-prompt "$limits" \
    --mm-processor-cache-gb 0 --max-model-len 2560 --max-num-batched-tokens 2560 --max-num-seqs 1 \
    --kv-cache-memory-bytes 1073741824 --stream-interval 1 \
    --worker-extension-cls benchmarks.media.media_observer.MediaObserver
