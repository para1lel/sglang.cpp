#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
base=/data1/shared/dongwu.chen
root=$base/environments/vllm-81a8cb063e62
gpu=${GPU:-1}
model=/data0/shared/Qwen3.8-27B-FP8
case_file=$(realpath "${1:-$here/small.json}")
mkdir -p "$root/runs"
run=$(mktemp -d "$root/runs/run.XXXXXX")
printf 'Run directory %s\n' "$run"
cd "$run"
date -u +%FT%TZ >started-utc.txt
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
[[ $CUDA_VISIBLE_DEVICES =~ ^GPU-[0-9a-fA-F-]{36}$ ]] || exit 1
gpu=$CUDA_VISIBLE_DEVICES
exec 9>"$root/$gpu.lock"
flock -n 9 || { echo "Another baseline owns $gpu" >&2; exit 1; }
fields=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version
fields+=,clocks.current.sm,clocks.current.memory,power.limit
nvidia-smi -i "$gpu" --query-gpu="$fields" --format=csv >gpu-before.csv
finish() {
    printf '%s\n' "$?" >exit-code.txt
    nvidia-smi -i "$gpu" --query-gpu="$fields" --format=csv >gpu-after.csv
}
trap finish EXIT
processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
if [[ -n $processes || $free -lt 65000 ]]; then
    echo "$gpu is occupied or has less than 65000 MiB free" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy ALL_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MAX_JOBS=2
export XDG_CACHE_HOME="$base/cache/vllm-81a8cb063e62"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$root/venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$base/software/cuda-compat-13.4/usr/local/cuda-13.4/compat"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ VLLM_WORKER_MULTIPROC_METHOD=spawn
unset PYTHONHOME PYTHONPATH CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
cp "$case_file" case.json
cp "$root/artifact.txt" "$root/wheel.sha256" "$root/freeze.txt" "$root/native.sha256" .
(export LC_ALL=C; cd "$model"; sha256sum *.safetensors *.json *.txt *.jinja) >checkpoint.sha256
expected=b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303
printf '%s  checkpoint.sha256\n' "$expected" | sha256sum --check
uname -a >host.txt
LC_ALL=C lscpu >>host.txt
timeout --kill-after=20s 10m "$root/venv/bin/python" -I "$here/measure.py" \
    case.json "$model" result.json 2>&1 | tee engine.log
cat result.json
