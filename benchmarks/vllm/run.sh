#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
base=/data1/shared/dongwu.chen
root=$base/environments/vllm-81a8cb063e62
gpu=${GPU:-1}
model=/data0/shared/Qwen3.8-27B-FP8
case_file=$(realpath "${1:-$here/small.json}")
runs=$(realpath -m "${VLLM_RUNS_DIR:-$root/runs}")
mkdir -p "$runs"
run=$(mktemp -d "$runs/run.XXXXXX")
printf 'Run directory %s\n' "$run"
cd "$run"
date -u +%FT%TZ >started-utc.txt
fields=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version
fields+=,clocks.current.sm,clocks.current.memory,power.limit,pci.bus_id
finish() {
    printf '%s\n' "$?" >exit-code.txt
    if [[ ${locked:-0} == 1 ]]; then
        nvidia-smi -i "$gpu" --query-gpu="$fields" --format=csv >gpu-after.csv
        nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader >compute-after.csv
    fi
}
trap finish EXIT
export CUDA_DEVICE_ORDER=PCI_BUS_ID
proxy=${all_proxy:-${ALL_PROXY:-http://127.0.0.1:7890}}
export http_proxy=${http_proxy:-${HTTP_PROXY:-$proxy}}
export https_proxy=${https_proxy:-${HTTPS_PROXY:-$proxy}}
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy ALL_PROXY=$proxy all_proxy=$proxy
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MAX_JOBS=2
mkdir cache config tmp
cache_seed=$base/cache/vllm-81a8cb063e62
if [[ -d $cache_seed ]]; then
    cp -aL --reflink=auto "$cache_seed/." cache/
fi
export XDG_CACHE_HOME="$run/cache" XDG_CONFIG_HOME="$run/config" TMPDIR="$run/tmp"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm" VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
export HF_HOME="$XDG_CACHE_HOME/huggingface" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export VLLM_RPC_BASE_PATH=tmp
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$root/venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$base/software/cuda-compat-13.4/usr/local/cuda-13.4/compat"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ VLLM_WORKER_MULTIPROC_METHOD=spawn
unset PYTHONHOME PYTHONPATH CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
cp "$case_file" case.json
cp "$here"/{run.sh,measure.py,wheel.sha256} "$root"/{artifact.txt,freeze.txt,native.sha256} .
sha256sum run.sh measure.py case.json artifact.txt freeze.txt native.sha256 wheel.sha256 \
    >inputs.sha256
python=$root/venv/bin/python
site=$("$python" -IB -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
(cd "$root/wheels" && sha256sum --check "$run/wheel.sha256")
(cd "$site" && sha256sum --check "$run/native.sha256")
(export LC_ALL=C; cd "$model"; sha256sum *.safetensors *.json *.txt *.jinja) >checkpoint.sha256
expected=b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303
printf '%s  checkpoint.sha256\n' "$expected" | sha256sum --check
uname -a >host.txt
LC_ALL=C lscpu >>host.txt
gpu=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
[[ $gpu =~ ^GPU-[0-9a-fA-F-]{36}$ ]] || exit 1
exec 9>>"$root/$gpu.lock"
flock -n 9 || { echo "Another baseline owns $gpu" >&2; exit 1; }
locked=1
nvidia-smi -i "$gpu" --query-gpu="$fields" --format=csv >gpu-before.csv
nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader >compute-before.csv
read -r free utilization < <(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ,)
if [[ -s compute-before.csv || $free -lt 65000 || $utilization -ne 0 ]]; then
    echo "$gpu is active or has less than 65000 MiB free" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$gpu
command=(timeout --kill-after=20s 10m "$python" -IB measure.py case.json "$model" result.json)
printf '%q ' "${command[@]}" >command.txt
printf '\n' >>command.txt
"${command[@]}" 2>&1 | tee engine.log
cat result.json
