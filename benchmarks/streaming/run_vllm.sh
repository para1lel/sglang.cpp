#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
base=/data1/shared/dongwu.chen
root=$base/environments/vllm-81a8cb063e62
model=/data0/shared/Qwen3.8-27B-FP8
pin=81a8cb063e6247fe254d7caf7e98a6691d71022a
gpu=$(nvidia-smi -i "${GPU:-3}" --query-gpu=uuid --format=csv,noheader)
port=${PORT:-17383}
[[ $gpu =~ ^GPU-[0-9a-fA-F-]{36}$ ]]
run_root=$repo/artifacts/workload-baselines
mkdir -p "$run_root"
run=$(mktemp -d "$run_root/run.XXXXXX")
printf 'Run directory %s\n' "$run"
cd "$run"

fields=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu
fields+=,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,power.limit
finish() {
    local code=$?
    local attempt
    trap - EXIT
    if [[ -n ${server:-} ]]; then
        kill -- "-$server" 2>/dev/null || true
        for ((attempt=0; attempt<20; ++attempt)); do
            kill -0 -- "-$server" 2>/dev/null || break
            sleep .25
        done
        kill -KILL -- "-$server" 2>/dev/null || true
        wait "$server" 2>/dev/null || true
    fi
    if [[ ${locked:-0} == 1 ]]; then
        nvidia-smi -i "$gpu" --query-gpu="$fields" --format=csv >gpu-after.csv || true
        nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader >compute-after.csv || true
    fi
    date -u +%FT%TZ >finished-utc.txt
    printf '%s\n' "$code" >exit-code.txt
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ >started-utc.txt
exec 9>>"$root/$gpu.lock"
flock -n 9 || { echo "GPU lock busy for $gpu" >&2; exit 75; }
locked=1
nvidia-smi -i "$gpu" --query-gpu="$fields" --format=csv >gpu-before.csv
nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader >compute-before.csv
read -r free utilization < <(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ,)
[[ ! -s compute-before.csv && $free -ge 65000 && $utilization -eq 0 ]] || exit 75

git --no-replace-objects -C "$repo" rev-parse HEAD >project-head.txt
test "$(git --no-replace-objects -C "$root/source" rev-parse HEAD)" = "$pin"
cp "$root"/{artifact.txt,freeze.txt,native.sha256} "$repo/benchmarks/vllm/wheel.sha256" .
cp "$here"/{run_vllm.sh,vllm_observer.py} .
(cd "$root/wheels" && sha256sum --check "$run/wheel.sha256") >wheel-check.txt
python=$root/venv/bin/python
site=$("$python" -IB -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
(cd "$site" && sha256sum --check "$run/native.sha256") >native-check.txt
(export LC_ALL=C; cd "$model"; sha256sum *.safetensors *.json *.txt *.jinja) >checkpoint.sha256 &
checkpoint_job=$!
uname -a >host.txt
LC_ALL=C lscpu >>host.txt
mkdir cache config tmp profile
seed_cache=${VLLM_STREAMING_SEED_CACHE:-$base/cache/vllm-81a8cb063e62}
printf '%s\n' "$seed_cache" >cache-seed.txt
if [[ -d $seed_cache ]]; then
    cp -aL --reflink=auto "$seed_cache/." cache/
fi

export CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MAX_JOBS=2
export XDG_CACHE_HOME="$run/cache" XDG_CONFIG_HOME="$run/config" TMPDIR="$run/tmp"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm" VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
export HF_HOME="$XDG_CACHE_HOME/huggingface" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME"
export VLLM_RPC_BASE_PATH=tmp VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_SERVER_DEV_MODE=1
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$root/venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$base/software/cuda-compat-13.4/usr/local/cuda-13.4/compat"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$run
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy ALL_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
unset PYTHONHOME CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
environment_keys='CUDA_|VLLM_|FLASHINFER_|HF_|TRITON_|OMP_NUM_THREADS=|MAX_JOBS='
environment_keys+='|XDG_CACHE_HOME=|LD_LIBRARY_PATH=|CC=|CXX=|TMPDIR=|PYTHONPATH='
env | LC_ALL=C sort | sed -nE "/^($environment_keys)/p" >effective-environment.txt
profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$run/profile\","
profiler_config+='"torch_profiler_with_stack":false,"ignore_frontend":true}'
command=("$python" -B -m vllm.entrypoints.openai.api_server
    --host 127.0.0.1 --port "$port" --model "$model" --served-model-name benchmark
    --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --mamba-cache-dtype bfloat16
    --mamba-ssm-cache-dtype float32
    --attention-config '{"backend":"FLASHINFER","disable_flashinfer_q_quantization":true}'
    --language-model-only --skip-tokenizer-init --generation-config vllm --seed 0
    --no-enable-prefix-caching --enable-prompt-tokens-details
    --max-model-len 2560 --max-num-batched-tokens 2560 --max-num-seqs 1
    --kv-cache-memory-bytes 1073741824 --stream-interval 1
    --worker-extension-cls vllm_observer.StreamingBaselineObserver
    --profiler-config "$profiler_config")
printf '%q ' "${command[@]}" >server-command.txt
printf '\n' >>server-command.txt
setsid "${command[@]}" >server.log 2>&1 9>&- &
server=$!
printf '%s\n' "$server" >server-pid.txt
url=http://127.0.0.1:$port
for ((attempt=0; attempt<600; ++attempt)); do
    kill -0 "$server" 2>/dev/null || { tail -50 server.log >&2; exit 1; }
    curl -fsS --max-time 1 "$url/health" >/dev/null 2>&1 && break
    sleep 1
done
curl -fsS --max-time 2 "$url/health" >/dev/null
wait "$checkpoint_job"
printf '%s  checkpoint.sha256\n' \
    b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303 | sha256sum --check
printf '%s\n' "$url" >ready.txt
bash "$here/collect_vllm.sh" "$url" "$server"
