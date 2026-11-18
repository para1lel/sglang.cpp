#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
repo=$PWD
run=$(realpath -e "${1:?run directory}")
[[ $run = "$repo/artifacts/variant-baselines/"* && ! -e $run/reservation-utc.txt ]]
base=/data1/shared/dongwu.chen
runtime=$base/environments/vllm-81a8cb063e62
prior=$repo/artifacts/concurrent-baselines/gpu2.20261118T1319Z
model=/data0/shared/Qwen3.8-27B-FP8
gpu=GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c
pin=81a8cb063e6247fe254d7caf7e98a6691d71022a
port=17383
url=http://127.0.0.1:$port
gpu_fields=timestamp,index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,memory.free
gpu_fields+=,utilization.gpu,temperature.gpu,clocks.current.sm,clocks.current.memory
gpu_fields+=,power.draw,power.limit,ecc.mode.current,mig.mode.current
cd "$run"
compute() {
    nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
}
stop_server() {
    if [[ -n ${server:-} ]]; then
        kill -- "-$server" 2>/dev/null || true
        for ((attempt=0; attempt<40; ++attempt)); do
            kill -0 -- "-$server" 2>/dev/null || break
            sleep .25
        done
        kill -KILL -- "-$server" 2>/dev/null || true
        wait "$server" 2>/dev/null || true
        unset server
    fi
}
finish() {
    local code=$?
    trap - EXIT
    stop_server
    if [[ -n ${telemetry:-} ]]; then
        kill "$telemetry" 2>/dev/null || true
        wait "$telemetry" 2>/dev/null || true
    fi
    if [[ ${locked:-0} = 1 ]]; then
        compute >compute-after.csv || true
        nvidia-smi -i "$gpu" --query-gpu="$gpu_fields" --format=csv >gpu-after.csv || true
        flock -u 9
    fi
    date -u +%FT%TZ >released-utc.txt
    printf '%s\n' "$code" >reservation-exit.txt
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ >reservation-utc.txt
printf '%s\n' "$$" >reservation-pid.txt
[[ $(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader) = "$gpu" ]] || exit 75
exec 9>>"$runtime/$gpu.lock"
flock -n 9 || exit 75
locked=1
compute >compute-before.csv
read -r free utilization < <(
    nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ,
)
[[ ! -s compute-before.csv && $free -ge 175000 && $utilization -eq 0 ]] || exit 75
[[ -z $(ss -Hltn "sport = :$port") ]] || exit 75
nvidia-smi -i "$gpu" --query-gpu="$gpu_fields" --format=csv >gpu-before.csv
nvidia-smi -i "$gpu" --query-gpu="$gpu_fields" --format=csv -lms 250 \
    >gpu-samples.csv 2>gpu-samples.stderr 9>&- &
telemetry=$!

test "$(git --no-optional-locks --no-replace-objects -C "$runtime/source" rev-parse HEAD)" = "$pin"
git --no-optional-locks -C "$runtime/source" diff --quiet HEAD --
printf '%s\n' "$pin" >vllm-pin.txt
printf '%s\n' 98b5b35916b7fcadd36859c30abf069e03def377 >project-input.txt
uname -a >host.txt
LC_ALL=C lscpu >>host.txt
mkdir source cache config tmp
cp "$repo/benchmarks/variants"/{reserve.sh,observer.py} source/
cp "$prior"/{concurrent_observer.py,vllm_observer.py} source/
cp "$runtime"/{artifact.txt,freeze.txt,native.sha256} .
cp "$prior/wheel.sha256" .
(cd "$runtime/wheels" && sha256sum --check "$run/wheel.sha256") >wheel-check.txt
python=$runtime/venv/bin/python
site=$("$python" -IB -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
(cd "$site" && sha256sum --check "$run/native.sha256") >native-check.txt
(
    export LC_ALL=C
    cd "$model"
    sha256sum *.safetensors *.json *.txt *.jinja
) >checkpoint.sha256 9>&- &
checkpoint_job=$!
cp -aL --reflink=auto "$prior/cache/." cache/

export CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MAX_JOBS=2
export XDG_CACHE_HOME="$run/cache" XDG_CONFIG_HOME="$run/config" TMPDIR="$run/tmp"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm" VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
export HF_HOME="$XDG_CACHE_HOME/huggingface" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME"
export VLLM_RPC_BASE_PATH=tmp VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_SERVER_DEV_MODE=1
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$runtime/venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$base/software/cuda-compat-13.4/usr/local/cuda-13.4/compat"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$run/source"
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
unset PYTHONHOME CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
environment_keys='CUDA_|VLLM_|FLASHINFER_|HF_|TRITON_|OMP_NUM_THREADS=|MAX_JOBS='
environment_keys+='|XDG_CACHE_HOME=|LD_LIBRARY_PATH=|CC=|CXX=|TMPDIR=|PYTHONPATH='
env | LC_ALL=C sort | sed -nE "/^($environment_keys)/p" >effective-environment.txt

for row in W3-prefix W3-MTP; do
    mkdir -p "$row/profile"
    compute >"$row/compute-before.csv"
    [[ ! -s $row/compute-before.csv && -z $(ss -Hltn "sport = :$port") ]] || exit 75
    if [[ $row = W3-prefix ]]; then
        mode=(--enable-prefix-caching --mamba-cache-mode align)
    else
        mode=(--no-enable-prefix-caching
              --speculative-config '{"method":"mtp","num_speculative_tokens":1}')
    fi
    profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$run/$row/profile\","
    profiler_config+='"torch_profiler_with_stack":false,"ignore_frontend":true}'
    command=("$python" -B -m vllm.entrypoints.openai.api_server
        --host 127.0.0.1 --port "$port" --model "$model" --served-model-name benchmark
        --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --mamba-cache-dtype bfloat16
        --mamba-ssm-cache-dtype float32
        --attention-config '{"backend":"FLASHINFER","disable_flashinfer_q_quantization":true}'
        --language-model-only --skip-tokenizer-init --generation-config vllm --seed 0
        --enable-prompt-tokens-details --enable-chunked-prefill --max-model-len 132096
        --max-num-batched-tokens 8192 --max-num-seqs 32
        --kv-cache-memory-bytes 137438953472 --stream-interval 1
        --worker-extension-cls observer.VariantObserver --profiler-config "$profiler_config"
        "${mode[@]}")
    printf '%q ' "${command[@]}" >"$row/server-command.txt"
    printf '\n' >>"$row/server-command.txt"
    date -u +%FT%TZ >"$row/start-utc.txt"
    setsid "${command[@]}" >"$row/server.log" 2>&1 9>&- &
    server=$!
    printf '%s\n' "$server" >"$row/server-pid.txt"
    for ((attempt=0; attempt<600; ++attempt)); do
        kill -0 "$server" 2>/dev/null || break
        curl -fsS --max-time 1 "$url/health" >/dev/null 2>&1 && break
        sleep 1
    done
    if curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1; then
        if [[ $row = W3-prefix ]]; then
            wait "$checkpoint_job"
            printf '%s  checkpoint.sha256\n' \
                b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303 |
                sha256sum --check
        fi
        printf '%s\n' "$url" >"$row/ready.txt"
        deadline=$((SECONDS + 3600))
        while [[ ! -e $row/release.txt && $SECONDS -lt $deadline ]]; do
            kill -0 "$server" 2>/dev/null || break
            sleep 1
        done
        if [[ ! -e $row/release.txt ]]; then
            printf '%s\n' "server exit or row wait deadline" >"$row/interrupted.txt"
        fi
    else
        printf '%s\n' "server exit or startup timeout" >"$row/startup-failed.txt"
    fi
    stop_server
    compute >"$row/compute-after.csv"
    date -u +%FT%TZ >"$row/stop-utc.txt"
done
