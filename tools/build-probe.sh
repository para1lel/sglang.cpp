#!/bin/bash
set -euo pipefail

base=${SGLANG_HOST_ROOT:?Use tools/build.sh to activate Pixi}
prefix=${CONDA_PREFIX:?Missing Pixi prefix}
host=${CXX:?Missing Pixi C++ compiler}
xmake="$base/bin/xmake"
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

if [[ ${PIXI_PROJECT_MANIFEST:-} != "$base/pixi.toml" || ${PIXI_ENVIRONMENT_NAME:-} != default ]];
then
    echo "Use tools/build.sh to activate the managed Pixi workspace." >&2
    exit 1
fi

for program in "$host" "$prefix/bin/nvcc" "$("$host" -print-prog-name=as)" \
    "$("$host" -print-prog-name=ld)" "$("$host" -print-prog-name=cc1plus)"; do
    if [[ ! -x "$program" || "$(realpath -- "$program")" != "$prefix/"* ]]; then
        echo "Expected a Pixi tool under $prefix, found $program" >&2
        exit 1
    fi
done

if [[ -n "$SGLANG_BUILD_DIR" ]]; then
    inputs=$(printf '%s\n' "$root" "$prefix" "$host" && sha256sum "$base/pixi.lock" "$xmake")
    identity=$(printf '%s' "$inputs" | sha256sum)
    work="$SGLANG_BUILD_DIR/${identity%% *}"
    mkdir -p -- "$work"
else
    work=$(mktemp -d "$base/cache/sglang-build.XXXXXXXX")
    trap 'rm -rf -- "$work"' EXIT
fi
echo "Build directory: $work"
cd -- "$root"
mkdir -p -- "$work/tmp"
export XMAKE_CONFIGDIR="$work/config" XMAKE_GLOBALDIR="$work/global"
export TMPDIR="$work/tmp" XMAKE_TMPDIR="$work/tmp" XMAKE_COLORTERM=nocolor
export CUDA_VISIBLE_DEVICES='' XMAKE_STATS=false
unset NVCC_PREPEND_FLAGS
"$base/bin/pixi" --version
"$xmake" --version | sed -n '1p'
"$host" --version | sed -n '1p'
"$prefix/bin/nvcc" --version
sha256sum "$base/pixi.toml" "$base/pixi.lock"
"$xmake" f -o "$work/build" -p linux -a x86_64 -m release
"$xmake" build -v toolchain-probe

probe="$work/build/bin/toolchain-probe"
images=$("$prefix/bin/cuobjdump" --list-elf "$probe")
printf '%s\n' "$images"
grep -Eq 'sm_100\.cubin$' <<< "$images"
runtime=$(ldd "$probe" | awk '$1 ~ /^libcudart\.so/ {print $3}')
if [[ "$(realpath -e -- "$runtime")" != "$prefix/"* ]]; then
    echo "Expected the probe to load the Pixi CUDA runtime, found $runtime" >&2
    exit 1
fi
echo "CUDA runtime: $runtime"
sha256sum "$probe"

if [[ -n "$SGLANG_GPU_INDEX" ]]; then
    gpu=$(nvidia-smi -i "$SGLANG_GPU_INDEX" --query-gpu=name,uuid --format=csv,noheader)
    echo "Running on physical GPU $SGLANG_GPU_INDEX: $gpu"
    CUDA_VISIBLE_DEVICES="${gpu#*, }" "$probe"
fi
