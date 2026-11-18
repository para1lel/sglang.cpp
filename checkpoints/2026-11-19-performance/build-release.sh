#!/usr/bin/env bash
set -euo pipefail

seal=${1:?Pass the published SHA-256 of SHA256SUMS}
tree=$(realpath "${2:?Pass a verified source tree}")
build=$(realpath -m "${3:?Pass a new build directory}")
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
arm=${tree##*/}
jobs=${JOBS:-3}

case "$build/" in
    "$tree/"*|"$bundle/"*)
        printf 'Build directory must be external to replay and bundle\n' >&2
        exit 2
        ;;
esac
unset VISION_NATIVE_SOURCE VISION_NATIVE_MANIFEST VISION_RUN
case "$arm" in
    serial-vector|serial-quant|serial-cta|w3-flash8192|w3-fused|wide-fast|wide-gqa)
        target=sglang-server ;;
    vision-rounding)
        target=vision-language-check
        export VISION_NATIVE_SOURCE="$build/source/vision/encoder"
        export VISION_NATIVE_MANIFEST="$bundle/vision" VISION_RUN="$build/encoder"
        ;;
    *) printf 'Unknown replay tree: %s\n' "$arm" >&2; exit 2 ;;
esac

python3 -B "$bundle/verify.py" --seal "$seal" --check-tree "$tree" "$arm"
: "${CONDA_PREFIX:?Activate the CUDA compiler environment}"
: "${CC:?Set CC to the GCC compiler}" "${CXX:?Set CXX to the G++ compiler}"
mkdir "$build"
complete=0
trap 'if [[ $complete -eq 0 ]]; then rm -rf -- "$build"; fi' EXIT
cp -a -- "$tree" "$build/source"
python3 -B "$bundle/verify.py" --seal "$seal" --check-tree "$build/source" "$arm"
mkdir "$build/cache" "$build/tmp"
export CUDA_VISIBLE_DEVICES='' XMAKE_STATS=false XMAKE_COLORTERM=nocolor
export TMPDIR="$build/tmp" XMAKE_TMPDIR="$build/tmp" XDG_CACHE_HOME="$build/cache"
export XMAKE_GLOBALDIR="$build/cache/global" XMAKE_CONFIGDIR="$build/cache/config"
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS

cd "$build/source"
xmake f -P . -m release -o "$build/engine" -y
xmake build -P . -j "$jobs" "$target"
test -x "$build/engine/bin/$target"
products=("$build/engine/bin/$target")
if [[ $arm == vision-rounding ]]; then
    cd "$VISION_NATIVE_SOURCE"
    export XMAKE_CONFIGDIR="$build/cache/encoder-config"
    xmake f -P . -m release -o "$build/encoder" -y
    xmake build -P . -j "$jobs" vision-encoder
    test -x "$build/encoder/build/bin/vision-encoder"
    products+=("$build/encoder/build/bin/vision-encoder")
fi
sha256sum "${products[@]}" | tee "$build/SHA256SUMS"
complete=1
