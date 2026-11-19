#!/usr/bin/env bash
set -euo pipefail

seal=${1:?Pass the SHA-256 of SHA256SUMS}
tree=$(realpath "${2:?Pass a verified source tree}")
build=$(realpath -m "${3:?Pass a new build directory}")
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
arm=${tree##*/}

case "$build/" in
    "$tree/"*|"$bundle/"*) exit 2 ;;
esac
case "$arm" in
    w2-2048-scores|w2-pair-scores) targets=(serial-score-probe) ;;
    w3-async2-scores|w3-score-control) targets=(batch-replay) ;;
    w6-width-four) targets=(schedule-parity) ;;
    segmented-prefill|mtp-adaptive|mtp-parallel-control|mtp-serial-control|\
    model-reference|model-reference-graphs) targets=(sglang-server sglang-generate) ;;
    w2-2048|w2-pair|w3-async2|w3-fused-control|w6-donor) targets=(sglang-server) ;;
    *) printf 'Unknown replay tree %s\n' "$arm" >&2; exit 2 ;;
esac

python3 -IB "$bundle/replay.py" --seal "$seal" --check-tree "$tree" "$arm"
: "${CONDA_PREFIX:?Activate the pinned CUDA compiler environment}"
: "${CC:?Set the pinned GCC compiler}" "${CXX:?Set the pinned G++ compiler}"
mkdir "$build"
complete=0
trap 'if [[ $complete -eq 0 ]]; then rm -rf -- "$build"; fi' EXIT
cp -a -- "$tree" "$build/source"
python3 -IB "$bundle/replay.py" --seal "$seal" --check-tree "$build/source" "$arm"
mkdir "$build/cache" "$build/tmp"
export CUDA_VISIBLE_DEVICES='' XMAKE_STATS=false XMAKE_COLORTERM=nocolor CCACHE_DISABLE=1
export TMPDIR="$build/tmp" XMAKE_TMPDIR="$build/tmp" XDG_CACHE_HOME="$build/cache"
export XMAKE_GLOBALDIR="$build/cache/global" XMAKE_CONFIGDIR="$build/cache/config"
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS

cd "$build/source"
xmake f -P . -p linux -a x86_64 -m release -o "$build/out" -y
for target in "${targets[@]}"; do
    xmake build -P . -j "${JOBS:-3}" "$target"
    test -x "$build/out/bin/$target"
done
(
    cd "$build/out/bin"
    sha256sum -- "${targets[@]}"
) | tee "$build/binaries.sha256"
complete=1
