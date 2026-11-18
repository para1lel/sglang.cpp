#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: bash tools/build.sh [--build-dir DIRECTORY] [--gpu INDEX]"
    echo "The default gate builds a fresh probe and removes its temporary files on exit."
    echo "Use --build-dir to retain incremental build files for this checkout."
    echo "Use --gpu to run once on the physical device listed by nvidia-smi."
}

output_dir=''
gpu_index=''
while (( $# )); do
    case "$1" in
        --build-dir)
            [[ -n ${2:-} ]] || { usage >&2; exit 2; }
            output_dir=$(realpath -m -- "$2")
            shift 2 ;;
        --gpu)
            [[ ${2:-} =~ ^[0-9]+$ ]] || { usage >&2; exit 2; }
            gpu_index=$2
            shift 2 ;;
        --help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
base=${SGLANG_HOST_ROOT:-/data1/shared/dongwu.chen}
/bin/bash "$tools/toolchain-xmake.sh"
proxy=http://127.0.0.1:7890
exec /usr/bin/env -i HOME="$HOME" PATH=/usr/bin:/bin \
    SGLANG_HOST_ROOT="$base" SGLANG_BUILD_DIR="$output_dir" SGLANG_GPU_INDEX="$gpu_index" \
    http_proxy="$proxy" https_proxy="$proxy" all_proxy="$proxy" \
    "$base/bin/pixi" run --manifest-path "$base/pixi.toml" --environment default --as-is \
    /bin/bash "$tools/build-probe.sh"
