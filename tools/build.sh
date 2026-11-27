#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: bash tools/build.sh [--build-dir DIRECTORY] [--gpu INDEX] [--nvrtc]"
    echo "                           [--mxfp8-check] [--offline]"
    echo "The default gate builds sglang-generate, its libraries, and the toolchain probes."
    echo "Use --build-dir to retain incremental build files for this checkout."
    echo "Use --gpu to run on the physical device listed by nvidia-smi."
    echo "Add --nvrtc to select the runtime compilation probe for the GPU run."
    echo "Use --mxfp8-check to build the MXFP8 GQA check from tests/mxfp8_gqa_check.cu."
    echo "Use --offline to require the installed Pixi environment and a local CUTLASS archive."
}

output_dir=''
gpu_index=''
gpu_probe=toolchain-probe
mxfp8_check=n
offline=n
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
        --nvrtc) gpu_probe=nvrtc-probe; shift ;;
        --mxfp8-check) mxfp8_check=y; shift ;;
        --offline) offline=y; shift ;;
        --help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
base=${SGLANG_HOST_ROOT:-/data1/shared/dongwu.chen}
/bin/bash "$tools/toolchain-xmake.sh"
pixi_flags=(--as-is)
network=()
if [[ "$offline" == y ]]; then
    pixi_flags=(--offline --locked --no-install)
else
    proxy=${all_proxy:-${ALL_PROXY:-http://127.0.0.1:7890}}
    network=(http_proxy="${http_proxy:-${HTTP_PROXY:-$proxy}}"
             https_proxy="${https_proxy:-${HTTPS_PROXY:-$proxy}}" all_proxy="$proxy")
fi
exec /usr/bin/env -i HOME="$HOME" PATH=/usr/bin:/bin \
    TMPDIR="${TMPDIR:-/tmp}" SGLANG_BUILD_OFFLINE="$offline" "${network[@]}" \
    SGLANG_HOST_ROOT="$base" SGLANG_BUILD_DIR="$output_dir" SGLANG_GPU_INDEX="$gpu_index" \
    SGLANG_GPU_PROBE="$gpu_probe" SGLANG_MXFP8_CHECK="$mxfp8_check" \
    SGLANG_CUTLASS_ARCHIVE="${SGLANG_CUTLASS_ARCHIVE:-}" \
    "$base/bin/pixi" run --manifest-path "$base/pixi.toml" \
    --environment default "${pixi_flags[@]}" \
    /bin/bash "$tools/build-probe.sh"
