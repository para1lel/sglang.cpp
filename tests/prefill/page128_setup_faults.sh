#!/usr/bin/env bash
set -euo pipefail

build=${1:?Pass the SM100 Release build directory}
toolkit="${CONDA_PREFIX:?Activate the default Pixi environment}/targets/x86_64-linux"
source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$source/../.." && pwd -P)

apis=(
    cudaGetDevice cudaDeviceGetAttribute cudaGetDriverEntryPointByVersion
    cudaFuncSetAttribute cudaPeekAtLastError cudaGetLastError
    cudaMallocAsync cudaMemsetAsync cudaLaunchKernel cudaFreeAsync
)
wrap=()
for api in "${apis[@]}"; do
    wrap+=("-Wl,--wrap=$api")
done

"${CXX:-c++}" -std=c++17 -O2 -I "$root/include" -isystem "$toolkit/include" \
    "$source/page128_setup_faults.cpp" "$build/lib/libsglang_prefill_fp8_sm100.a" "${wrap[@]}" \
    -L "$toolkit/lib" -Wl,-rpath,"$toolkit/lib" -lcudart -o "$build/page128-setup-faults"
CUDA_VISIBLE_DEVICES='' "$build/page128-setup-faults"
