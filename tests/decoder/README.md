# Decoder CUDA checks

## FP8 projection aliases

From the repository root in the managed Pixi `default` environment, build the native CUDA library and link the check against it. Reserve a B200 and set `GPU_UUID` to its device UUID before running.

```bash
out="$PWD/build/fp8-alias"
xmake f -P "$PWD" -o "$out" -p linux -a x86_64 -m release
xmake build -P "$PWD" sglang_cuda
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
"$CXX" -std=c++17 -O3 -Wall -Wextra -Werror -Iinclude -I"$toolkit/include" \
    tests/decoder/decode_fp8_alias_check.cpp -L"$out/lib" -L"$toolkit/lib" \
    -Wl,-rpath,"$toolkit/lib" -Wl,--wrap=cudaLaunchKernel -lsglang_cuda -lcudart \
    -o "$out/decode-fp8-alias-check"
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" "$out/decode-fp8-alias-check"
```

Expect `FP8 alias checks passed` and exit status zero. The check uses a small deterministic fixture and needs no checkpoint. It compares lawful aliases with ordinary GEMV and SwiGLU. Rejected aliases must return `cudaErrorInvalidValue` before submitting a CUDA kernel.
