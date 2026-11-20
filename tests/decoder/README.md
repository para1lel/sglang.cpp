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

## Decode bandwidth

Use the same Pixi environment, Release configuration and reserved B200. Build `sglang-generate`
to obtain the production decoder and CUDA libraries. Set `checkpoint` to the local
Qwen3.8-27B-FP8 checkpoint.

```bash
xmake build -P "$PWD" sglang-generate
checkpoint=/data0/shared/Qwen3.8-27B-FP8
"$CXX" -std=c++17 -O3 -Wall -Wextra -Werror -Iinclude -I"$toolkit/include" \
    tests/decoder/decode_bandwidth_check.cpp -L"$out/lib" -L"$toolkit/lib" \
    -Wl,-rpath,"$toolkit/lib" -Wl,--wrap=cudaLaunchKernel -lsglang_cuda -lcudart \
    -o "$out/decode-bandwidth-check"
"$CXX" -std=c++17 -O3 -Wall -Wextra -Werror -Iinclude -I"$toolkit/include" \
    tests/decoder/decode_graph_check.cpp -L"$out/lib" -L"$toolkit/lib" \
    -Wl,-rpath,"$toolkit/lib" \
    -Wl,--wrap=cudaGraphLaunch -Wl,--wrap=cudaStreamBeginCapture \
    -lsglang_decoder -lsglang_checkpoint -lsglang_json -lsglang_cuda -lsglang_prefill_fp8_sm100 \
    -lcudart -o "$out/decode-graph-check"
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" "$out/decode-bandwidth-check"
for policy in exact tensor-core; do
    CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/decode-graph-check" "$checkpoint" "$policy"
done
```

The operator check compares output bits and guards with ordinary BF16 GEMV and FP8 quantization.
It checks successful kernel launch counts and grid and block dimensions for optimized and fallback
cases. Invalid arguments must submit no kernels.

The checkpoint check expects one capture on the first request, three graph launches on each of
two default requests, and no capture or launch on the eager request. The second default request
must reuse the first graph. All three requests must return identical greedy IDs and full score
bits within each prefill policy.

## GDN prefill

In the managed Pixi environment, enable the opt-in GDN checks and reserve a B200.
The operator check uses synthetic inputs and the route check loads the local checkpoint.

```bash
out="$PWD/build/gdn-checks"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
xmake f -P "$PWD" -o "$out" -p linux -a x86_64 -m release --gdn-check=y
xmake build -P "$PWD" gdn-check gdn-routes
jq -r '.prompt[]' tests/prefix/route-cases.json > "$out/prompt.ids"
export CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}"
"$out/bin/gdn-check"
"$out/bin/gdn-check" short paired "$out/ordered.bin"
"$out/bin/gdn-check" chunked
for policy in exact tensor-core; do
    "$out/bin/gdn-routes" "$checkpoint" "$out/prompt.ids" "$policy"
done
```

The chunked check observes the matrix and recurrence launches, compares outputs and state with
an FP64 sequence, and verifies causal outputs and workspace guards. The route check observes
the first absolute tile, restored prefix, mixed prefill and decode, and reuse of a retired lane.
Its exact mode emits four generated IDs and 16 scores per step for comparison across builds.

## Checkpoint metadata

Build the checkpoint library in the managed Pixi environment. Both checks run
on the host. The metadata probe compares file offsets derived from Linux
`/proc/self/maps` with an independent read of the index and shard headers.
The fixture suite covers sparse offsets above 4 GiB, historical JSON number
spellings, and invalid or truncated metadata.

```bash
out="$PWD/build/checkpoint-metadata"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
xmake f -P "$PWD" -o "$out" -p linux -a x86_64 -m release
xmake build -P "$PWD" sglang_checkpoint
for check in checkpoint_check checkpoint_metadata_probe; do
    "$CXX" -std=c++17 -O3 -Wall -Wextra -Werror -Iinclude \
        "tests/decoder/$check.cpp" -L"$out/lib" -lsglang_checkpoint -lsglang_json -o "$out/$check"
done
"$out/checkpoint_check" "$checkpoint/config.json" "$out/checkpoint-fixtures"
python3 -B tests/decoder/test_checkpoint_metadata.py \
    "$out/checkpoint_metadata_probe" "$checkpoint" --fixtures "$out"
```
