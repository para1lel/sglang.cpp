# Decoder CUDA checks

## Batched target GDN

Use the Release decoder libraries, the Qwen3.8-27B-FP8 checkpoint, and a reserved B200.
Pass four distinct files of whitespace-separated checkpoint token IDs, each with at least 145
tokens. The check compares score and normalized rows and committed KV and GDN state with
scalar decode for cuts from zero through five and a replacement token. It also checks
two-row graph replay and the exact FP8 default.

```bash
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
"$CXX" -std=c++20 -O3 -pthread -Iinclude -isystem "$toolkit/include" \
    tests/decoder/gdn_verify_checkpoint.cpp \
    -Wl,--start-group "$out"/lib/libsglang_*.a -Wl,--end-group \
    -L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib" -lcudart -lcuda -lcublasLt -o "$out/gdn-verify-check"
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    "$out/gdn-verify-check" "$checkpoint" lane0.txt lane1.txt lane2.txt lane3.txt
```

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
cases. For the four-lane head it exercises the first row pair, full and partial blocks, odd rows,
unaligned inputs or weights, and a different hidden width. Invalid arguments must submit no kernels.

The checkpoint check expects one capture on the first request, three graph launches on each of
two default requests, and no capture or launch on the eager request. The second default request
must reuse the first graph. All three requests must return identical greedy IDs and full score
bits within each prefill policy.

## Grouped decode LM head

Use the same Release build, Pixi environment, checkpoint and reserved B200.
The route check observes the width-four kernel, argmax, seeded sampler and host winner copies.
It covers sparse slots with different positions, eager decode, graph capture and replay, capture
rejection, width changes, a mixed prefill/decode step, and slot reuse.

```bash
sampler=_ZN6sglang7decoder13sample_logitsEPK13__nv_bfloat16iPiNS0_18SamplingParametersEiPvP11CUstream_st
"$CXX" -std=c++20 -O3 -Wall -Wextra -Werror -Iinclude -I"$toolkit/include" \
    tests/decoder/grouped_head_routes.cpp -L"$out/lib" -L"$toolkit/lib" \
    -Wl,-rpath,"$toolkit/lib" -Wl,--wrap=cudaLaunchKernel \
    -Wl,--wrap=cudaStreamBeginCapture -Wl,--wrap=cudaStreamEndCapture \
    -Wl,--wrap=cudaGraphInstantiate -Wl,--wrap=cudaGraphLaunch \
    -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap="$sampler" \
    -lsglang_decoder -lsglang_checkpoint -lsglang_json -lsglang_cuda -lsglang_prefill_fp8_sm100 \
    -lcudart -o "$out/grouped-head-routes"
jq -r '.prompt[]' tests/prefix/route-cases.json >"$out/prompt.ids"
export CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}"
for mode in eager graph rejected; do
    "$out/grouped-head-routes" "$checkpoint" "$out/prompt.ids" candidate "$mode" \
        "$out/$mode.bf16" >"$out/$mode.log"
done
```

For a matched comparison, link the same route check against the original decoder libraries and
use `base` in place of `candidate`. Compare the `token` lines and each mode's binary files.
Each file contains one full BF16 logit row per returned token in the order printed in the log.

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

## Wide long-context decode

Use the managed Pixi environment and a reserved B200 with `GPU_UUID` set.
The operator check uses synthetic MXFP8 attention and needs no checkpoint.
The route check uses the local checkpoint and cycles the supplied prompt IDs.

```bash
out="$PWD/build/long-decode"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
xmake f -P "$PWD" -o "$out" -p linux -a x86_64 -m release --long-decode-check=y
xmake build -P "$PWD" long-decode-check long-decode-routes
python3 -c 'import json, sys; print(*json.load(open(sys.argv[1]))["prompt"])' \
    tests/prefix/route-cases.json >"$out/prompt.ids"
for mode in full recovery small unaligned graph; do
    CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
        "$out/bin/long-decode-check" "$mode"
done
for policy in exact tensor-core; do
    CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/bin/long-decode-routes" \
        "$checkpoint" "$out/prompt.ids" candidate "$policy"
done
```

The operator check verifies split and combine launch geometry, shuffled pages,
contexts through 262,144 tokens, FP64 recovery, guard bytes, rejected arguments,
and injected launch failures. The graph mode rebinds page tables, positions, and
output buffers across replays, including padded lanes.

The checkpoint check uses sparse slots, mixed long and short lengths, fixed-seed
sampling, slot reuse, and short and long graph replay. Under `tensor-core`, five
long lanes use the padded width-eight bucket. The `exact` policy uses short prompts.
Use `paired` or `legacy-tensor` with the corresponding historical libraries to
check their launch sequences.

An optional `LOGITS` path records raw BF16 logits for two groups of three
width-five steps. An optional `TEACHER` file after `LOGITS` supplies 15
whitespace-delimited IDs for the final three steps, ordered by step and ascending
slot. To align the histories from the start of decode, put one additional ID at the
beginning of the file. The check forces that ID as the first output in all five lanes.
