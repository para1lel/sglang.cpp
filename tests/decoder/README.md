# Decoder CUDA checks

Run from the repository root in the managed Pixi `default` environment. Set `GPU_UUID` to an exclusively reserved B200 for CUDA checks. Start with this Bash setup. Run the additional toolkit setup for the standalone CUDA links below.

```bash
release=(-p linux -a x86_64 -m release)
checkpoint=${checkpoint:-/data0/shared/Qwen3.8-27B-FP8}
```

```bash
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
nccl=$(python -I -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl")')
cuda_lib=(-L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib")
nccl_lib=(-Wl,-rpath,"$nccl/lib" "$nccl/lib/libnccl.so.2")
operator_lib=(-Wl,--wrap=cudaLaunchKernel -lsglang_cuda -lsglang_cuda_support -lcudart)
```

## Batched target GDN

Set `out` to the Release build directory containing the decoder archives. Supply four distinct token ID files of at least 145 whitespace-separated IDs each. Match scores, normalized rows, and committed KV and GDN state against scalar decode at cuts 0–5 and on replacement. The single-lane five-row probe uses eager warmup, graph capture, and graph reuse with cuts of 5, 2, and 0.

```bash
"$CXX" -std=c++20 -O3 -pthread -Iinclude -isystem "$toolkit/include" \
    tests/decoder/gdn_verify_checkpoint.cpp \
    -Wl,--start-group "$out"/lib/libsglang_*.a -Wl,--end-group \
    "${cuda_lib[@]}" -L"$toolkit/lib/stubs" "${nccl_lib[@]}" \
    -lcudart -lcuda -o "$out/gdn-verify-check"
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    "$out/gdn-verify-check" "$checkpoint" lane0.txt lane1.txt lane2.txt lane3.txt
```

## FP8 projection aliases

Build the native CUDA library. This deterministic fixture uses no checkpoint.

```bash
out="$PWD/build/fp8-alias"
xmake f -P "$PWD" -o "$out" "${release[@]}"
xmake build -P "$PWD" sglang_cuda
"$CXX" -std=c++17 -O3 -Wall -Wextra -Werror -Iinclude -I"$toolkit/include" \
    tests/decoder/decode_fp8_alias_check.cpp -L"$out/lib" \
    "${cuda_lib[@]}" "${operator_lib[@]}" -o "$out/decode-fp8-alias-check"
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" "$out/decode-fp8-alias-check"
```

Expect exit zero and `FP8 alias checks passed`. Lawful aliases must match ordinary GEMV and SwiGLU. Rejected aliases return `cudaErrorInvalidValue` before kernel submission.

## Decode bandwidth

Build `sglang-generate` for the production decoder and CUDA libraries.

```bash
out="$PWD/build/decode-bandwidth"
xmake f -P "$PWD" -o "$out" "${release[@]}"
xmake build -P "$PWD" sglang-generate
"$CXX" -std=c++17 -O3 -Wall -Wextra -Werror -Iinclude -I"$toolkit/include" \
    tests/decoder/decode_bandwidth_check.cpp -L"$out/lib" \
    "${cuda_lib[@]}" "${operator_lib[@]}" -o "$out/decode-bandwidth-check"
"$CXX" -std=c++17 -O3 -Wall -Wextra -Werror -Iinclude -I"$toolkit/include" \
    tests/decoder/decode_graph_check.cpp "${cuda_lib[@]}" -L"$toolkit/lib/stubs" \
    -Wl,--wrap=cudaGraphLaunch -Wl,--wrap=cudaStreamBeginCapture \
    -Wl,--start-group "$out"/lib/libsglang_*.a -Wl,--end-group \
    "${nccl_lib[@]}" -lcuda -lcudart -pthread -o "$out/decode-graph-check"
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" "$out/decode-bandwidth-check"
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/decode-graph-check" "$checkpoint"
```

The operator probe matches BF16 GEMV and FP8 quantization bits and guards, and checks launch counts and dimensions on optimized and fallback paths. Four-lane head cases cover the first row pair, full and partial blocks, odd rows, unaligned input or weights, and another hidden width. Invalid inputs submit zero kernels.

The checkpoint probe requires one capture on the first default request and three launches on each of two default requests using the same graph. Eager captures and launches zero. All three match greedy IDs and full score bits.

## Grouped decode LM head

Use the Release archives above. Check the width-four kernel, argmax, seeded sampler, and host winner copies for sparse slots at different positions. Cover eager, capture, replay, capture rejection, width changes, mixed prefill/decode, and slot reuse.

```bash
sampler=_ZN6sglang7decoder13sample_logitsEPK13__nv_bfloat16iPiNS0_18SamplingParametersEiPvP11CUstream_st
"$CXX" -std=c++20 -O3 -Wall -Wextra -Werror -Iinclude -I"$toolkit/include" \
    tests/decoder/grouped_head_routes.cpp "${cuda_lib[@]}" -L"$toolkit/lib/stubs" \
    -Wl,--wrap=cudaLaunchKernel \
    -Wl,--wrap=cudaStreamBeginCapture -Wl,--wrap=cudaStreamEndCapture \
    -Wl,--wrap=cudaGraphInstantiate -Wl,--wrap=cudaGraphLaunch \
    -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap="$sampler" \
    -Wl,--start-group "$out"/lib/libsglang_*.a -Wl,--end-group \
    "${nccl_lib[@]}" -lcuda -lcudart -pthread -o "$out/grouped-head-routes"
jq -r '.prompt[]' tests/prefix/route-cases.json >"$out/prompt.ids"
export CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}"
for mode in eager graph rejected; do
    "$out/grouped-head-routes" "$checkpoint" "$out/prompt.ids" candidate "$mode" \
        "$out/$mode.bf16" >"$out/$mode.log"
done
```

Link the same probe against the original archives and pass `base` instead of `candidate`. Match each mode's `token` lines and binary files. Each file holds one full BF16 logit row per returned token in log order.

## GDN prefill

The opt-in operator probe uses synthetic input. For the route probe, start an isolated [LMCache daemon](../../docs/serving/lmcache.md#start-the-daemon) on the reserved GPU with namespace `sglang`. Set `LMCACHE_SOCKET` to its socket.

```bash
out="$PWD/build/gdn-checks"
xmake f -P "$PWD" -o "$out" "${release[@]}" --gdn-check=y
xmake build -P "$PWD" gdn-check gdn-routes
jq -r '.prompt[]' tests/prefix/route-cases.json > "$out/prompt.ids"
export CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}"
"$out/bin/gdn-check"
"$out/bin/gdn-check" short paired "$out/ordered.bin"
"$out/bin/gdn-check" chunked
LMCACHE_SOCKET="$LMCACHE_SOCKET" "$out/bin/gdn-routes" "$checkpoint" "$out/prompt.ids"
```

Check matrix and recurrence launches, FP64 output and state parity within tolerances, causality, and workspace guards. The checkpoint probe covers the first absolute tile, prefix restore, mixed prefill/decode, retired lane reuse, and cold versus restored continuation.

## Checkpoint metadata

Both checks run on the host. Compare `/proc/self/maps` offsets with an independent read of the index and shard headers. Fixtures cover sparse offsets above 4 GiB, historical JSON number spellings, and invalid or truncated metadata.

```bash
out="$PWD/build/checkpoint-metadata"
xmake f -P "$PWD" -o "$out" "${release[@]}"
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

The operator probe uses synthetic MXFP8 attention. The graph probe cycles checkpoint token IDs.

```bash
out="$PWD/build/long-decode"
xmake f -P "$PWD" -o "$out" "${release[@]}" --long-decode-check=y
xmake build -P "$PWD" long-decode-check long-decode-routes
python3 -c 'import json, sys; print(*json.load(open(sys.argv[1]))["prompt"])' \
    tests/prefix/route-cases.json >"$out/prompt.ids"
for mode in full recovery small unaligned; do
    CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
        "$out/bin/long-decode-check" "$mode"
done
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/bin/long-decode-routes" "$checkpoint" "$out/prompt.ids"
```

The operator probe covers split/combine geometry, shuffled pages, contexts to 262,144 tokens, FP64 recovery, guards, invalid input, and launch faults. The checkpoint probe captures and replays singleton graphs across sparse slots, mixed lengths, seeded samples, and slot reuse. Count split/combine on eligible multi-lane B200 decode and require zero replay masks there.

An optional `LOGITS` path records raw BF16 logits for two groups of three
width-five steps. An optional `TEACHER` file after `LOGITS` supplies 15
whitespace-delimited IDs for the final three steps, ordered by step and ascending
slot. To align the histories from the start of decode, put one additional ID at the
beginning of the file. The check forces that ID as the first output in all five lanes.

### K4 boundary crossings

Build `long-decode-routes` with the commands above. Run the prompt generator with
Python and `tokenizers` (available in the pinned vLLM environment). It writes one
fixed 32,768-token prompt per lane and verifies the generated IDs against a digest.

```bash
python -B tests/decoder/make_k4_route_prompts.py "$checkpoint/tokenizer.json" 32 > "$out/k4-32.ids"
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
    "$out/bin/long-decode-routes" "$checkpoint" --mtp-crossings "$out/k4-32.ids"
```

The check requires full K4 acceptance and the expected output IDs. Each wave
must launch one grouped long reader and combine per GQA layer, with zero legacy
verification readers.

### Mixed K4 verification

Use the same prompt generator with `16` or `32` lanes. Set `mixed_ids` to the
resulting ID file.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
    "$out/bin/long-decode-routes" "$checkpoint" --mtp-mixed "$mixed_ids"
```

The check combines requests with all five rows eligible for the long reader,
requests with only a long suffix, and requests below position 32,768. It checks
one grouped reader and combine per GQA layer, ordinary eager readers, and
the expected output IDs for long-eligible requests.
