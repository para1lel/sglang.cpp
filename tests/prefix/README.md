# Prefix cache CUDA and route checks

Run from the repository root in the managed Pixi default environment. Reserve a B200 and set
`GPU_UUID` to its UUID. The state check runs without model weights.

```bash
out="$PWD/build/prefix-checks"
xmake f -P "$PWD" -o "$out" -p linux -a x86_64 -m release
xmake build -P "$PWD" sglang-server
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
"$CXX" -std=c++17 -O3 -Wall -Wextra -Werror -Iinclude -I"$toolkit/include" \
    tests/prefix/cache_state.cpp -L"$out/lib" -L"$toolkit/lib" \
    -Wl,-rpath,"$toolkit/lib" -Wl,--wrap=cudaMemcpy2DAsync \
    -lsglang_decoder -lsglang_checkpoint -lsglang_json -lsglang_cuda -lsglang_prefill_fp8_sm100 \
    -lcudart -o "$out/prefix-state"
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" "$out/prefix-state"
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/prefix-state" --full
```

Expect `prefix state passed`. The fixture checks the four MXFP8 planes across all full-attention
layers, untouched tail pages, recurrent and convolution data, exact-hit hidden state, salt
isolation, byte accounting, and eviction. The remapped cases permute either page table and then
both page tables independently. Failed matches must leave the destination completely untouched.
The source is overwritten after insertion to check that the cache owns a snapshot. The
`--full` case verifies all 50 pages through independent permutations at the 6,400-token limit.
The guard cases reject invalid indices, duplicate prefix pages, tail aliasing, invalid
capacities, and bad checkpoint inputs before they touch live state or evict a cache entry.
Injected transfer failures check that an interrupted store publishes no entry and an
interrupted restore records no hit. Both paths must allow a clean retry.
Run `--identity`, `--remapped`, or `--guards` to isolate one mode.

For the route check, use the local Qwen3.8-27B-FP8 checkpoint and the same reserved device.
Use an unused port and a fresh output directory.

```bash
port=17472
checkpoint=/data0/shared/Qwen3.8-27B-FP8
for mode in enabled disabled; do
    cache_flags=()
    check_flags=()
    if [[ $mode == enabled ]]; then
        cache_flags+=(--prefix-cache-entries 2)
    else
        check_flags+=(--disabled)
    fi
    CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/bin/sglang-server" --checkpoint "$checkpoint" \
        --port "$port" --fp8-prefill exact --max-active 4 --batch-cuda-graphs \
        "${cache_flags[@]}" >"$out/$mode.log" 2>&1 &
    server=$!
    trap 'kill "$server" 2>/dev/null || true' EXIT
    curl -fsS --retry 30 --retry-connrefused --retry-delay 1 \
        "http://127.0.0.1:$port/health"
    python3 -B tests/prefix/routes.py --port "$port" --evidence "$out/$mode-routes" \
        "${check_flags[@]}"
    kill -TERM "$server"
    wait "$server"
    trap - EXIT
done
```

The expected greedy IDs were captured with fresh exact-prefill requests on revision
`d69c77ffaff160f2b5693d82455328cda513e171`. The script checks request and usage metadata,
token IDs around page boundaries, cancellation with a concurrent peer, a replacement request,
the omitted cache flag, and salt handling on all completion routes.

For builds with finite EBNF, capture a fresh exact-prefill reference with the cache flag
omitted. Start a separate cache-enabled exact server with at least four active lanes and
run the warm check against the reference. Use a new evidence directory in each command.

```bash
python3 -B tests/prefix/grammar_routes.py reference --port "$cold_port" \
    --evidence "$out/cold-grammar" --reference "$out/grammar-reference.json"
python3 -B tests/prefix/grammar_routes.py warm --port "$warm_port" \
    --evidence "$out/warm-grammar" --reference "$out/grammar-reference.json"
```

Use the same checkpoint and keep MTP off on both servers. The warm check covers an exact
hit, an unaligned suffix, a changed seed, a changed grammar, UTF-8 chat streaming, and an
invalid grammar followed by a successful hit. It repeats the choices across concurrent
warm and cold lanes. The test compares output IDs, decoded choices, finish reasons, and
reported cache hits.
