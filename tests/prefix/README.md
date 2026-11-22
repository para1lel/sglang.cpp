# Prefix cache checks

## CPU reader queue

Run from the repository root in the managed Pixi default environment. The test drives the
production HTTP reader and completion queue over loopback. It uses a decoder-service double
and explicit barriers for final responses and reset acknowledgement. It needs no GPU or weights.

```bash
out="$PWD/build/prefix-readers"
mkdir -p "$out"
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -pthread -ffunction-sections -fdata-sections \
    -Iinclude -I"$toolkit/include" tests/prefix/reader_queue.cpp \
    apps/server/{prefix_controls,chat_output,flush_cache,http,loads,completions}.cpp \
    src/{json,utf8}.cpp \
    -Wl,--gc-sections \
    -Wl,--wrap=accept4 -o "$out/reader-queue"
for mode in --cache-off --cache-on --reader-failure --worker-failure --flush; do
    timeout 20s "$out/reader-queue" "$mode"
done
```

Each mode prints `HTTP reader queue passed` on success. The cache modes check health and
readiness during partial header and body reads, replacement of the oldest stalled reader, and
completion queue overflow, cancellation, FIFO order, and slot reuse. Fully buffered 48 KiB
requests reach idle workers or wait in the saturated queue. Requests beyond the queue capacity
receive HTTP 503. An incomplete header and body both expire on the original read deadline even
when the clients send more bytes. A held final response and a blocked write to a client that stops
reading make reset return HTTP 409 until the completion releases its slot. Health and readiness
respond while reset acknowledgement is held, and the queued completion starts after reset.
Shutdown closes active, queued, and partial connections. The failure modes inject a fatal accept
error and an unavailable decoder worker, and check error propagation, readiness, and descriptor
cleanup.

The flush mode holds a completion at its final response. It checks ordered timeout budgets,
busy replies, reader responsiveness, and separate reset acknowledgements after the lane is idle.

## CUDA state and routes

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

For tensor-core cache parity, prepare the frozen W3 variant fixtures described in
[the variant benchmark](../../benchmarks/variants/README.md) and run the following
with the same release build, checkpoint, and reserved B200.

```bash
GPU_UUID="$GPU_UUID" bash tests/prefix/run_tensor_state.sh \
    "$out" "$checkpoint" "$fixtures" "$out/tensor-prefix-state"
```

The probe compares a cache-disabled width-one `DecoderBatch` oracle with a restore
alongside a decoding peer. It runs page boundaries and the frozen 6,400-token checkpoint
with an 8,192-token request, remaps source and destination pages independently, and
cancels before an incomplete prefix can publish. It compares active KV payloads and
scales, GDN history and recurrence, hidden and residual vectors, the seeded first ID,
and every BF16 logit.

## Prefix cache with MTP

Use an exact-prefill release build and a reserved B200. Supply the frozen 257- and 8192-ID
files used by the matching-prefix checks. Run from the repository root with the managed
Pixi default environment active.

```bash
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
draft=$(nm --defined-only "$out/lib/libsglang_decoder.a" | \
    awk '$3 ~ /^_ZN6sglang7decoder15MtpDraftRequest7enqueue/ && $3 !~ /[.]/ {print $3}')
test -n "$draft"
"$CXX" -std=c++17 -O3 -pthread -fno-access-control -Iinclude -Isrc -I"$toolkit/include" \
    tests/prefix/mtp_state.cpp -L"$out/lib" -L"$toolkit/lib" -Wl,-rpath,"$toolkit/lib" \
    -Wl,--wrap=cudaMemcpy2DAsync -Wl,--wrap="$draft" \
    -lsglang_decoder -lsglang_checkpoint -lsglang_json -lsglang_cuda -lsglang_prefill_fp8_sm100 \
    -lsglang_tokenizer -lcudart -o "$out/prefix-mtp-state"
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/prefix-mtp-state" "$checkpoint" "$short_ids" "$long_ids"
CUDA_VISIBLE_DEVICES="$GPU_UUID" python3 -B tests/prefix/mtp_routes.py \
    --server "$out/bin/sglang-server" --checkpoint "$checkpoint" --output "$out/mtp-routes"
```

The native check compares exact target KV, GDN state, and full first logits against fresh
requests. It checks a K0 prime followed by a positive-depth miss and same-key enrichment,
K1–K4 consumers, K0 full and partial hits on a paired entry, and a changed suffix that shifts
prefill tiles. The link observer checks the first rebuilt draft row against the new prompt
suffix on a partial hit and the current selected first token on a full hit. Draft-side copy
faults exercise interrupted publication, restore, and retry. Cache accounting checks the
combined resident charge at capacities one and four.

The HTTP check compares seeded, constrained text and chat requests at depths one and four
with fresh depth-zero controls. It checks the IDs, stop, usage, draft metrics, real verifier
rounds, and cache reset. It saves the effective commands, wire responses, and traces.
