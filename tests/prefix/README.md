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
"$CXX" -std=c++17 -O3 -Wall -Wextra -Werror -pthread -Iinclude -I"$toolkit/include" \
    tests/prefix/cache_state.cpp -L"$out/lib" -L"$toolkit/lib" \
    -Wl,-rpath,"$toolkit/lib" -Wl,--wrap=cudaMemcpy2DAsync -Wl,--wrap=cudaMalloc \
    -Wl,--wrap=cudaMemGetInfo \
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
`--full` case verifies all 64 pages at the first 8,192-row cadence boundary through
independent permutations. The fixture checks the 262,144-token cap without allocating
a full-context snapshot.
The guard cases reject invalid indices, duplicate prefix pages, tail aliasing, invalid
capacities, and bad checkpoint inputs before they touch live state or evict a cache entry.
Injected transfer failures check that an interrupted store publishes no entry and an
interrupted restore records no hit. Both paths must allow a clean retry.
Run `--identity`, `--remapped`, or `--guards` to isolate one mode.

For HTTP cache coverage, run the [feature-complete route test](../media/http-contract.md) on the same reserved device and checkpoint. Compare cold, exact-hit, and suffix-hit usage with the matched cache-off arm. Cover full and partial pages, media identity, salt isolation, cancellation, and slot reuse.

The [archived route procedure](https://github.com/para1lel/sglang.cpp/blob/28148cda233e5522274e258d5eb5caa9ab83e313/tests/prefix/README.md#cuda-state-and-routes) used fixed greedy IDs and page-rounded checkpoint expectations. Its oracle was captured at `d69c77ffaff160f2b5693d82455328cda513e171`.

For builds with finite EBNF, start a cold server with `--prefix-cache-entries 0`.
Start a warm server with at least four active lanes and prefix reuse enabled. Match the
checkpoint, backend, and step budget. Use a new evidence directory in each command.

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

### Committed target state

Run the native state comparison against the same checkpoint and the frozen long-token file.
The runner takes its 128-token W1 prompt from the streaming benchmark fixture.

```bash
GPU_UUID="$GPU_UUID" bash tests/prefix/run_tensor_state.sh \
    "$out" "$checkpoint" "$long_ids" "$out/tensor-prefix-state"
```

The check covers exact and suffix hits, independent page permutations, interrupted prefill,
and periodic publication at 8,192 and 16,384 while the request still has work. It compares
KV payload and scales, GDN state, hidden and residual vectors, and full first BF16 logits.
Both eager and graph W1 generation verify one 255-row snapshot after 128 emitted tokens.
The restore processes the last emitted ID as a one-token suffix.
The completion-claim check verifies that a denied claim at 8,191 has no final snapshot
and committed cadence at 8,192 retains exactly one checkpoint.

### Historical Tensor Core cache parity

This archived check predates the shared prefill executor. Use the pinned revision below
for both arms. On current builds, cache enablement does not select a separate prefill backend.

To reproduce the Tensor Core parity check, use revision
`39ecda9f370d0c3605296e6a028ef3561294d6e7` for both the release build and the test runner.
Prepare that revision's frozen W3 variant fixtures and use the same checkpoint and reserved
B200. [W3 fixture preparation](https://github.com/para1lel/sglang.cpp/blob/39ecda9f370d0c3605296e6a028ef3561294d6e7/benchmarks/variants/README.md)

```bash
GPU_UUID="$GPU_UUID" bash tests/prefix/run_tensor_state.sh \
    "$out" "$checkpoint" "$fixtures" "$out/tensor-prefix-state"
```

On that revision, both batches select Tensor Core prefill. The probe compares a fresh
width-one request with a restored request alongside a decoding peer. It checks page
boundaries and an 8,192-token request with a 6,400-token checkpoint. It remaps source and
destination pages independently and checks that canceling incomplete prefill leaves no entry.
It compares active KV payloads and scales, GDN history and recurrence, hidden and residual
vectors, the seeded first ID, and every BF16 logit.

## Prefix cache with MTP

Use a release build and a reserved B200. Supply the frozen 257- and 8192-ID files used by
the matching-prefix checks and the `w1.ids` emitted by the committed target runner above.
Run from the repository root with the managed Pixi default environment active.

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
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/prefix-mtp-state" \
    "$checkpoint" "$short_ids" "$long_ids" "$out/tensor-prefix-state/w1.ids"
CUDA_VISIBLE_DEVICES="$GPU_UUID" python3 -B tests/prefix/mtp_routes.py \
    --server "$out/bin/sglang-server" --checkpoint "$checkpoint" --output "$out/mtp-routes"
```

Both native batches use the same backend policy and natural prefill boundaries. The test
resets the cold cache before each admission and checks that it computes the whole prompt
without restoring target or draft rows. At matched boundaries it compares target KV, GDN
state, and full first logits byte for byte. [Cold control](mtp_state.cpp#L60)<!-- prefix: "void check_p" -->

The fixture checks a K0 prime followed by a positive-depth miss and same-key enrichment.
It covers K1–K4 consumers, K0 full and partial hits on a paired entry, and a changed suffix
that shifts prefill tiles. The link observer checks the first rebuilt draft row against
the new prompt suffix on a partial hit and the current selected first token on a full hit.
Draft-side copy faults exercise interrupted publication, restore, and retry. Cache
accounting checks the combined resident charge at capacities one and four. The committed K4
fixture reaches 8,192 before completing a turn, then stops on a round with real proposal
rejection. It checks that rejected rows are excluded from the terminal key and compares
restored target and draft KV, GDN state, the final committed workspace, and all first logits.
The last emitted ID runs as the next-turn suffix.

The HTTP check compares seeded, constrained text and chat requests at depths one and four
with fresh depth-zero controls. It checks the IDs, stop, usage, draft metrics, real verifier
rounds, and cache reset. It saves the effective commands, wire responses, and traces.
