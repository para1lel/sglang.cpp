# Native prefix cache

Native reuses GQA, GDN, and draft checkpoints through a local LMCache daemon. The daemon moves payloads between live GPU allocations and CPU L1.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Configure prefix reuse, validate salts, and report usage. | [Salt validation](../apps/server/text_request.cpp#L619)<!-- prefix: "void set_cac" --> · [Usage writer](../apps/server/text_completion.cpp#L50)<!-- prefix: "void append_" --> |
| HTTP controls | Dispatch cache reset, metrics, and bounded scheduler profiling. | [Control routes](../apps/server/prefix_controls.cpp#L483)<!-- prefix: "void Connect" --> |
| Native cache | Builds prefix identities and describes GPU allocation regions. | [Prefix keys](../src/runtime/cache/prefix_payload.cpp#L108)<!-- prefix: "PrefixCacheK" --> · [Client](../include/sglang/runtime/cache/lmcache_client.h#L56)<!-- prefix: "class LmCach" --> |
| LMCache | Resolves coverage, executes transfers, and retains CPU L1 records across native workers. | [Lookup](../tools/lmcache/sglang_lmcache.py#L165)<!-- prefix: "def coverage" --> · [Storage](../tools/lmcache/sglang_lmcache.py#L464)<!-- prefix: "class Record" --> |

## LMCache daemon

Start one [LMCache 0.5.8 daemon](serving/lmcache.md#start-the-daemon) as the native server’s Linux user. Expose the same physical GPU set to both processes and use namespace `sglang`. This includes both ranks in TP2.

## Runtime path

Daemon reuse is on by default. Set an absolute `--lmcache-socket`, or disable reuse with `--prefix-cache-entries 0`; positive entries are invalid. `--eager` independently controls [batch and TP2 replay](batch-graphs.md#runtime-path).

Text, chat, benchmark IDs, and `/generate` accept `cache_salt` of up to 256 UTF-8 bytes, default empty. Invalid salts return 400 even with reuse off. See [native per-item and null handling](native-generate.md#constraints-and-failure-behavior). C++ uses `DecoderOptions::prefix_cache_salt`. Reuse supports greedy and sampling, plus text/chat finite choices. Benchmark IDs stay greedy and positive MTP requires draft weights.

OpenAI `prompt_tokens` counts the whole prompt. `prompt_tokens_details.cached_tokens` counts coherently restored target IDs, or zero on a miss or when disabled. GQA-only coverage has separate metrics. [Native usage](native-generate.md#json-and-sse) uses a different field. Zero output runs prefill only for prompt scores.

### Checkpoint boundaries

Let `a` be the restored GDN endpoint or zero, and `b` the exact prompt end. The planner selects `b` and every `b − 2^k > a` for `k ≥ 7`. Prefill visits these exact token counts in ascending order. It skips endpoints already usable with the required GQA and draft coverage. The scheduler reserves bounded snapshot storage and can defer a grant while peers advance. [Planner](../src/runtime/execution/decoder_batch_prefix.cpp#L116)<!-- prefix: "void Decoder" --> · [Snapshot admission](../src/runtime/execution/decoder_batch_prefix.cpp#L155)<!-- prefix: "DecoderBatch" -->

Each new snapshot records GDN state and the final hidden and residual rows at its endpoint. GQA and available shifted draft are exported through that endpoint. This also covers zero-output requests that prefill for scores. [Snapshot capture](../src/runtime/execution/decoder_prefill_tiles.cpp#L471)<!-- prefix: "void capture" --> · [Publication](../src/runtime/execution/decoder_batch_prefix.cpp#L226)<!-- prefix: "void Decoder" -->

Successful completion can publish one terminal checkpoint at committed target position `U` when it advances beyond `b`. With one output, the prompt already supplies `U = b`, so there is no duplicate. Stop and EOS can change how many target rows are consumed. Use the committed cursor to determine `U`. Intermediate decode rounds publish no checkpoints. [Terminal store](../src/runtime/execution/decoder_batch_prefix.cpp#L86)<!-- prefix: "void Decoder" -->

A later request that preserves the exact generated token prefix can resume at `U`, subject to ordinary GQA, draft and L1 coverage. If the target never consumed the final emitted ID, the follow-up prefills it with its suffix. The physical generated span is `O = U − b` processed rows. Count it once in the logical endpoint and account for acknowledged GQA, shifted draft and materialized GDN bytes separately. [Export](../src/runtime/execution/decoder_batch_prefix.cpp#L214)<!-- prefix: "void Decoder" --> · [GQA and draft encoding](../src/runtime/cache/lmcache_client.cpp#L274)<!-- prefix: "void append_" -->

### Checkpoint compatibility and resume

GQA and draft records hold contiguous MXFP8 chunks of up to 256 rows. A GDN checkpoint holds convolution history, recurrent state, and the final BF16 hidden and residual rows. Each record is keyed by its prefix end. [Chunk encoding](../src/runtime/cache/lmcache_client.cpp#L274)<!-- prefix: "void append_" --> · [Checkpoint encoding](../src/runtime/cache/lmcache_client.cpp#L298)<!-- prefix: "void append_" -->

The daemon tracks resident GDN endpoints from L1 write and deletion events and tests their exact prefix digests. It returns matching GDN endpoints in descending order and the longest contiguous GQA match independently. Choose the largest common GDN endpoint within both GQA coverage and the request's restore limit. With MTP, rank 0 also needs contiguous shifted draft through `P−1` for an endpoint `P`. [Resident endpoints](../tools/lmcache/sglang_lmcache.py#L67)<!-- prefix: "class GdnEnd" --> · [Coverage](../tools/lmcache/sglang_lmcache.py#L165)<!-- prefix: "def coverage" --> · [Rank intersection](../src/runtime/cache/lmcache_group.cpp#L59)<!-- prefix: "PrefixCacheR" -->

Resume at the usable GDN boundary, or row zero when there is none. Recompute intervening GDN rows and read restored GQA through its independent coverage. Append GQA only for uncovered rows. Native commits the extents after successful transfer on every rank. [Ranked restore](serving/lmcache.md#ranked-tp2-transport) · [Restore view](../src/decoder_prefix_cache.cpp#L375)<!-- prefix: "PrefixDevice" --> · [Commit](../src/decoder_prefix_cache.cpp#L437)<!-- prefix: "int commit_r" -->

`sglang_cpp:lmcache_gqa_computed_tokens_total` counts regenerated target GQA cache rows. Restored rows skip key normalization, rotary and MXFP8 cache append. A tile fully covered by GQA skips K/V projection; a tile crossing the boundary can still run packed QKV. [Tile projection](../src/runtime/execution/decoder_prefill_tiles.cpp#L270)<!-- prefix: "void Chunk::" --> · [Tile append](../src/runtime/execution/decoder_prefill_tiles.cpp#L285)<!-- prefix: "void Chunk::" --> · [Accounting](../src/decoder_batch.cpp#L784)<!-- prefix: "prefix_stats" -->

Even a fully published checkpoint can be evicted before the next lookup. The [AgentX pressure diagnostic](agentx-256k.md#focused-cache-pressure-diagnostic) records that sequence on a 64 GiB TP1 cache.

A paired boundary at `P` seeds shifted draft row `P-1` with the first suffix ID on a partial hit or the first committed output on a full hit. Full hits score from restored hidden/residual rows using the request's [grammar](text-serving.md#finite-choice-constraints) and [sampler](text-serving.md#sampling). Restore and full-hit scoring do not charge [target rows](batch-graphs.md#runtime-path) or change other lanes' decode policy.

Prompt scoring restores no farther than `logprob_start_len` and leaves at least the last prompt row to compute. Offset zero, including OpenAI echo scoring, restores no rows. Generated-only or [pointwise label scores](serving/pointwise-score.md) can use full hits. [Lookup limit](../apps/server/detail/batch_service_cache.cpp#L14)<!-- prefix: "const int ma" --> · [Full hit](../src/decoder_batch.cpp#L311)<!-- prefix: "if (lane.res" -->

[Archived TP2 replay receipts](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/prefix-cache.md#checkpoint-compatibility-and-resume) include eight cohorts, cache restores, cancellation and ranked payloads.

### Media identity and prefill

Keys bind covered media patch identity, layout, feature row, and rotary positions. A change inside the covered prefix changes its identity. External C++ features can run without a cache identity. Key construction stops before the first such media row. [Media key construction](../src/runtime/cache/prefix_payload.cpp#L130)<!-- prefix: "const auto e" --> · [Media execution](multimodal-decoder.md#runtime-path)

Cold and restored suffix rows use the same [prefill planner](native-prefill-workspace.md#runtime-path). The former exact-only cache route and its fixed-shape Tensor Core probe remain archived. [Historical cache guide](https://github.com/para1lel/sglang.cpp/blob/28148cda233e5522274e258d5eb5caa9ab83e313/docs/prefix-cache.md#tensor-core-scheduling)

## Constraints and failure behavior

Page-rounded GPU blocks stay live through readiness fences and L1 transfers. Nonterminal STORE retains GQA/draft and copies GDN state and BF16 hidden/residual. Its four source allocations stay live through copy completion, including cancellation. Unconfirmed cleanup quarantines both. The STORE freeing stream fences asynchronous frees. Terminal STORE retains the original blocks without copying. See [resident charges](batch-graphs.md#resident-memory-admission) and [transport](serving/lmcache.md#native-transport).

LOOKUP, GET, and STORE each have up to four workers. Prefill snapshots use bounded rank-local pools that reserve capacity for exact prompt ends. Terminal stores reserve separate queue capacity. A terminal finish claims completion before waiting, so a preceding abort can suppress that store. [Pool bounds](../src/runtime/execution/decoder_batch_prefix.cpp#L155)<!-- prefix: "DecoderBatch" --> · [Terminal claim](../src/runtime/execution/decoder_batch_prefix.cpp#L86)<!-- prefix: "void Decoder" -->

After submission the lane can advance or retire. STORE owns metadata, immutable views, and capacity through acknowledgement or lease release. The daemon serializes writes; a new request can miss an unacknowledged one. Refusals count as declines and transport faults as I/O errors. Failed writes leave delivered output intact and the checkpoint unacknowledged. Submitted stores may finish after abort and drain on normal shutdown.

Abort can abandon LOOKUP. GET holds its destination and lane through completion even after shutdown interrupts the socket. A disconnected armed GET or STORE holds its resources until matching lease release or original daemon exit. Timeouts and replacement daemons leave the lease live. A hung GPU can hold shutdown indefinitely. An authorized operator can stop new admissions and retire the daemon.

Failed restore or a non-CUDA future error clears partial state before cold prefill. CUDA errors propagate and terminal faults disable the batch. Poll outstanding restores after recoverable failure until retirement. Direct MTP polls restore and cancellation, reports `cached_prompt_tokens`, and waits for terminal submission. A successful `claim_completion` preserves the checkpoint across cancellation. `reserve_prefix_store: false` retries. Synchronous callers must release capacity independently. [Lease status](../tools/lmcache/sglang_lmcache.py#L663)<!-- prefix: "def lease_st" --> · [Restore](../src/runtime/execution/decoder_batch_prefix.cpp#L37)<!-- prefix: "DecoderBatch" --> · [Direct MTP](../src/decoder_batch.cpp#L837)<!-- prefix: "void Decoder" -->

### Native HTTP controls

The metrics, reset, and profile routes are available when LMCache is configured, even when the daemon is unreachable. The [HTTP admission and shutdown rules](http-server.md#http-admission-and-shutdown) also apply with reuse disabled.

| Route | Input | Result |
| --- | --- | --- |
| Health routes | Empty GET | Use the [liveness, readiness, and generation progress responses](http-server.md#health-and-readiness). |
| `POST /reset_prefix_cache` | `{}` | Returns `{"success":true}` after clearing the idle daemon store. Returns HTTP 409 while busy and HTTP 503 on daemon reset failure. |
| `GET /metrics` | Empty body | Returns Prometheus text with native provenance and `engine="0"`. |
| `POST /start_profile` | Exactly three fields. `profile_prefix` is a string of 1–64 bytes. `max_iterations` is an integer from 1 to 64. `profiler_kwargs` equals `{"with_stack":false}`. | Opens one capture and returns `{"success":true,"profiler":"native-batch-steps"}`. Returns HTTP 409 if a capture is open. |
| `POST /stop_profile` | `{}` | Closes the capture and adds `profile_prefix` and `steps` to the JSON response. Returns HTTP 409 if no capture is open. |

Reset and [flush](#cache-flush) clear the entire daemon store across workers and both TP2 ranks through one reset. Restrict forwarded access to these controls. Successful controls return 200 and malformed POST controls return 400. Reset stays busy through pending stores and waits for the daemon acknowledgement. Queued generations wait through the reset response write. Counters persist and health/readiness stay available.

`vllm:prefix_cache_hits_total` counts coherently restored prompt IDs. `vllm:generation_tokens_total` counts published IDs. `sglang_cpp:lmcache_` reports availability, misses, I/O errors, transfer rows, bytes, time, and pending stores. Joint acknowledgements count once. Rank receipts count successful rank replies. Prefill counters report planned, resident, published, and declined checkpoints, plus acknowledged stores and failures. [Metric series](../apps/server/loads.cpp#L264)<!-- prefix: "std::string " --> · [Prefill counters](../apps/server/loads.cpp#L300)<!-- prefix: "emit(\"sglang" -->

A profile samples completed nonempty `advance_chunked` iterations started after capture opened. Each sample has host `elapsed_ns`, `width`, `prefill_rows`, `decode_lanes` and the `graph_slots` mask. At the sample cap it stays open. [Sample fields](../apps/server/detail/batch_trace.cpp#L115)<!-- prefix: "BatchProfile" -->

### Cache flush

`GET /flush_cache` and `POST /flush_cache` work with prefix reuse on or off. Both ignore a successfully framed body. Other methods return HTTP 405 with `Allow: GET, POST`. HEAD omits the body. The optional `timeout` query parameter is a nonnegative number in seconds. The reader form-decodes query fields (`+` means space) and uses the last `timeout`. The default is zero. Positive infinity allows an indefinite wait for idle. [Native input](../apps/server/flush_cache.cpp#L90)<!-- prefix: "FlushCacheRe" --> · [Pinned handler](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1060)<!-- prefix: "@app.api_rou" -->

| Condition | HTTP response |
| --- | --- |
| Idle, including with reuse off | 200, `Cache flushed.\nPlease check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)\n` |
| Busy at zero timeout | 400, `Flush cache failed.\n` |
| Still busy when a positive timeout expires | 400, `Timed out waiting for idle state.` |
| Empty, malformed, negative, or NaN timeout | 400 JSON with `object="error"`, `type="Bad Request"`, `param=null`, `code=400`, and a `message` identifying `query.timeout`. |

The first three bodies are plain text without a `Content-Type` header. Each `\n` represents one line feed. The reader validates the timeout before queueing. [Replies](../apps/server/flush_cache.cpp#L107)<!-- prefix: "void write_f" --> · [Validation error](../apps/server/flush_cache.cpp#L75)<!-- prefix: "void write_i" -->

The worker claims flushes FIFO and starts the timeout at claim. Busy includes queued or claimed completions, generation and tokenization, active generation, a health job, pending stores, retiring GETs, and reset. Other requests may progress during a deferred wait. With reuse on, daemon reset failure returns 503. Tokenizer, compiled tool grammars, and cumulative counters persist. [Scheduler guard](../apps/server/batch_service.cpp#L795)<!-- prefix: "bool BatchSe" --> · [Pinned upstream compatibility details](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/prefix-cache.md#cache-flush).

## Verification

Run from the repository root with the Release binary (`SGLANG_SERVER`), pinned checkpoint (`CHECKPOINT`), and [frozen W3 fixtures](../benchmarks/concurrent/README.md#regenerate-and-measure) (`FIXTURES`). Set `EVIDENCE` to a new directory. Set `LMCACHE_PYTHON` to the [LMCache 0.5.8 interpreter](serving/lmcache.md#start-the-daemon) and `GPU_UUID` to the reserved B200. Choose a free loopback TCP port for `PORT`. Install Linux `strace` with nanosecond timestamps on `PATH` for the terminal-write check. The test launcher installs `install_device_fence` inside the private daemon before it serves. The fixture resets this daemon repeatedly.

```bash
(
set -eu
export CUDA_VISIBLE_DEVICES="${GPU_UUID:?}"
FENCE_DIR=$(mktemp -d /tmp/sglang-fence.XXXXXXXX)
LMCACHE_SOCKET="$FENCE_DIR/cache.sock"
SGLANG_TEST_PYTHON="${LMCACHE_PYTHON:?}" SGLANG_TEST_FENCE_DIR="$FENCE_DIR" \
    LMCACHE_PYTHON="$PWD/tests/prefix/fenced_python.sh" \
    bash tools/lmcache/serve.sh --socket "$LMCACHE_SOCKET" --namespace sglang \
        --port "${PORT:?}" --memory-gib 4 >"$FENCE_DIR/daemon.log" 2>&1 &
daemon_pid=$!
trap 'kill "$daemon_pid" 2>/dev/null || true; wait "$daemon_pid" || true' EXIT
PYTHONPATH="$PWD" python3 -B benchmarks/paired/lmcache_evidence.py \
    ready "$LMCACHE_SOCKET" "$daemon_pid" --wire SLC3
PYTHONPATH="$PWD" python3 -B tests/prefix/lmcache_routes.py \
    "$SGLANG_SERVER" "$CHECKPOINT" "$FIXTURES" "$LMCACHE_SOCKET" "$EVIDENCE" \
    --device-fence-dir "$FENCE_DIR"
)
```

Require exit zero and an empty `errors` array in `relay.json`. In `summary.json`, require an empty `failures` object and all three results below.

- `cancelled_armed_get` must report `armed_ns < active_ns < abort_ns < peer_progress_ns < fence_released_ns < lease_released_ns < reuse_ns < peer_progress_after_reuse_ns` and `replacement_cached_tokens == 1391`. The peer advances while the aborted GET still owns its destination. [Armed abort](../tests/prefix/lmcache_routes.py#L552)<!-- prefix: "def abort_ar" -->
- `shutdown_armed_get` must report `exit_code == 0`, `active_ns < released_ns`, and `active_ns < held_reply_ns`. The server stays alive through the active lease, then exits after the matching release. The daemon still answers READY. [Armed shutdown](../tests/prefix/lmcache_routes.py#L600)<!-- prefix: "def shutdown" -->
- `exact_terminal.endpoint` must equal `97` for the 96-token fixture. For both `exact_terminal.ordering.length` and `.stop`, require `owner_send_end_unix_ns < barrier_open_unix_ns < claimant_send_start_unix_ns`. The deferred claimant sends only after the STORE reply barrier opens. Require one GDN checkpoint at 97 and a 97-row hit on exact continuation. [Boundary fixture](../tests/prefix/lmcache_routes.py#L704)<!-- prefix: "def exact_te" -->

The same run covers selective reuse, divergence, salt and shifted draft reuse, held LOOKUP and STORE, cancellation, and daemon reuse across worker restarts. [Device state and graph controls](../tests/prefix/README.md#device-state-and-graph-controls) compare native payloads and scores through that daemon.

### Live control regression

Check malformed controls, busy/idle reset, bounded profiles, and peer graph eligibility. After cold reset require zero cached IDs. On replay after acknowledge and drain, require the expected restored length and identical first scores, IDs, finish and usage. [Former resident-cache procedure](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/docs/prefix-cache.md#verification). [In-process fence installation](../tests/prefix/lmcache_relay.py#L486)<!-- prefix: "def install_" --> · [Terminal send trace](../tests/prefix/lmcache_send_trace.py#L31)<!-- prefix: "def terminal" -->

### Flush verification

At idle, try both flush methods with reuse on and off. Check the [status, body, and headers](#cache-flush), including `?timeout=invalid`. With reuse on, prime a cacheable prompt and repeat it. Verify that a successful flush preserves cumulative counters and that the next two requests miss, then hit.

Hold a stream active and send three overlapping HTTP flushes in order with short, short, and zero timeouts. Keep the stream active through both deadlines. Expect two timeout replies followed by the busy reply. Verify that a failed flush preserves a primed hit. Repeat with two longer waits and finish the stream. Expect both flushes to succeed in order. Check `/health` while the flush worker waits.

### W3-prefix

Use the current [K4 warm-prefix diagnostic](../benchmarks/paired/README.md#prefix-reuse-diagnostic) for both engines. The [historical K0 W3-prefix procedure](https://github.com/para1lel/sglang.cpp/blob/28148cda233e5522274e258d5eb5caa9ab83e313/docs/prefix-cache.md#w3-prefix) preserves the 25,600 restored and 7,168 computed row gates.
