# Native prefix cache

Native keeps GQA pages and exact GDN checkpoints in separate GPU pools. A background worker backs resident data up to a local LMCache daemon under pressure. [Resident pools](../src/runtime/cache/prefix_residency.cpp#L538)<!-- prefix: "PrefixReside" --> · [Pressure worker](../src/runtime/cache/prefix_residency.cpp#L470)<!-- prefix: "void PrefixR" -->

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Configure prefix reuse, validate salts, and report usage. | [Salt validation](../apps/server/text_request.cpp#L627)<!-- prefix: "void set_cac" --> · [Usage writer](../apps/server/text_completion.cpp#L52)<!-- prefix: "void append_" --> |
| HTTP controls | Dispatch cache reset, metrics, and bounded scheduler profiling. | [Control routes](../apps/server/prefix_controls.cpp#L431)<!-- prefix: "void Connect" --> |
| Native cache | Builds prefix identities, leases GPU pages and checkpoints, and schedules CPU backup. | [Prefix keys](../src/runtime/cache/prefix_payload.cpp#L108)<!-- prefix: "PrefixCacheK" --> · [Pool capacity](../src/runtime/cache/prefix_residency.cpp#L144)<!-- prefix: "PoolLimits p" --> |
| LMCache | Resolves coverage, executes transfers, and retains CPU L1 records across native workers. | [Lookup](../tools/lmcache/sglang_lmcache.py#L165)<!-- prefix: "def coverage" --> · [Storage](../tools/lmcache/sglang_lmcache.py#L469)<!-- prefix: "class Record" --> |

## LMCache daemon

For CPU backup, start one [LMCache 0.5.8 daemon](serving/lmcache.md#start-the-daemon) as the native server’s Linux user. Expose the same physical GPUs to both processes and use namespace `sglang`. TP2 shares the daemon across both ranks.

## Runtime path

Native enables GPU prefix reuse and configures LMCache by default. GPU reuse also works when the daemon is unavailable. Use an absolute `--lmcache-socket` to select the daemon. Use `--prefix-cache-entries 0` to disable both tiers. Positive values are invalid. `--prefix-gqa-pages` and `--prefix-gdn-slots` default to zero for automatic sizing. Set positive values to constrain a pressure test. `--eager` independently controls [batch and TP2 replay](batch-graphs.md#runtime-path). [Server options](../apps/sglang-server.cpp#L217)<!-- prefix: "} else if (o" --> · [Daemon lookup](../apps/server/batch_cache.cpp#L463)<!-- prefix: "void BatchCa" --> · [Startup pools](serving/request-memory.md)

Automatic sizing first uses a base GDN allocation to size the target and draft page pools. It keeps those page counts when it expands the GDN pool into the remaining cache arena. Expansion uses the smallest remaining fit across ranks and leaves one slot-sized span per configured lane for callback snapshots. It preserves the base allocation if there is no room to expand. A positive `--prefix-gdn-slots` stays on the explicit cap path. [Base sizing](../src/runtime/cache/prefix_residency.cpp#L144)<!-- prefix: "PoolLimits p" --> · [Automatic expansion](../src/runtime/cache/prefix_residency.cpp#L175)<!-- prefix: "if (enabled " -->

Text, chat, benchmark IDs, and `/generate` accept `cache_salt` of up to 256 UTF-8 bytes, default empty. Invalid salts return 400 even with reuse off. See [native per-item and null handling](native-generate.md#constraints-and-failure-behavior). C++ uses `DecoderOptions::prefix_cache_salt`. Reuse supports greedy and sampling, plus text/chat finite choices. Benchmark IDs stay greedy and positive MTP requires draft weights.

OpenAI `prompt_tokens` counts the whole prompt. `prompt_tokens_details.cached_tokens` counts coherently restored target IDs, or zero on a miss or when disabled. GQA-only coverage has separate metrics. [Native usage](native-generate.md#json-and-sse) uses a different field. Zero output runs prefill only for prompt scores.

### Checkpoint boundaries

Let `a` be the restored GDN endpoint or zero, and `b` the exact prompt end. The resident planner selects `b` and every `b − 2^k > a` with integer `k ≥ 10`. Prefill visits them in ascending order. If a backup or an earlier capture can free a slot, prefill stops before the blocked endpoint and retries. Otherwise the planner declines the unavailable checkpoint and prefill can continue. [Planner](../src/runtime/execution/decoder_batch_prefix.cpp#L180)<!-- prefix: "void Decoder" --> · [Reservations](../src/runtime/execution/decoder_batch_prefix.cpp#L231)<!-- prefix: "bool Decoder" -->

Each checkpoint records GDN state and the final hidden and residual rows at its endpoint. GQA and shifted draft coverage are tracked separately. A request that prefills for prompt scores can also publish its state. [Snapshot capture](../src/runtime/execution/decoder_prefill_tiles.cpp#L519)<!-- prefix: "void capture" --> · [Resident publication](../src/runtime/execution/decoder_batch_prefix.cpp#L597)<!-- prefix: "void Decoder" -->

Successful completion can publish a terminal checkpoint at committed target position `U` when it advances beyond `b`. With one output, the prompt already supplies `U = b`, so there is no duplicate. Stop and EOS can change how many target rows are consumed. Use the committed cursor to determine `U`. Intermediate decode rounds publish no checkpoints. [Terminal publication](../src/runtime/execution/decoder_batch_prefix.cpp#L650)<!-- prefix: "void Decoder" -->

A later request that preserves the exact generated prefix can resume at `U` when its target and draft coverage suffice. If the target never consumed the final emitted ID, the follow-up prefills it with its suffix. The physical generated span is `O = U − b` processed rows. Count it once in the logical endpoint. Account for transferred GQA, shifted draft, and materialized GDN bytes separately.

### Checkpoint compatibility and resume

GQA and draft reuse contiguous MXFP8 spans of up to 256 rows. A GDN checkpoint holds convolution history, recurrent state, and final BF16 hidden and residual rows. TP2 shards the history and recurrent state and stores both score rows on each rank. Keys bind the prefix ending at each record. [GPU page lookup](../src/runtime/cache/prefix_residency.cpp#L347)<!-- prefix: "void lookup_" --> · [GDN state sizes](../src/runtime/cache/prefix_cache_layout.cpp#L19)<!-- prefix: "std::array\u003cs" -->

GPU and CPU lookups track contiguous GQA coverage and exact GDN endpoints independently. CPU transfers can extend resident GQA and draft coverage or supply a later usable GDN checkpoint. Resume at the largest available GDN endpoint within GQA coverage and the request's restore limit. With MTP, endpoint `P` also requires `P−1` contiguous shifted draft rows. TP2 requires compatible GQA, GDN, and rank-local draft coverage across both ranks. [Resident lookup](../src/runtime/cache/prefix_residency.cpp#L553)<!-- prefix: "ResidentPref" --> · [Rank intersection and key cutover](serving/lmcache.md#ranked-tp2-transport)

CPU and GPU each report GQA and draft coverage contiguous from row zero. Earlier rows in HBM cannot bridge a gap in the CPU inventory to recover a CPU-only suffix. The decoder stages at most the highest coherent CPU GDN endpoint when it improves on the resident endpoint and a checkpoint slot is immediately available. It can still stage GQA and draft coverage when no GDN slot is available. It retains the best coherent resident endpoint for fallback. [Restore staging](../src/runtime/execution/decoder_batch_prefix.cpp#L427)<!-- prefix: "bool Decoder" -->

If a ranked GET misses and includes GQA with GDN or draft state, the server retries all ranks with GQA alone. It does not try a second CPU GDN endpoint. The decoder resumes from coherent resident GDN or computes GDN from row zero. It reads GQA through its independent coverage and appends only uncovered rows. A transfer commits after all required ranks complete. [Ranked retry](../apps/server/batch_cache.cpp#L415)<!-- prefix: "bool read_ra" --> · [Resident restore commit](../src/runtime/execution/decoder_batch_prefix.cpp#L517)<!-- prefix: "void Decoder" -->

Only a committed restore from resident GDN refreshes that checkpoint's GPU recency. Lookup and duplicate checkpoint reuse can find or hold a snapshot without touching recency. Matched GDN counters can include an endpoint deeper than the committed resume position. [Lookup accounting](../src/runtime/cache/prefix_residency.cpp#L609)<!-- prefix: "void PrefixR" --> · [Duplicate lookup](../src/runtime/cache/prefix_residency.cpp#L600)<!-- prefix: "PrefixSnapsh" --> · [Daemon recency](serving/lmcache.md#native-transport)

`sglang_cpp:lmcache_gqa_computed_tokens_total` counts regenerated target GQA cache rows. Restored rows skip key normalization, rotary and MXFP8 cache append. A tile fully covered by GQA skips K/V projection; a tile crossing the boundary can still run packed QKV. [Tile projection](../src/runtime/execution/decoder_prefill_tiles.cpp#L250)<!-- prefix: "void Chunk::" --> · [Tile append](../src/runtime/execution/decoder_prefill_tiles.cpp#L265)<!-- prefix: "void Chunk::" --> · [Accounting](../src/decoder_batch.cpp#L844)<!-- prefix: "prefix_stats" -->

A paired boundary at `P` seeds shifted draft row `P-1` with the first suffix ID on a partial hit or the first committed output on a full hit. Full hits score from restored hidden/residual rows using the request's [grammar](text-serving.md#finite-choice-constraints) and [sampler](text-serving.md#sampling). Restore and full-hit scoring do not charge [target rows](batch-graphs.md#runtime-path) or change other lanes' decode policy.

Prompt scoring restores no farther than `logprob_start_len` and leaves at least the last prompt row to compute. Offset zero, including OpenAI echo scoring, restores no rows. Generated-only or [pointwise label scores](serving/pointwise-score.md) can use full hits. [Lookup limit](../apps/server/detail/batch_service_cache.cpp#L31)<!-- prefix: "const int ma" --> · [Full hit](../src/decoder_batch.cpp#L399)<!-- prefix: "if (lane.res" -->

### Media identity and prefill

Keys bind covered media patch identity, layout, feature row, and rotary positions. A change inside the covered prefix changes its identity. External C++ features can run without a cache identity. Key construction stops before the first such media row. [Media key construction](../src/runtime/cache/prefix_payload.cpp#L130)<!-- prefix: "const auto e" --> · [Media execution](multimodal-decoder.md#runtime-path)

Cold and restored suffix rows use the same [prefill planner](native-prefill-workspace.md#runtime-path). The former exact-only cache route and its fixed-shape Tensor Core probe remain archived. [Historical cache guide](https://github.com/para1lel/sglang.cpp/blob/28148cda233e5522274e258d5eb5caa9ab83e313/docs/prefix-cache.md#tensor-core-scheduling)

## Constraints and failure behavior

The GPU pools hold immutable prefix pages and checkpoints. Active requests and armed transfers retain their page and checkpoint leases. [Page reservation](../src/runtime/cache/prefix_residency.cpp#L623)<!-- prefix: "PrefixPages " --> · [Checkpoint reservation](../src/runtime/cache/prefix_residency.cpp#L655)<!-- prefix: "PrefixSnapsh" -->

Reclaim first evicts the oldest eligible backed record in the requested pool. If a backup can make room, reclaim requests pressure work and leaves the record resident while the transfer is pending. A GDN reservation can defer behind an active target or draft transfer. Explicit GDN pressure gives the checkpoint pool first choice of the next backup. [Reclaim order](../src/runtime/cache/prefix_residency.cpp#L519)<!-- prefix: "Reclaim Pref" --> · [Backup priority](../src/runtime/cache/prefix_residency.cpp#L435)<!-- prefix: "std::pair\u003cst" -->

The worker marks a record backed after all required ranks complete STORE and release their leases. After a failed backup, the record stays unbacked. When reclaim cannot arrange a usable backup, it can evict an unpinned unbacked record after a failed attempt or when no backup client exists. Pinned records remain in place. The backed flag records past acknowledgement. A later restore still requires current CPU coverage. [Worker result](../src/runtime/cache/prefix_residency.cpp#L470)<!-- prefix: "void PrefixR" --> · [Victim checks](../src/runtime/cache/prefix_residency.cpp#L502)<!-- prefix: "bool PrefixR" -->

When a pending backup blocks all selected prefill and decode is idle, the scheduler waits briefly for checkpoint progress. If no backup or earlier capture can free an eligible slot, it may decline a planned checkpoint. CPU GDN restore and optional terminal capture use an immediately available slot or proceed without one. [Scheduler wait](../src/decoder_batch.cpp#L360)<!-- prefix: "if (max_rows" --> · [Terminal reservation](../src/runtime/execution/decoder_batch_prefix.cpp#L664)<!-- prefix: "auto snapsho" -->

Abort can abandon LOOKUP. An armed GET or STORE retains its resources until matching lease release or original daemon exit. A timeout or replacement daemon leaves that lease live. Failed restores clear staged state before cold prefill. Terminal CUDA faults disable the batch. [Lease status](../tools/lmcache/sglang_lmcache.py#L668)<!-- prefix: "def lease_st" --> · [Restore handling](../src/runtime/execution/decoder_batch_prefix.cpp#L86)<!-- prefix: "DecoderBatch" -->

Direct callers that supply explicit restore and store callbacks follow the transfer contract. They own callback capacity and wait for terminal submission when a store is required. [Direct MTP](../src/decoder_batch.cpp#L909)<!-- prefix: "void Decoder" -->

### Native HTTP controls

The metrics, reset, and profile routes are available when LMCache is configured, even when the daemon is unreachable. The [HTTP admission and shutdown rules](http-server.md#http-admission-and-shutdown) also apply with reuse disabled.

| Route | Input | Result |
| --- | --- | --- |
| Health routes | Empty GET | Use the [liveness, readiness, and generation progress responses](http-server.md#health-and-readiness). |
| `POST /reset_prefix_cache` | `{}` | Returns `{"success":true}` after clearing idle GPU prefix records and the configured daemon store. Returns HTTP 409 while busy and HTTP 503 on daemon reset failure. |
| `GET /metrics` | Empty body | Returns Prometheus text with native provenance and `engine="0"`. |
| `POST /start_profile` | Exactly three fields. `profile_prefix` is a string of 1–64 bytes. `max_iterations` is an integer from 1 to 64. `profiler_kwargs` equals `{"with_stack":false}`. | Opens one capture and returns `{"success":true,"profiler":"native-batch-steps"}`. Returns HTTP 409 if a capture is open. |
| `POST /stop_profile` | `{}` | Closes the capture and adds `profile_prefix` and `steps` to the JSON response. Returns HTTP 409 if no capture is open. |

Reset and [flush](#cache-flush) clear resident GPU records and the configured daemon store across both TP2 ranks. The scheduler drains native transfer tasks first. The pool pauses new backups and waits for the active ranked store before clearing records. The scheduler then waits for the daemon reset acknowledgement. Pool capacity and cumulative counters persist. Successful controls return 200 and malformed POST controls return 400. Queued generations wait through the reset response write. [Scheduler reset](../apps/server/detail/batch_service_cache.cpp#L64)<!-- prefix: "std::optiona" --> · [Pool reset](../src/runtime/cache/prefix_residency.cpp#L773)<!-- prefix: "void PrefixR" -->

`vllm:prefix_cache_hits_total` counts coherently restored prompt IDs. `vllm:generation_tokens_total` counts published IDs. `sglang_cpp:prefix_` tracks pool capacity, occupancy, independent resident reuse, backups, and eviction. `sglang_cpp:lmcache_` reports daemon I/O and retains aggregate and per-rank transfer counters. Joint acknowledgements count once and per-rank receipts count successful rank replies. `lmcache_pending_stores` stays positive while an eligible backup is queued or its final pool accounting is pending. Pool and transport snapshots can cover different instants. Prefill counters distinguish planned, resident, published, and declined checkpoints. [Pool counters](../apps/server/loads.cpp#L263)<!-- prefix: "void emit_re" --> · [Pending stores](../apps/server/batch_service.cpp#L743)<!-- prefix: "BatchMetrics" --> · [Prefill counters](../apps/server/loads.cpp#L312)<!-- prefix: "emit(\"sglang" -->

A profile samples completed nonempty `advance_chunked` iterations started after capture opened. Each sample has host `elapsed_ns`, `width`, `prefill_rows`, `decode_lanes`, `graph_slots` for ordinary decode replay, and `verification_graph_slots` for MTP target verification replay. Both masks include only logical lanes that replayed and are present even when zero. Both are zero with `--eager`. At the sample cap the capture stays open. [Sample fields](../apps/server/detail/batch_trace.cpp#L125)<!-- prefix: "return {elap" --> · [JSON field](../apps/server/prefix_controls.cpp#L141)<!-- prefix: "R\"(,\"verific" -->

### Cache flush

`GET /flush_cache` and `POST /flush_cache` work with prefix reuse on or off. Both ignore a successfully framed body. Other methods return HTTP 405 with `Allow: GET, POST`. HEAD omits the body. The optional `timeout` query parameter is a nonnegative number in seconds. The reader form-decodes query fields (`+` means space) and uses the last `timeout`. The default is zero. Positive infinity allows an indefinite wait for idle. [Native input](../apps/server/flush_cache.cpp#L90)<!-- prefix: "FlushCacheRe" --> · [Pinned handler](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1060)<!-- prefix: "@app.api_rou" -->

| Condition | HTTP response |
| --- | --- |
| Idle, including with reuse off | 200, `Cache flushed.\nPlease check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)\n` |
| Busy at zero timeout | 400, `Flush cache failed.\n` |
| Still busy when a positive timeout expires | 400, `Timed out waiting for idle state.` |
| Flush queue full | 503 JSON with message `Native cache flush queue is full`. |
| Empty, malformed, negative, or NaN timeout | 400 JSON with `object="error"`, `type="Bad Request"`, `param=null`, `code=400`, and a `message` identifying `query.timeout`. |

The first three bodies are plain text without a `Content-Type` header. Each `\n` represents one line feed. The reader validates the timeout before queueing. [Replies](../apps/server/flush_cache.cpp#L107)<!-- prefix: "void write_f" --> · [Validation error](../apps/server/flush_cache.cpp#L75)<!-- prefix: "void write_i" -->

A dedicated worker claims flushes FIFO and starts the timeout at claim. The queue holds up to `--max-active` waiting flushes. The listener waits for no queued or claimed model work, no reader-side `/generate` preparation, and no reset. The scheduler also checks active jobs and generations and drains LMCache transfers before resetting. Tokenization, token counts, and standalone parsing use their own CPU queue and can continue independently. While busy, the flush worker checks for disconnect at least every 250 ms. With reuse on, daemon reset failure returns 503. Tokenizer, compiled tool grammars, and cumulative counters persist. [Enqueue](../apps/server/prefix_controls.cpp#L354)<!-- prefix: "void Connect" --> · [Flush worker](../apps/server/prefix_controls.cpp#L390)<!-- prefix: "void Connect" --> · [Wait and claim](../apps/server/prefix_controls.cpp#L360)<!-- prefix: "FlushCacheRe" --> · [Queue guard](../apps/server/prefix_controls.cpp#L286)<!-- prefix: "bool Connect" --> · [Scheduler reset](../apps/server/batch_service.cpp#L778)<!-- prefix: "bool BatchSe" -->.

## Verification

Run the pool route fixture against a Release server and the pinned checkpoint. Set `CUDA_VISIBLE_DEVICES` to the reserved B200 UUIDs in rank order, `TP` to the visible GPU count, and `DEVICE_BYTES` to a Bash array of each board's total bytes. Use the [LMCache 0.5.8 interpreter](serving/lmcache.md#start-the-daemon). The fixture starts a private daemon.

```bash
export CUDA_VISIBLE_DEVICES LMCACHE_PYTHON
PYTHONPATH="$PWD" python3 -B tests/prefix/pool_routes.py \
    "$SGLANG_SERVER" "$CHECKPOINT" "$EVIDENCE" \
    --tp "$TP" --device-bytes "${DEVICE_BYTES[@]}"
PYTHONPATH="$PWD" python3 -B tests/prefix/pool_routes.py \
    "$SGLANG_SERVER" "$CHECKPOINT" "${EVIDENCE}-anchors" \
    --tp "$TP" --device-bytes "${DEVICE_BYTES[@]}" --anchors-only
```

Keep `run.json`, `relay.json`, and the metric scrapes. The pressure fixture checks idle reuse, acknowledged backups under pressure, backed eviction, reload, cancellation, reset, and clean shutdown. The anchor fixture checks exact offsets and deeper GQA coverage than the GDN resume boundary. [Pressure fixture](../tests/prefix/pool_routes.py#L344)<!-- prefix: "def pressure" --> · [Anchor fixture](../tests/prefix/pool_routes.py#L459)<!-- prefix: "def anchors(" --> · [GPU pool checks](performance.md#gpu-pool-engineering-checks)

Check held IPC leases, worker restart, and terminal STORE reply order with the [device state and graph controls](../tests/prefix/README.md#device-state-and-graph-controls).

The synthetic allocator controls exercised TP1 and TP2 with draft enabled and disabled. Automatic GDN sizing grew from 1 to 10 slots at TP1 and from 2 to 20 at TP2. Target and draft page counts stayed fixed. Eight ranked pressure cases covered two or three GDN slots, a held target or draft transfer, and successful or failed GDN acknowledgement. Both waiting callers deferred. The GDN candidate stayed resident until acknowledgement or failure. See `artifacts/generalization/residency-proof/cross-family-native.sha256`.

On physical TP2 with MTP4, the capacity control grew from 107 to 1,163 GDN slots with 4,096 target and 4,096 draft pages unchanged. After 108 distinct prefixes, the oldest reuse hit 33 resident GDN rows with automatic sizing. The smaller pool read 153,985,024 GDN bytes from CPU for the same output ID. See `artifacts/generalization/residency-proof/physical-scope.json`.

The final physical TP2 cache fixture held a ranked STORE reply with one pending store. The control recorded three unbacked evictions. The revised pressure policy recorded 62 completed backups, 44 backed evictions, and zero failed backups or unbacked evictions. After drain it recorded 69 completed backups, 53 backed evictions, and zero unbacked evictions or pending stores. CPU reload reproduced token ID `17` and read 8,650,752 target GQA bytes, 153,985,024 GDN bytes, and 538,560 draft bytes across ranks. Restart checks covered complete restore and separate missing-rank target and draft records. See `artifacts/generalization/residency-proof/cross-family-physical.sha256`.

### Live control regression

Check malformed controls, busy/idle reset, bounded profiles, and peer graph eligibility. After cold reset require zero cached IDs. On replay after acknowledge and drain, require the expected restored length and identical first scores, IDs, finish and usage. [In-process fence installation](../tests/prefix/lmcache_relay.py#L496)<!-- prefix: "def install_" --> · [Terminal send trace](../tests/prefix/lmcache_send_trace.py#L31)<!-- prefix: "def terminal" -->

### Flush verification

At idle, try both flush methods with reuse on and off. Check the [status, body, and headers](#cache-flush), including `?timeout=invalid`. With reuse on, prime a cacheable prompt and repeat it. Verify that a successful flush preserves cumulative counters and that the next two requests miss, then hit.

Set `--max-active` to at least two. Hold a stream active and send three overlapping HTTP flushes in order with short, short, and zero timeouts. Keep the stream active through both deadlines. Expect two timeout replies followed by the busy reply. Verify that a failed flush preserves a primed hit. Repeat with two longer waits and finish the stream. Expect both flushes to succeed in order. Check `/health` while the flush worker waits.

### Prefix continuation

Follow the [LMCache route checks](../tests/prefix/README.md#device-state-and-graph-controls) for restored and independently computed row counts, cache misses, and first-score and continuation parity.
