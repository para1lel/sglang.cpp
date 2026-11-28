# Native prefix cache

Native keeps GQA pages and exact GDN checkpoints in separate GPU pools. A background worker backs resident data up to a local LMCache daemon under pressure. [Resident pools](../src/runtime/cache/prefix_residency.cpp#L478)<!-- prefix: "PrefixReside" --> · [Pressure worker](../src/runtime/cache/prefix_residency.cpp#L410)<!-- prefix: "void PrefixR" -->

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Configure prefix reuse, validate salts, and report usage. | [Salt validation](../apps/server/text_request.cpp#L619)<!-- prefix: "void set_cac" --> · [Usage writer](../apps/server/text_completion.cpp#L50)<!-- prefix: "void append_" --> |
| HTTP controls | Dispatch cache reset, metrics, and bounded scheduler profiling. | [Control routes](../apps/server/prefix_controls.cpp#L483)<!-- prefix: "void Connect" --> |
| Native cache | Builds prefix identities, leases GPU pages and checkpoints, and schedules CPU backup. | [Prefix keys](../src/runtime/cache/prefix_payload.cpp#L108)<!-- prefix: "PrefixCacheK" --> · [Pool capacity](../src/runtime/cache/prefix_residency.cpp#L142)<!-- prefix: "PoolLimits p" --> |
| LMCache | Resolves coverage, executes transfers, and retains CPU L1 records across native workers. | [Lookup](../tools/lmcache/sglang_lmcache.py#L165)<!-- prefix: "def coverage" --> · [Storage](../tools/lmcache/sglang_lmcache.py#L469)<!-- prefix: "class Record" --> |

## LMCache daemon

For CPU backup, start one [LMCache 0.5.8 daemon](serving/lmcache.md#start-the-daemon) as the native server’s Linux user. Expose the same physical GPUs to both processes and use namespace `sglang`. TP2 shares the daemon across both ranks.

## Runtime path

Native enables GPU prefix reuse and configures LMCache by default. GPU reuse also works when the daemon is unavailable. Use an absolute `--lmcache-socket` to select the daemon. Use `--prefix-cache-entries 0` to disable both tiers. Positive values are invalid. `--prefix-gqa-pages` and `--prefix-gdn-slots` default to zero for automatic sizing. Set positive values to constrain a pressure test. `--eager` independently controls [batch and TP2 replay](batch-graphs.md#runtime-path). [Server options](../apps/sglang-server.cpp#L204)<!-- prefix: "} else if (o" --> · [Daemon lookup](../apps/server/batch_cache.cpp#L463)<!-- prefix: "void BatchCa" --> · [Startup pools](serving/request-memory.md)

Text, chat, benchmark IDs, and `/generate` accept `cache_salt` of up to 256 UTF-8 bytes, default empty. Invalid salts return 400 even with reuse off. See [native per-item and null handling](native-generate.md#constraints-and-failure-behavior). C++ uses `DecoderOptions::prefix_cache_salt`. Reuse supports greedy and sampling, plus text/chat finite choices. Benchmark IDs stay greedy and positive MTP requires draft weights.

OpenAI `prompt_tokens` counts the whole prompt. `prompt_tokens_details.cached_tokens` counts coherently restored target IDs, or zero on a miss or when disabled. GQA-only coverage has separate metrics. [Native usage](native-generate.md#json-and-sse) uses a different field. Zero output runs prefill only for prompt scores.

### Checkpoint boundaries

Let `a` be the restored GDN endpoint or zero, and `b` the exact prompt end. The resident planner selects `b` and every `b − 2^k > a` with integer `k ≥ 10`. Prefill visits them in ascending order. If a backup or an earlier capture can free a slot, prefill stops before the blocked endpoint and retries. Otherwise the planner declines the unavailable checkpoint. [Planner](../src/runtime/execution/decoder_batch_prefix.cpp#L151)<!-- prefix: "void Decoder" --> · [Reservations](../src/runtime/execution/decoder_batch_prefix.cpp#L217)<!-- prefix: "for (const a" -->

Each checkpoint records GDN state and the final hidden and residual rows at its endpoint. GQA and shifted draft coverage are tracked separately. A request that prefills for prompt scores can also publish its state. [Snapshot capture](../src/runtime/execution/decoder_prefill_tiles.cpp#L493)<!-- prefix: "void capture" --> · [Resident publication](../src/runtime/execution/decoder_batch_prefix.cpp#L547)<!-- prefix: "void Decoder" -->

Successful completion can publish a terminal checkpoint at committed target position `U` when it advances beyond `b`. With one output, the prompt already supplies `U = b`, so there is no duplicate. Stop and EOS can change how many target rows are consumed. Use the committed cursor to determine `U`. Intermediate decode rounds publish no checkpoints. [Terminal publication](../src/runtime/execution/decoder_batch_prefix.cpp#L587)<!-- prefix: "void Decoder" -->

A later request that preserves the exact generated prefix can resume at `U` when its target and draft coverage suffice. If the target never consumed the final emitted ID, the follow-up prefills it with its suffix. The physical generated span is `O = U − b` processed rows. Count it once in the logical endpoint. Account for transferred GQA, shifted draft, and materialized GDN bytes separately.

### Checkpoint compatibility and resume

GQA and draft reuse contiguous MXFP8 spans of up to 256 rows. A GDN checkpoint holds convolution history, recurrent state, and final BF16 hidden and residual rows. Keys bind the prefix ending at each record. [GPU page lookup](../src/runtime/cache/prefix_residency.cpp#L283)<!-- prefix: "void lookup_" --> · [LMCache records](../src/runtime/cache/lmcache_client.cpp#L276)<!-- prefix: "void append_" -->

GPU and CPU lookups track contiguous GQA coverage and exact GDN endpoints independently. CPU transfers can extend resident GQA and draft coverage or supply a later usable GDN checkpoint. Resume at the largest available GDN endpoint within GQA coverage and the request's restore limit. With MTP, rank 0 also needs contiguous shifted draft through `P−1` for an endpoint `P`. TP2 uses the coverage common to both target ranks. [Resident lookup](../src/runtime/cache/prefix_residency.cpp#L493)<!-- prefix: "ResidentPref" --> · [CPU extension](../src/runtime/execution/decoder_batch_prefix.cpp#L382)<!-- prefix: "bool Decoder" --> · [Rank intersection](../src/runtime/cache/lmcache_group.cpp#L130)<!-- prefix: "PrefixCacheR" -->

Resume GDN at the usable boundary, or compute it from row zero. Read GQA through its independent coverage and append GQA only for uncovered rows. A cache transfer commits after all required ranks complete. [Ranked restore](serving/lmcache.md#ranked-tp2-transport) · [Restore commit](../src/runtime/execution/decoder_batch_prefix.cpp#L99)<!-- prefix: "const int re" -->

`sglang_cpp:lmcache_gqa_computed_tokens_total` counts regenerated target GQA cache rows. Restored rows skip key normalization, rotary and MXFP8 cache append. A tile fully covered by GQA skips K/V projection; a tile crossing the boundary can still run packed QKV. [Tile projection](../src/runtime/execution/decoder_prefill_tiles.cpp#L270)<!-- prefix: "void Chunk::" --> · [Tile append](../src/runtime/execution/decoder_prefill_tiles.cpp#L285)<!-- prefix: "void Chunk::" --> · [Accounting](../src/decoder_batch.cpp#L881)<!-- prefix: "prefix_stats" -->

Cache pressure can evict a published record before the next lookup. The [AgentX pressure diagnostic](agentx-256k.md#focused-cache-pressure-diagnostic) illustrates this on the earlier direct-store implementation.

A paired boundary at `P` seeds shifted draft row `P-1` with the first suffix ID on a partial hit or the first committed output on a full hit. Full hits score from restored hidden/residual rows using the request's [grammar](text-serving.md#finite-choice-constraints) and [sampler](text-serving.md#sampling). Restore and full-hit scoring do not charge [target rows](batch-graphs.md#runtime-path) or change other lanes' decode policy.

Prompt scoring restores no farther than `logprob_start_len` and leaves at least the last prompt row to compute. Offset zero, including OpenAI echo scoring, restores no rows. Generated-only or [pointwise label scores](serving/pointwise-score.md) can use full hits. [Lookup limit](../apps/server/detail/batch_service_cache.cpp#L31)<!-- prefix: "const int ma" --> · [Full hit](../src/decoder_batch.cpp#L414)<!-- prefix: "if (lane.res" -->

[Archived TP2 replay receipts](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/prefix-cache.md#checkpoint-compatibility-and-resume) include eight cohorts, cache restores, cancellation and ranked payloads.

### Media identity and prefill

Keys bind covered media patch identity, layout, feature row, and rotary positions. A change inside the covered prefix changes its identity. External C++ features can run without a cache identity. Key construction stops before the first such media row. [Media key construction](../src/runtime/cache/prefix_payload.cpp#L130)<!-- prefix: "const auto e" --> · [Media execution](multimodal-decoder.md#runtime-path)

Cold and restored suffix rows use the same [prefill planner](native-prefill-workspace.md#runtime-path). The former exact-only cache route and its fixed-shape Tensor Core probe remain archived. [Historical cache guide](https://github.com/para1lel/sglang.cpp/blob/28148cda233e5522274e258d5eb5caa9ab83e313/docs/prefix-cache.md#tensor-core-scheduling)

## Constraints and failure behavior

The GPU pools hold immutable prefix pages and checkpoints. Active requests and armed transfers retain their page and checkpoint leases. [Page reservation](../src/runtime/cache/prefix_residency.cpp#L554)<!-- prefix: "PrefixPages " --> · [Checkpoint reservation](../src/runtime/cache/prefix_residency.cpp#L586)<!-- prefix: "PrefixSnapsh" -->

The pressure worker selects old unpinned records when a pool is low on space or an allocation requests relief. It marks a record backed after every required rank acknowledges its store and releases the transfer. Reclaim prefers backed records, then can evict an unbacked unpinned record. Backup failure leaves the record unbacked. [Pressure](../src/runtime/cache/prefix_residency.cpp#L365)<!-- prefix: "bool PrefixR" --> · [Eviction order](../src/runtime/cache/prefix_residency.cpp#L457)<!-- prefix: "bool PrefixR" -->

Abort can abandon LOOKUP. An armed GET or STORE retains its resources until matching lease release or original daemon exit. A timeout or replacement daemon leaves that lease live. Failed restores clear staged state before cold prefill. Terminal CUDA faults disable the batch. [Lease status](../tools/lmcache/sglang_lmcache.py#L668)<!-- prefix: "def lease_st" --> · [Restore handling](../src/runtime/execution/decoder_batch_prefix.cpp#L62)<!-- prefix: "DecoderBatch" -->

Direct callers that supply explicit restore and store callbacks follow the transfer contract. They own callback capacity and wait for terminal submission when a store is required. [Direct MTP](../src/decoder_batch.cpp#L949)<!-- prefix: "void Decoder" -->

### Native HTTP controls

The metrics, reset, and profile routes are available when LMCache is configured, even when the daemon is unreachable. The [HTTP admission and shutdown rules](http-server.md#http-admission-and-shutdown) also apply with reuse disabled.

| Route | Input | Result |
| --- | --- | --- |
| Health routes | Empty GET | Use the [liveness, readiness, and generation progress responses](http-server.md#health-and-readiness). |
| `POST /reset_prefix_cache` | `{}` | Returns `{"success":true}` after clearing idle GPU prefix records and the configured daemon store. Returns HTTP 409 while busy and HTTP 503 on daemon reset failure. |
| `GET /metrics` | Empty body | Returns Prometheus text with native provenance and `engine="0"`. |
| `POST /start_profile` | Exactly three fields. `profile_prefix` is a string of 1–64 bytes. `max_iterations` is an integer from 1 to 64. `profiler_kwargs` equals `{"with_stack":false}`. | Opens one capture and returns `{"success":true,"profiler":"native-batch-steps"}`. Returns HTTP 409 if a capture is open. |
| `POST /stop_profile` | `{}` | Closes the capture and adds `profile_prefix` and `steps` to the JSON response. Returns HTTP 409 if no capture is open. |

Reset and [flush](#cache-flush) clear resident GPU records and the configured daemon store across both TP2 ranks. The scheduler drains native transfer tasks first. The pool pauses new backups and waits for the active ranked store before clearing records. The scheduler then waits for the daemon reset acknowledgement. Pool capacity and cumulative counters persist. Successful controls return 200 and malformed POST controls return 400. Queued generations wait through the reset response write. [Scheduler reset](../apps/server/detail/batch_service_cache.cpp#L64)<!-- prefix: "std::optiona" --> · [Pool reset](../src/runtime/cache/prefix_residency.cpp#L701)<!-- prefix: "void PrefixR" -->

`vllm:prefix_cache_hits_total` counts coherently restored prompt IDs. `vllm:generation_tokens_total` counts published IDs. `sglang_cpp:prefix_` tracks pool capacity, occupancy, independent resident reuse, backups, and eviction. `sglang_cpp:lmcache_` reports daemon I/O and retains aggregate and per-rank transfer counters. Joint acknowledgements count once and per-rank receipts count successful rank replies. `lmcache_pending_stores` stays positive while an eligible backup is queued or its final pool accounting is pending. Pool and transport snapshots can cover different instants. Prefill counters distinguish planned, resident, published, and declined checkpoints. [Pool counters](../apps/server/loads.cpp#L263)<!-- prefix: "void emit_re" --> · [Pending stores](../apps/server/batch_service.cpp#L805)<!-- prefix: "BatchMetrics" --> · [Prefill counters](../apps/server/loads.cpp#L312)<!-- prefix: "emit(\"sglang" -->

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

The worker claims flushes FIFO and starts the timeout at claim. Busy includes queued or claimed completions, generation and tokenization, active generation, a health job, pending stores, retiring GETs, and reset. Other requests may progress during a deferred wait. With reuse on, daemon reset failure returns 503. Tokenizer, compiled tool grammars, and cumulative counters persist. [Scheduler guard](../apps/server/batch_service.cpp#L832)<!-- prefix: "bool BatchSe" --> · [Pinned upstream compatibility details](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/prefix-cache.md#cache-flush).

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

Keep `run.json`, `relay.json`, and the metric scrapes. The pressure fixture checks idle reuse, acknowledged backups under pressure, backed eviction, reload, cancellation, reset, and clean shutdown. The anchor fixture checks exact offsets and deeper GQA coverage than the GDN resume boundary. [Pressure fixture](../tests/prefix/pool_routes.py#L344)<!-- prefix: "def pressure" --> · [Anchor fixture](../tests/prefix/pool_routes.py#L459)<!-- prefix: "def anchors(" --> · [TP1 and TP2 evidence](performance.md#gpu-pool-engineering-checks)

The archived direct-store fixture covers held IPC leases, worker restart, and terminal STORE reply order. [Archived v4 procedure](https://github.com/para1lel/sglang.cpp/blob/3b138b699bdf47b02a7dde8ef5e07142ee699d3f/docs/prefix-cache.md#verification) · [Device state and graph controls](../tests/prefix/README.md#device-state-and-graph-controls)

### Live control regression

Check malformed controls, busy/idle reset, bounded profiles, and peer graph eligibility. After cold reset require zero cached IDs. On replay after acknowledge and drain, require the expected restored length and identical first scores, IDs, finish and usage. [Former resident-cache procedure](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/docs/prefix-cache.md#verification). [In-process fence installation](../tests/prefix/lmcache_relay.py#L496)<!-- prefix: "def install_" --> · [Terminal send trace](../tests/prefix/lmcache_send_trace.py#L31)<!-- prefix: "def terminal" -->

### Flush verification

At idle, try both flush methods with reuse on and off. Check the [status, body, and headers](#cache-flush), including `?timeout=invalid`. With reuse on, prime a cacheable prompt and repeat it. Verify that a successful flush preserves cumulative counters and that the next two requests miss, then hit.

Hold a stream active and send three overlapping HTTP flushes in order with short, short, and zero timeouts. Keep the stream active through both deadlines. Expect two timeout replies followed by the busy reply. Verify that a failed flush preserves a primed hit. Repeat with two longer waits and finish the stream. Expect both flushes to succeed in order. Check `/health` while the flush worker waits.

### W3-prefix

Use the current [K4 warm-prefix diagnostic](../benchmarks/paired/README.md#prefix-reuse-diagnostic) for both engines. The [historical K0 W3-prefix procedure](https://github.com/para1lel/sglang.cpp/blob/28148cda233e5522274e258d5eb5caa9ab83e313/docs/prefix-cache.md#w3-prefix) preserves the 25,600 restored and 7,168 computed row gates.
