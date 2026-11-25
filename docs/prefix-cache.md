# Native prefix cache

Native reuses GQA, GDN, and draft checkpoints through a local LMCache daemon. The daemon moves payloads between live GPU allocations and CPU L1.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Configure prefix reuse, validate salts, and report usage. | [Salt validation](../apps/server/text_request.cpp#L615)<!-- prefix: "void set_cac" --> · [Usage writer](../apps/server/text_completion.cpp#L50)<!-- prefix: "void append_" --> |
| HTTP controls | Dispatch cache reset, metrics, and bounded scheduler profiling. | [Control routes](../apps/server/prefix_controls.cpp#L483)<!-- prefix: "void Connect" --> |
| Native cache | Builds prefix identities and describes GPU allocation regions. | [Prefix keys](../src/runtime/cache/prefix_payload.cpp#L85)<!-- prefix: "PrefixCacheK" --> · [Client](../include/sglang/runtime/cache/lmcache_client.h#L53)<!-- prefix: "class LmCach" --> |
| LMCache | Resolves coverage, executes transfers, and retains CPU L1 records across native workers. | [Lookup](../tools/lmcache/sglang_lmcache.py#L142)<!-- prefix: "def resolve(" --> · [Storage](../tools/lmcache/sglang_lmcache.py#L430)<!-- prefix: "class Record" --> |

## LMCache daemon

Start a dedicated LMCache 0.5.8 daemon as the same Linux user as the native server. Expose the same physical GPU UUID to both processes. Use namespace `sglang`. The [daemon guide](serving/lmcache.md#start-the-daemon) gives the launch command, socket defaults, and capacity options.

## Runtime path

The server attempts daemon-backed reuse by default. Use `--lmcache-socket PATH` to select an absolute Unix socket and `--prefix-cache-entries 0` to opt out. Positive entry counts are invalid. `--eager` controls graph replay independently.

Native text, chat, benchmark ID, and `/generate` requests accept an optional `cache_salt` string of up to 256 UTF-8 bytes. Omission uses an empty string. See [native generation options](native-generate.md#constraints-and-failure-behavior) for null and per-item salts. Invalid salts return HTTP 400 even when reuse is disabled. Native text, chat, and `/generate` support greedy and sampled requests with prefix reuse. Text and chat also support finite-choice requests. The benchmark ID route keeps its greedy policy.

Positive MTP depths require loaded draft weights. C++ request options carry the salt as `DecoderOptions::prefix_cache_salt`. [Token-ID parser](../apps/server/completions.cpp#L28)<!-- prefix: "CompletionRe" -->

Generation usage counts the full prompt in `prompt_tokens`. `prompt_tokens_details.cached_tokens` counts the prompt rows restored as a complete target prefix. GQA reuse beyond that prefix has its own counters. A miss or disabled cache reports zero. A zero-output request runs prefill only when it needs prompt scores. [Token-ID usage](../apps/server/completions.cpp#L97)<!-- prefix: "std::string " -->

### Checkpoint boundaries

The server attempts a checkpoint every 8,192 processed target tokens and at successful request end. The model context limit is 262,144 rows. Prefill and MTP grants stop at periodic boundaries when checkpointing is active. If the request ends at a boundary, one checkpoint suffices. The key chain incorporates the salt and model row IDs, rotary positions, and covered media. [Key construction](../src/runtime/cache/prefix_payload.cpp#L85)<!-- prefix: "PrefixCacheK" -->

The request-end checkpoint holds the latest committed target state. A request can end before the target processes its last emitted token. A 128-token prompt that emits 128 tokens at the length limit leaves 255 processed IDs. A follow-up prompt containing the full history can restore those 255 IDs and prefill the last emitted ID with its new suffix. A session can continue across requests. Closing a connection alone creates no checkpoint.

### Checkpoint compatibility and resume

GQA and draft records hold contiguous MXFP8 chunks of up to 256 rows. A GDN checkpoint holds convolution history, recurrent state, and the final BF16 hidden and residual rows. Each record is keyed by its prefix end. [Chunk encoding](../src/runtime/cache/lmcache_client.cpp#L257)<!-- prefix: "void append_" --> · [Checkpoint encoding](../src/runtime/cache/lmcache_client.cpp#L278)<!-- prefix: "void append_" -->

The daemon tracks resident GDN endpoints from L1 write completion and deletion events. Lookup tests the request's exact prefix digests at those endpoints against L1. It reports the highest matching GDN boundary and the longest contiguous GQA match. The usable GDN boundary requires contiguous GQA through that row and respects the request's restore limit. At an MTP boundary `P`, it also requires contiguous draft coverage of `P-1` rows. GQA coverage can extend beyond the usable GDN boundary. [Resident endpoints](../tools/lmcache/sglang_lmcache.py#L47)<!-- prefix: "class GdnEnd" --> · [Coverage selection](../tools/lmcache/sglang_lmcache.py#L142)<!-- prefix: "def resolve(" -->

Resume at the usable GDN boundary, or row zero when there is none. Recompute intervening GDN rows and use restored GQA read-only through its coverage. Append GQA only for uncovered rows. Native commits these extents after a successful transfer. [Restore view](../src/decoder_prefix_cache.cpp#L237)<!-- prefix: "PrefixDevice" --> · [Commit](../src/decoder_prefix_cache.cpp#L279)<!-- prefix: "int commit_r" -->

The next shifted draft row at `P-1` uses the next token's embedding. A partial hit can seed it with the first suffix ID. Full-hit verification uses the first committed output ID.

Prompt token scores need the target row before each scored token. Restore stops at `logprob_start_len` and leaves at least the last prompt row to compute. An offset of zero, including OpenAI echo scoring, restores no rows. Generated-only scores and [pointwise label scores](serving/pointwise-score.md#runtime-path) can use a full hit. [Lookup limit](../apps/server/batch_service_cache.cpp#L12)<!-- prefix: "const int ma" -->

On a full hit, the target computes the next-token logits from the restored hidden and residual rows. The current [grammar](text-serving.md#finite-choice-constraints) and [seed and sampler](text-serving.md#sampling) control selection. Restore and full-hit scoring stay outside the target row cap. Other lanes retain their normal decode selection. At positive depth, any later verification seeds draft position `P-1` with the first committed output ID. [Full hit](../src/decoder_batch.cpp#L502)<!-- prefix: "if (lane.res" --> · [MTP row accounting](mtp.md#runtime-path)

### Media identity and prefill

Keys bind covered media patch identity, layout, feature row, and rotary positions. A change inside the covered prefix changes its identity. External C++ features can run without a cache identity. Key construction stops before the first such media row. [Media key construction](../src/runtime/cache/prefix_payload.cpp#L103)<!-- prefix: "const auto e" --> · [Media execution](multimodal-decoder.md#runtime-path)

Cold and restored suffix rows use the same [prefill planner](native-prefill-workspace.md#runtime-path). The former exact-only cache route and its fixed-shape Tensor Core probe remain archived. [Historical cache guide](https://github.com/para1lel/sglang.cpp/blob/28148cda233e5522274e258d5eb5caa9ab83e313/docs/prefix-cache.md#tensor-core-scheduling)

## Constraints and failure behavior

Each request keeps its page-rounded GPU allocation. Native exports allocation views and a readiness event. The daemon imports the views, waits for readiness, and moves the payload. CPU L1 retains records across native workers. [Request reservation](batch-graphs.md#resident-memory-admission) · [GPU transport](serving/lmcache.md#native-transport)

A periodic store view retains the GQA and draft allocations and copies GDN state, hidden rows, and residual rows into a private device allocation. The producer stream records readiness after these copies. The four source blocks stay live until the copy completes, including after cancellation. If cleanup cannot confirm completion, it quarantines the copy storage and its sources. A terminal view retains the original allocations for all three families without copying them. Retained blocks stay out of the recycler even if a request grows or retires. [State view](../src/decoder_prefix_cache.cpp#L168)<!-- prefix: "PrefixDevice" --> · [Producer order](../src/decoder_prefix_cache.cpp#L60)<!-- prefix: "void submit(" --> · [Producer fence](../src/decoder_prefix_cache.cpp#L95)<!-- prefix: "cudaError_t " --> · [Export](../src/decoder_prefix_cache.cpp#L258)<!-- prefix: "PrefixDevice" --> · [Allocation retention](../src/runtime/backend/cuda_resources.cpp#L447)<!-- prefix: "void Request" -->

Native uses separate pools for LOOKUP, GET, and STORE, each with up to four workers. The store pool accepts up to twice the configured lane count in total, including queued and active stores. When capacity is full, the producing lane waits at its checkpoint and other eligible lanes advance. A successful terminal step claims completion before it waits for capacity. Cancellation can discard an unsubmitted periodic checkpoint. If abort wins the terminal claim, the service skips that checkpoint even at an 8,192-token boundary. [Native pools](../apps/server/batch_cache.cpp#L118)<!-- prefix: "State(std::u" --> · [Store capacity](../apps/server/batch_cache.cpp#L204)<!-- prefix: "bool BatchCa" --> · [Terminal claim](../src/decoder_batch.cpp#L784)<!-- prefix: "if (terminal" --> · [Deferred cancellation](../include/sglang/runtime/server/batch_service_state.h#L48)<!-- prefix: "bool discard" -->

Once STORE is submitted, the producing lane can continue or retire and its stream can finish. The store worker owns the key metadata, immutable allocation view, and capacity reservation through the daemon response or lease release. The daemon serializes writers and acknowledges after storing the records. A later request can miss a checkpoint still being written. Abort or disconnect can abandon a lookup. A pending GET keeps its destination and slot until the transfer finishes. If an armed transfer loses its socket, the client waits for lease release or daemon exit before releasing its views. [Store completion](../apps/server/batch_cache.cpp#L222)<!-- prefix: "void BatchCa" --> · [Daemon store](../tools/lmcache/sglang_lmcache.py#L470)<!-- prefix: "def store(se" --> · [GET retirement](../src/decoder_batch.cpp#L399)<!-- prefix: "if (!prefix." --> · [Lease release](../src/runtime/cache/lmcache_client.cpp#L330)<!-- prefix: "void wait_fo" -->

A failed restore restarts that request cold. A daemon refusal increments the declined-store count. A transport failure increments the I/O error count. An asynchronous write failure leaves the checkpoint unacknowledged and preserves any result already delivered to the client. Previously submitted checkpoints can finish after the request aborts. CUDA errors follow the [batch error rules](batch-graphs.md#constraints-and-failure-behavior).

### Native HTTP controls

The metrics, reset, and profile routes are available when LMCache is configured, even when the daemon is unreachable. The [HTTP admission and shutdown rules](http-server.md#http-admission-and-shutdown) also apply with reuse disabled.

| Route | Input | Result |
| --- | --- | --- |
| Health routes | Empty GET | Use the [liveness, readiness, and generation progress responses](http-server.md#health-and-readiness). |
| `POST /reset_prefix_cache` | `{}` | Returns `{"success":true}` after clearing the idle daemon store. Returns HTTP 409 while busy and HTTP 503 on daemon reset failure. |
| `GET /metrics` | Empty body | Returns Prometheus text with native provenance and `engine="0"`. |
| `POST /start_profile` | Exactly three fields. `profile_prefix` is a string of 1–64 bytes. `max_iterations` is an integer from 1 to 64. `profiler_kwargs` equals `{"with_stack":false}`. | Opens one capture and returns `{"success":true,"profiler":"native-batch-steps"}`. Returns HTTP 409 if a capture is open. |
| `POST /stop_profile` | `{}` | Closes the capture and adds `profile_prefix` and `steps` to the JSON response. Returns HTTP 409 if no capture is open. |

Successful controls return HTTP 200. Malformed POST control bodies return HTTP 400. Pending stores keep reset busy after the last stream closes. Reset succeeds after the ordered daemon acknowledgement. Cumulative counters persist. New completions can queue during reset. The worker waits for the reset response write before claiming them. Health and readiness remain available. [Reset admission](../apps/server/prefix_controls.cpp#L374)<!-- prefix: "void Connect" --> · [Reset response](../apps/server/prefix_controls.cpp#L387)<!-- prefix: "void Connect" --> · [Daemon reset](../apps/server/batch_cache.cpp#L248)<!-- prefix: "void BatchCa" --> · [Profile input](../apps/server/prefix_controls.cpp#L211)<!-- prefix: "void start_p" -->

The `vllm:` names support the frozen client. `num_requests_running` counts claimed completions until socket closure. `num_requests_waiting` excludes disconnected clients. `prefix_cache_hits_total` counts coherent restored prompt IDs. The first published user output records prompt length and computed prefill rows. `generation_tokens_total` counts published output IDs. The `sglang_cpp:lmcache_` series tracks availability, misses, I/O errors, acknowledged and declined writes, per-family rows and bytes, and elapsed nanoseconds. `sglang_cpp:lmcache_pending_stores` includes checkpoints waiting for transfer capacity and submitted stores until their allocation views are released. `sglang_cpp:prefix_cache_stores_total` also counts acknowledged checkpoints. [Series](../apps/server/loads.cpp#L203)<!-- prefix: "std::string " --> · [Probe accounting](http-server.md#generation-probe-lifecycle)

`DecoderBatch::prefix_cache_stats().stores` counts acknowledged checkpoints, including receipts arriving after the producing lane retires. A C++ `store_prefix` callback receives a batch-scoped counter as its fifth argument. Increment it once on successful STORE, before releasing the capacity reservation. Pending, declined, and failed stores leave the counter unchanged. With cache tracing enabled, a receipt wakes the idle worker so the trace can report the new count. Both HTTP store counters read transport receipts. [Batch stats](../src/decoder_batch.cpp#L431)<!-- prefix: "DecoderPrefi" --> · [Receipt](../apps/server/batch_cache.cpp#L234)<!-- prefix: "if (state-\u003er" --> · [Trace wake](../apps/server/batch_service.cpp#L132)<!-- prefix: "(trace && lm" --> · [HTTP snapshot](../apps/server/batch_service.cpp#L745)<!-- prefix: "metrics.cach" -->

A native capture records completed `advance_chunked` calls from nonempty iterations that start after profile start. Each sample contains `elapsed_ns` in host time, `width`, total `prefill_rows`, `decode_lanes`, and the `graph_slots` bit mask. The capture stays open after reaching its sample limit. [Step timing](../apps/server/batch_service.cpp#L357)<!-- prefix: "steps = batc" --> · [Sample recording](../apps/server/batch_service.cpp#L380)<!-- prefix: "profile->ste" --> · [Sample fields](../include/sglang/server/batch_trace.h#L127)<!-- prefix: "inline Batch" --> · [Response](../apps/server/prefix_controls.cpp#L230)<!-- prefix: "void stop_pr" -->

### Cache flush

`GET /flush_cache` and `POST /flush_cache` work with prefix reuse on or off. Both ignore a successfully framed body. Other methods return HTTP 405 with `Allow: GET, POST`. HEAD omits the body. The optional `timeout` query parameter is a nonnegative number in seconds. The reader form-decodes query fields (`+` means space) and uses the last `timeout`. The default is zero. Positive infinity allows an indefinite wait for idle. [Native input](../apps/server/flush_cache.cpp#L90)<!-- prefix: "FlushCacheRe" --> · [Pinned handler](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1060)<!-- prefix: "@app.api_rou" -->

| Condition | HTTP response |
| --- | --- |
| Idle, including with reuse off | 200, `Cache flushed.\nPlease check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)\n` |
| Busy at zero timeout | 400, `Flush cache failed.\n` |
| Still busy when a positive timeout expires | 400, `Timed out waiting for idle state.` |
| Empty, malformed, negative, or NaN timeout | 400 JSON with `object="error"`, `type="Bad Request"`, `param=null`, `code=400`, and a `message` identifying `query.timeout`. |

The first three bodies are plain text without a `Content-Type` header. Each `\n` represents one line feed. The reader validates the timeout before queueing. [Replies](../apps/server/flush_cache.cpp#L107)<!-- prefix: "void write_f" --> · [Validation error](../apps/server/flush_cache.cpp#L75)<!-- prefix: "void write_i" -->

A separate worker claims HTTP flushes in FIFO order. Each timeout starts when the worker handles that request. Queued or claimed completion, generation, and tokenization requests count as busy. The scheduler also checks for active generation, an installed health job, pending stores, retiring GETs, or another reset. During a deferred wait, other requests can enter and finish. [FIFO](../apps/server/prefix_controls.cpp#L437)<!-- prefix: "void Connect" --> · [Wait loop](../apps/server/prefix_controls.cpp#L407)<!-- prefix: "FlushCacheRe" --> · [Scheduler guard](../apps/server/batch_service.cpp#L757)<!-- prefix: "bool BatchSe" -->

The pinned single-tokenizer HTTP communicator holds its FIFO lock through the scheduler reply. A second direct scheduler call that bypasses the communicator can return `Another flush_cache is already in progress.` while a deferred request is pending. [Communicator](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/communicator.py#L41)<!-- prefix: "async def qu" --> · [Scheduler wrapper](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler_components/flush_wrapper.py#L22)<!-- prefix: "def handle(s" -->

With reuse enabled, native flush clears the LMCache store and returns HTTP 503 if that reset fails. The native tokenizer, compiled tool grammars, and cumulative counters persist. The pinned SGLang scheduler also resets memory pools, grammar state, and metrics. Its tokenizer clears the preprocessing cache when a multimodal processor is present. [Tokenizer](../apps/server/text_completion.cpp#L20)<!-- prefix: "const Tokeni" --> · [Tool grammars](../src/runtime/model/qwen_tool_constraints.cpp#L71)<!-- prefix: "std::shared_" --> · [Pinned scheduler](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler.py#L6285)<!-- prefix: "self.tree_ca" --> · [Pinned tokenizer](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_control_mixin.py#L330)<!-- prefix: "async def fl" -->

## Verification

Run the [LMCache route fixture](../tests/prefix/lmcache_routes.py#L752)<!-- prefix: "def main():" --> with the Release server, pinned checkpoint, and private LMCache 0.5.8 daemon. Use the [frozen concurrent fixtures](../benchmarks/concurrent/README.md#regenerate-and-measure). Set `CUDA_VISIBLE_DEVICES` to the same reserved B200 UUID in both processes. Use namespace `sglang` and a fresh evidence directory.

```bash
PYTHONPATH="$PWD" python3 -B tests/prefix/lmcache_routes.py \
    "$SGLANG_SERVER" "$CHECKPOINT" "$FIXTURES" "$LMCACHE_SOCKET" "$EVIDENCE"
```

Require exit zero, `summary.json`, and `relay.json`. Check selective reuse, divergent prefixes, salt isolation, shifted draft reuse, held lookups and writes, cancellation, a dropped reply, and reuse across worker restarts. For armed-transfer verification, call `install_device_fence` in the private test daemon before serving. Pass the same directory to the fixture with `--device-fence-dir`. Fill both store reservations on a one-lane server, then abort a request after it claims a length finish or text stop at exactly 8,192 rows. Require the claimed finish, one GDN checkpoint at that endpoint, and an 8,192-row hit on replay. For a held STORE acknowledgement, require writer completion and peer progress during the hold. For an armed GET, require peer output during the hold and reuse of its destination slot only after lease release. [Fence helper](../tests/prefix/lmcache_relay.py#L265)<!-- prefix: "def install_" --> · [Armed abort fixture](../tests/prefix/lmcache_routes.py#L592)<!-- prefix: "def abort_ar" --> · [Boundary fixture](../tests/prefix/lmcache_routes.py#L702)<!-- prefix: "def exact_te" -->

Use the [device state and graph controls](../tests/prefix/README.md#device-state-and-graph-controls) to compare native payloads and scores through the same daemon.

### Live control regression

Check malformed control bodies, idle and busy reset, bounded profile samples, and peer graph eligibility. Reset the daemon before the cold arm. Compare first scores, output IDs, usage, and finish reasons against the restored arm. The [former resident-cache guide](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/docs/prefix-cache.md#verification) retains GPU-slot, Tensor Core parity, and control-test procedures for the former implementation.

### Flush verification

At idle, try both flush methods with reuse on and off. Check the [status, body, and headers](#cache-flush), including `?timeout=invalid`. With reuse on, prime a cacheable prompt and repeat it. Verify that a successful flush preserves cumulative counters and that the next two requests miss, then hit.

Hold a stream active and send three overlapping HTTP flushes in order with short, short, and zero timeouts. Keep the stream active through both deadlines. Expect two timeout replies followed by the busy reply. Verify that a failed flush preserves a primed hit. Repeat with two longer waits and finish the stream. Expect both flushes to succeed in order. Check `/health` while the flush worker waits.

### W3-prefix

The historical K0 variant used exact prefill, a 6,401-ID off-branch prime, and four 8,192-ID requests. Its warm gate required 25,600 restored IDs and 7,168 computed prompt rows. The deadlines and capacity gates apply to that pinned configuration. Use the current [K4 warm-prefix diagnostic](../benchmarks/paired/README.md#prefix-reuse-diagnostic) for observed hits on the two engines. [Archived W3-prefix procedure](https://github.com/para1lel/sglang.cpp/blob/28148cda233e5522274e258d5eb5caa9ab83e313/docs/prefix-cache.md#w3-prefix)
