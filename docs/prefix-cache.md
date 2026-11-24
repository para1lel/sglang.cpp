# Native prefix cache

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Configure prefix reuse, validate salts, and report usage. | [CLI](../apps/sglang-server.cpp#L143)<!-- prefix: "} else if (o" --> · [Salt validation](../apps/server/text_request.cpp#L711)<!-- prefix: "void set_cac" --> · [Usage writer](../apps/server/text_completion.cpp#L50)<!-- prefix: "void append_" --> |
| HTTP controls | Dispatch cache reset, metrics, and bounded scheduler profiling. | [Control routes](../apps/server/prefix_controls.cpp#L485)<!-- prefix: "void Connect" --> |
| Batch | Restore request state, schedule remaining rows, and publish committed checkpoints. | [Admission](../src/decoder_batch.cpp#L397)<!-- prefix: "int DecoderB" --> · [Checkpoint publication](../src/decoder_batch.cpp#L852)<!-- prefix: "void Decoder" --> |
| Prefix index | The host `PrefixIndex` matches token IDs and salts, tracks target and draft coverage, and manages recency and validity by stable slot ID. | [Index](../include/sglang/runtime/cache/prefix_index.h#L18)<!-- prefix: "class Prefix" --> · [Match](../include/sglang/runtime/cache/prefix_index.h#L58)<!-- prefix: "Match find(c" --> |
| Cache coordination | `PrefixCache` sequences index lookup, snapshot transfers, and publication and records cache counters. | [Cache](../include/sglang/decoder_prefix_cache.h#L11)<!-- prefix: "class Prefix" --> · [Store sequence](../src/decoder_prefix_cache.cpp#L403)<!-- prefix: "void PrefixC" --> |
| Snapshot storage | `GpuPrefixSnapshotStore` owns GPU payloads by slot. Transfers follow the request's live page mapping. | [GPU store](../include/sglang/runtime/cache/gpu_prefix_snapshot_store.h#L12)<!-- prefix: "class GpuPre" --> · [Snapshot copy](../src/decoder_prefix_cache.cpp#L267)<!-- prefix: "void GpuPref" --> · [Restore copy](../src/decoder_prefix_cache.cpp#L287)<!-- prefix: "void GpuPref" --> |

## Runtime path

The server enables prefix reuse with four slots by default. Set `--prefix-cache-entries 0` to disable reuse or `1`–`4` to set the slot limit. `--eager` controls graph replay independently. [Cache option](../apps/sglang-server.cpp#L39)<!-- prefix: "int prefix_c" -->

Native text, chat, benchmark ID, and `/generate` requests accept an optional `cache_salt` string of up to 256 UTF-8 bytes. Omission uses an empty string. See [native generation options](native-generate.md#constraints-and-failure-behavior) for null and per-item salts. Invalid salts return HTTP 400 even when reuse is disabled. Native text, chat, and `/generate` support greedy and sampled requests with prefix reuse. Text and chat also support finite-choice requests. The benchmark ID route keeps its greedy policy.

C++ callers set `DecoderBatchOptions::prefix_cache_entries` and `DecoderOptions::prefix_cache_salt`. `DecoderBatch::admit` rejects salts longer than 256 bytes and returns the restored token count on success. Positive MTP depths require loaded draft weights. [Batch options](../include/sglang/decoder_batch.h#L59)<!-- prefix: "struct Decod" --> · [MTP admission](../src/decoder_batch.cpp#L427)<!-- prefix: "if (options." --> · [Token-ID parser](../apps/server/completions.cpp#L28)<!-- prefix: "CompletionRe" -->

Generation usage counts the full prompt in `prompt_tokens`. `prompt_tokens_details.cached_tokens` counts restored prompt IDs. A miss or disabled cache reports zero. A zero-output generation request runs prefill only if it needs prompt scores. [Token-ID usage](../apps/server/completions.cpp#L97)<!-- prefix: "std::string " -->

### Checkpoint boundaries

The server attempts a checkpoint every 8,192 processed target tokens, up to the 262,144-token model context. It also attempts to save newly processed state at successful request end. It limits prefill and MTP grants so they reach each periodic boundary without crossing it. One checkpoint suffices when a request ends on a periodic boundary. Lookup chooses the longest eligible match by token IDs, salt, and covered media. [Cadence and cap](../include/sglang/decoder_batch.h#L19-L21)<!-- prefix: "inline const" --> · [Grant limit](../src/decoder_batch.cpp#L45)<!-- prefix: "int checkpoi" --> · [Lookup](../include/sglang/runtime/cache/prefix_index.h#L58)<!-- prefix: "Match find(c" -->

The request-end checkpoint holds the latest committed target state. A request can end before the target processes its last emitted token. A 128-token prompt that emits 128 tokens at the length limit leaves 255 processed IDs. A follow-up prompt containing the full history can restore those 255 IDs and prefill the last emitted ID with its new suffix. A session can continue across requests. Closing a connection alone creates no checkpoint.

### Checkpoint compatibility and resume

For a checkpoint of `P` processed IDs, target KV covers positions `0` through `P-1`. The entry also stores GDN state and the terminal hidden and residual rows. Paired entries retain `P-1` draft rows. Each draft position needs the next token's embedding, so position `P-1` stays open. Target and draft copies leave unused rows of the last page untouched. [Target copy](../src/decoder_prefix_cache.cpp#L69)<!-- prefix: "void copy_ta" --> · [Restore](../src/decoder_prefix_cache.cpp#L378)<!-- prefix: "int PrefixCa" -->

K0 can use either a target-only or a paired entry. Depths 1–4 share paired entries, and lookup skips target-only entries even if they match a longer prefix. If no eligible pair matches, the lane prefills from the start. Publication can fill in the draft state of the same target-only key without an eviction. Publishing K0 state for a key that already has a pair preserves the pair. [Existing coverage check](../include/sglang/runtime/cache/prefix_index.h#L91)<!-- prefix: "bool touch_i" -->

A partial hit copies state into the lane's private allocations. At positive depth, the lane seeds draft position `P-1` with the first token ID in the new suffix. It then prefills the suffix. The same delayed seed applies when a cold prefill continues past its checkpoint. [Delayed draft seed](../src/decoder_batch.cpp#L597)<!-- prefix: "if (rows[slo" -->

Prompt token scores need the target row before each scored token. Restore is limited to `logprob_start_len` and leaves at least the last prompt row to compute. An offset of zero, including OpenAI echo scoring, restores no rows. [Pointwise scoring](serving/pointwise-score.md#runtime-path) can use a full hit for next-token label scores. [Restore limit](../src/decoder_batch.cpp#L368)<!-- prefix: "if (auto& ca" -->

On a full hit, the target computes the next-token logits from the restored hidden and residual rows. The current [grammar](text-serving.md#finite-choice-constraints) and [seed and sampler](text-serving.md#sampling) control selection. Restore and full-hit scoring stay outside the target row cap. Other lanes retain their normal decode selection. At positive depth, any later verification seeds draft position `P-1` with the first committed output ID. [Full hit](../src/decoder_batch.cpp#L580)<!-- prefix: "if (lane.res" --> · [MTP row accounting](mtp.md#runtime-path)

### Media identity and prefill

The server binds each covered media span to its vision encoder, decoded patches, feature layout, and rotary positions. A change inside the covered prefix causes a miss. Generated rows use the same rotary position on all axes. A replay must match those positions. External C++ features can run without a cache identity. The cache skips checkpoints that cover those features. [Media match](../src/decoder_prefix_cache.cpp#L329)<!-- prefix: "bool PrefixM" --> · [Media execution](multimodal-decoder.md#runtime-path)

Cold and restored suffix rows use the same [prefill planner](native-prefill-workspace.md#runtime-path). The former exact-only cache route and its fixed-shape Tensor Core probe remain archived. [Historical cache guide](https://github.com/para1lel/sglang.cpp/blob/28148cda233e5522274e258d5eb5caa9ab83e313/docs/prefix-cache.md#tensor-core-scheduling)

## Constraints and failure behavior

Each checkpoint holds its covered MXFP8 pages for 16 target attention layers. It also holds GDN history and recurrent state for 48 layers and terminal BF16 hidden and residual rows. A paired checkpoint also holds draft pages. Slot configuration allocates no GPU snapshot payload at startup. The store allocates payloads on demand. The lane retains its normal page-rounded [request reservation](batch-graphs.md#resident-memory-admission). [Allocation](../src/decoder_prefix_cache.cpp#L221)<!-- prefix: "std::optiona" -->

A live resident request can preallocate snapshot pages at a periodic checkpoint. Its planned end must be later in the same 8,192-token interval and within the live request reservation. The store uses the planned extent to size target pages and, if paired, draft pages for one fewer row. New allocations preserve the shared 2 GiB device reserve. If headroom is insufficient or allocation runs out of memory, the store retries at committed coverage. Other CUDA errors propagate. [Resident extent](../src/decoder_batch.cpp#L867)<!-- prefix: "const int te" --> · [Interval guard](../src/decoder_prefix_cache.cpp#L423)<!-- prefix: "const bool t" --> · [Capacity retry](../src/decoder_prefix_cache.cpp#L259)<!-- prefix: "if (capacity" -->

Target and draft can reuse buffers already in the selected slot. For a resident request, each buffer may exceed the current allocation request by up to 64 pages. The requested size must be at least 64 pages. This check uses the planned extent on the first attempt and committed coverage on retry. Below that threshold, and for nonresident requests, reuse requires an exact page count. Store and restore copy only committed target rows and their paired draft coverage. Spare pages count toward allocated bytes. They add no index coverage and are released with the slot on reclamation or reset. [Buffer reuse](../src/decoder_prefix_cache.cpp#L224)<!-- prefix: "const auto r" -->

The store gathers and scatters pages in logical order. It rejects invalid mappings and aliases into the suffix. The destination owns its copy. A hit refreshes recency. A full cache replaces the least recently used entry. Admission can reclaim snapshots while keeping the shared device memory reserve. An abort that wins the completion claim prevents a new request-end checkpoint. A completed periodic checkpoint can remain available. [Slot selection](../include/sglang/runtime/cache/prefix_index.h#L98)<!-- prefix: "Write select" --> · [Admission reclamation](../src/decoder_prefix_cache.cpp#L449)<!-- prefix: "void PrefixC" -->

Publication validates the live page mapping and prepares GPU storage before invalidating a slot. Insufficient headroom or an allocation failure skips publication and preserves the existing entry. After invalidation, a transfer failure leaves the slot unpublished. The index publishes coverage after the target and draft transfers complete. [Store sequence](../src/decoder_prefix_cache.cpp#L403)<!-- prefix: "void PrefixC" --> · [Completed copy](../src/decoder_prefix_cache.cpp#L284)<!-- prefix: "check(cudaSt" -->

Restore runs during [admission](batch-graphs.md#resident-memory-admission). A failed restore leaves the source checkpoint and hit counters intact. The decoder retires the new lane and preserves its peers if the failure permits recovery and the stream drains. A recoverable CUDA error during publication fails the step and retires the active lanes after the stream drains. New lanes can use the batch and reuse the free entry. A terminal CUDA error, failed drain, or checkpoint publication invariant error disables the batch. An invalid sampled or constrained selection retires only its lane. [Admission recovery](../src/decoder_batch.cpp#L464)<!-- prefix: "} catch (con" --> · [Step recovery](../src/decoder_batch.cpp#L955)<!-- prefix: "} catch (con" -->

With `--trace-batches`, `entries` and `pages` count published checkpoints. `device_bytes` counts allocated snapshot payloads. The hits, cached tokens, computed prompt rows, stores, and evictions are cumulative. Request memory and batch scratch have separate counters. [Trace](../include/sglang/server/batch_trace.h#L75)<!-- prefix: "inline void " --> · [Counters](../src/decoder_prefix_cache.cpp#L468)<!-- prefix: "DecoderPrefi" -->

`draft_device_bytes` counts the draft subset of `device_bytes`. `draft_cached_rows` counts restored draft rows. `draft_prefill_rows` counts rows seeded during prompt prefill, including the delayed row after a partial hit. `draft_misses` counts positive-depth lookups that found a matching target-only entry and no eligible pair. [Draft series](../apps/server/loads.cpp#L228)<!-- prefix: "emit(\"sglang" -->

### Native HTTP controls

The metrics, reset, and profile routes require native prefix reuse. They return HTTP 404 with reuse off. [HTTP admission and shutdown](http-server.md#http-admission-and-shutdown) apply with reuse on or off.

| Route | Input | Result |
| --- | --- | --- |
| Health routes | Empty GET | Use the [liveness, readiness, and generation progress responses](http-server.md#health-and-readiness). |
| `POST /reset_prefix_cache` | `{}` | Returns `{"success":true}` after the idle decoder worker invalidates all checkpoints. Returns HTTP 409 for a waiting or active completion, an installed health job, or another reset in progress. |
| `GET /metrics` | Empty body | Returns Prometheus text with native provenance and `engine="0"`. |
| `POST /start_profile` | Exactly three fields. `profile_prefix` is a string of 1–64 bytes. `max_iterations` is an integer from 1 to 64. `profiler_kwargs` equals `{"with_stack":false}`. | Opens one capture and returns `{"success":true,"profiler":"native-batch-steps"}`. Returns HTTP 409 if a capture is open. |
| `POST /stop_profile` | `{}` | Closes the capture and adds `profile_prefix` and `steps` to the JSON response. Returns HTTP 409 if no capture is open. |

Successful controls return HTTP 200. Malformed POST control bodies return HTTP 400. Reset removes all checkpoints and frees their payloads. Cumulative counters persist. New completions can enter the queue during reset. The worker waits for the reset response write before claiming them. Health and readiness remain available. [Reset admission](../apps/server/prefix_controls.cpp#L376)<!-- prefix: "void Connect" --> · [Reset response](../apps/server/prefix_controls.cpp#L389)<!-- prefix: "void Connect" --> · [Invalidation](../src/decoder_prefix_cache.cpp#L463)<!-- prefix: "void PrefixC" --> · [Profile input](../apps/server/prefix_controls.cpp#L213)<!-- prefix: "void start_p" -->

The `vllm:` names support the frozen client. `num_requests_running` counts claimed completions through socket closure. `num_requests_waiting` excludes disconnected clients. `kv_cache_usage_perc` divides published pages by configured slots times the maximum pages per slot. `prefix_cache_hits_total` counts restored prompt IDs. `external_prefix_cache_hits_total` is zero. The first published user output adds the prompt length to `prompt_tokens_total` and increments `request_prefill_kv_computed_tokens_count`. `request_prefill_kv_computed_tokens_sum` counts computed prompt rows. `generation_tokens_total` counts published user output IDs. The `sglang_cpp:` series report entries, pages, allocated bytes, hit requests, stores, and evictions. [Series](../apps/server/loads.cpp#L203)<!-- prefix: "std::string " --> · [Probe accounting](http-server.md#generation-probe-lifecycle)

A native capture records completed `advance_chunked` calls from nonempty iterations that start after profile start. Each sample contains `elapsed_ns` in host time, `width`, total `prefill_rows`, `decode_lanes`, and the `graph_slots` bit mask. The capture stays open after reaching its sample limit. [Step timing](../apps/server/batch_service.cpp#L456)<!-- prefix: "steps = batc" --> · [Sample recording](../apps/server/batch_service.cpp#L480)<!-- prefix: "profile->ste" --> · [Sample fields](../include/sglang/server/batch_trace.h#L127)<!-- prefix: "inline Batch" --> · [Response](../apps/server/prefix_controls.cpp#L232)<!-- prefix: "void stop_pr" -->

### Cache flush

`GET /flush_cache` and `POST /flush_cache` work with prefix reuse on or off. Both ignore a successfully framed body. Other methods return HTTP 405 with `Allow: GET, POST`. HEAD omits the body. The optional `timeout` query parameter is a nonnegative number in seconds. The reader form-decodes query fields (`+` means space) and uses the last `timeout`. The default is zero. Positive infinity allows an indefinite wait for idle. [Native input](../apps/server/flush_cache.cpp#L90)<!-- prefix: "FlushCacheRe" --> · [Pinned handler](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1060)<!-- prefix: "@app.api_rou" -->

| Condition | HTTP response |
| --- | --- |
| Idle, including with reuse off | 200, `Cache flushed.\nPlease check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)\n` |
| Busy at zero timeout | 400, `Flush cache failed.\n` |
| Still busy when a positive timeout expires | 400, `Timed out waiting for idle state.` |
| Empty, malformed, negative, or NaN timeout | 400 JSON with `object="error"`, `type="Bad Request"`, `param=null`, `code=400`, and a `message` identifying `query.timeout`. |

The first three bodies are plain text without a `Content-Type` header. Each `\n` represents one line feed. The reader validates the timeout before queueing. [Replies](../apps/server/flush_cache.cpp#L107)<!-- prefix: "void write_f" --> · [Validation error](../apps/server/flush_cache.cpp#L75)<!-- prefix: "void write_i" -->

A separate worker claims HTTP flushes in FIFO order. Each timeout starts when the worker handles that request. Queued or claimed completion, generation, and tokenization requests count as busy. The scheduler also checks for active generation, an installed health job, or another reset. During a deferred wait, other requests can enter and finish. [FIFO](../apps/server/prefix_controls.cpp#L439)<!-- prefix: "void Connect" --> · [Wait loop](../apps/server/prefix_controls.cpp#L409)<!-- prefix: "FlushCacheRe" --> · [Scheduler guard](../apps/server/batch_service.cpp#L839)<!-- prefix: "bool BatchSe" -->

The pinned single-tokenizer HTTP communicator holds its FIFO lock through the scheduler reply. A second direct scheduler call that bypasses the communicator can return `Another flush_cache is already in progress.` while a deferred request is pending. [Communicator](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/communicator.py#L41)<!-- prefix: "async def qu" --> · [Scheduler wrapper](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler_components/flush_wrapper.py#L22)<!-- prefix: "def handle(s" -->

Native flush invalidates all published checkpoints and frees their device payloads. Cumulative counters, the loaded tokenizer, and compiled tool grammars persist. The pinned SGLang scheduler also resets memory pools, grammar state, and metrics. Its tokenizer clears the preprocessing cache when a multimodal processor is present. [Tokenizer](../apps/server/text_completion.cpp#L20)<!-- prefix: "const Tokeni" --> · [Tool grammars](../apps/server/chat_tool_constraints.cpp#L71)<!-- prefix: "std::shared_" --> · [Pinned scheduler](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler.py#L6285)<!-- prefix: "self.tree_ca" --> · [Pinned tokenizer](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_control_mixin.py#L330)<!-- prefix: "async def fl" -->

## Verification

Follow the [CUDA state and route checks](../tests/prefix/README.md#cuda-state-and-routes) for exact coverage, remapping, and transfer failures. Use the [committed target state checks](../tests/prefix/README.md#committed-target-state) for periodic boundaries, request-end reuse, and cancellation. Compare cold and restored continuations at a fixed prefill policy. Check usage, finish reasons, identity changes, and eviction.

For resident capacity, compare a preallocated checkpoint with exact allocation. Require identical committed target and draft state and unchanged logical coverage. Check allocated bytes separately. Test reuse without allocation, pressure fallback, failed copy recovery, and release on reset.

Follow the [MTP prefix checks](../tests/prefix/README.md#prefix-cache-with-mtp). Reset the reference cache before each cold admission and confirm it restores no target or draft rows. Cover target-only misses, paired restores at different depths, and paired restores into K0. Check first scores, continuation, and exact target and draft coverage. Include a changed suffix, budgets zero and one, remapped arenas, constrained selection, and cancellation.

Use the [pinned Tensor Core cache parity procedure](../tests/prefix/README.md#historical-tensor-core-cache-parity) to reproduce the historical byte comparison. For a separate workspace allocation failure check on that mode, require that the failed step publishes no checkpoint.

Repeat full and partial hits under a second seed or finite-choice grammar. Compare each warm result with its own cold request, including its first output. Check that full-hit scoring stays outside the decode width. Verify peer graph eligibility separately. Inject a selection failure and check that the failed lane retires and its peers finish.

Reset a populated idle cache and check zero entries, pages, and allocated bytes. Verify cumulative counters stay constant. Try reset during a stream and require HTTP 409. Check malformed bodies, duplicate profile start, and stop without an open capture. After a profiled cohort, check the sample cap and step fields.

Run the shared [transport checks](http-server.md#verification) for queue saturation, disconnects, and slot release. Use the native decoder on a reserved B200 for CUDA correctness, output IDs, and frozen timing acceptance.

### Live control regression

Expose only the reserved B200 as CUDA device 0. Start the native server with these options and tracing off.

```bash
"$SGLANG_SERVER" --checkpoint "$CHECKPOINT" --device 0 --port "$port" \
    --max-active 2 --prefix-cache-entries 2 \
    --mtp-steps 0 --batch-cuda-graphs
```

Wait for `GET /ready` to return HTTP 200. Run the test from the repository root. Use a new evidence directory under an existing parent.

```bash
/data1/shared/dongwu.chen/.pixi/envs/default/bin/python -B \
    tests/prefix/live_controls.py --port "$port" --evidence "$run/live-controls"
```

Require exit zero and `passed.json`. The [live test](../tests/prefix/live_controls.py#L225)<!-- prefix: "def run(cont" --> checks malformed controls, idle and busy reset, exact and partial hits, and bounded profiles. With two K0 streams decoding, require four steps with `width=2`, `decode_lanes=2`, `prefill_rows=0`, and `graph_slots=3`. Also check eviction, queue overflow, canceled waiters, and survivor completion.

### Flush verification

At idle, try both flush methods with reuse on and off. Check the [status, body, and headers](#cache-flush), including `?timeout=invalid`. With reuse on, prime a cacheable prompt and repeat it. Flush and check zero entries, pages, and allocated bytes. Verify cumulative counters stay constant. Send the prompt twice and expect a miss followed by a hit.

Hold a stream active and send three overlapping HTTP flushes in order with short, short, and zero timeouts. Keep the stream active through both deadlines. Expect two timeout replies followed by the busy reply. Verify that a failed flush preserves a primed hit. Repeat with two longer waits and finish the stream. Expect both flushes to succeed in order. Check `/health` while the flush worker waits.

### W3-prefix

The historical K0 variant used exact prefill, a 6,401-ID off-branch prime, and four 8,192-ID requests. Its warm gate required 25,600 restored IDs and 7,168 computed prompt rows. The deadlines and capacity gates apply to that pinned configuration. Use the current [K4 warm-prefix diagnostic](../benchmarks/paired/README.md#prefix-reuse-diagnostic) for observed hits on the two engines. [Archived W3-prefix procedure](https://github.com/para1lel/sglang.cpp/blob/28148cda233e5522274e258d5eb5caa9ab83e313/docs/prefix-cache.md#w3-prefix)
