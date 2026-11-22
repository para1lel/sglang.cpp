# Native prefix cache

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Configure prefix reuse, validate salts, and report usage. | [CLI](../apps/sglang-server.cpp#L174)<!-- prefix: "} else if (o" --> · [Salt validation](../apps/server/text_request.cpp#L720)<!-- prefix: "void set_cac" --> · [Usage writer](../apps/server/text_completion.cpp#L50)<!-- prefix: "void append_" --> |
| HTTP controls | Dispatch cache reset, metrics, and bounded scheduler profiling. | [Control routes](../apps/server/prefix_controls.cpp#L482)<!-- prefix: "void Connect" --> |
| Batch | Restore request state, schedule remaining prompt rows, and publish completed checkpoints. | [Admission](../src/decoder_batch.cpp#L384)<!-- prefix: "int DecoderB" --> · [Checkpoint publication](../src/decoder_batch.cpp#L922)<!-- prefix: "cache->store" --> |
| Cache | Match checkpoint keys and own the copies until eviction or batch teardown. | [Lookup](../src/decoder_prefix_cache.cpp#L130)<!-- prefix: "int PrefixCa" --> · [Restore](../src/decoder_prefix_cache.cpp#L151)<!-- prefix: "int PrefixCa" --> · [Store](../src/decoder_prefix_cache.cpp#L185)<!-- prefix: "void PrefixC" --> |

## Runtime path

Use `--prefix-cache-entries 1` through `4` for native prefix reuse. The default `0` disables reuse. Omit `--eager`. The default `exact` prefill policy supports prefix reuse at MTP depths 0–4. With tensor-core prefill and prefix reuse, the server uses K0 if you omit the depth option. If you set a depth with tensor-core prefix reuse, use `--mtp-steps 0`. The server rejects explicit positive depths for this combination. Prefix reuse works at every server capacity, including `--max-active 1`. The [default graph policy](batch-graphs.md) applies to eligible decode steps. [CLI validation](../apps/sglang-server.cpp#L88)<!-- prefix: "void configu" --> · [Control flag](../apps/sglang-server.cpp#L338)<!-- prefix: "}, health, a" -->

The three [ordered text policies](fp8-prefill-projections.md#tensor-core-projections) require prefix reuse off. Media requests bypass prefix reuse under every policy.

Native text, chat, benchmark ID, and `/generate` requests accept an optional `cache_salt` string of up to 256 UTF-8 bytes. Omission uses an empty string. See [native generation options](native-generate.md#constraints-and-failure-behavior) for null and per-item salts. Invalid salts return HTTP 400 even when reuse is disabled. Native text, chat, and `/generate` support greedy and sampled requests with prefix reuse. Text and chat also support finite-choice requests. The benchmark ID route keeps its greedy policy.

C++ callers set `DecoderBatchOptions::prefix_cache_entries` and `DecoderOptions::prefix_cache_salt`. `DecoderBatch::admit` rejects salts longer than 256 bytes and returns the restored token count on success. Tensor-core batches with prefix reuse reject positive MTP depth at admission. [Batch options](../include/sglang/decoder_batch.h#L54)<!-- prefix: "struct Decod" --> · [MTP guard](../src/decoder_batch.cpp#L475)<!-- prefix: "if (state_->" --> · [Token-ID parser](../apps/server/completions.cpp#L28)<!-- prefix: "CompletionRe" -->

JSON usage and SSE usage for generation count the full prompt in `prompt_tokens`. `prompt_tokens_details.cached_tokens` counts restored prompt IDs. It is zero on a miss, with the cache off, or for media. A generation request with a zero output budget prefills only if it needs prompt scores. [Token-ID usage](../apps/server/completions.cpp#L97)<!-- prefix: "std::string " -->

The checkpoint boundary is the largest multiple of 128 within the prompt length and 6,400-token limit. Prompts shorter than 128 tokens have no checkpoint. Lookup chooses the longest eligible checkpoint within this boundary with matching prefix IDs and salt. [Boundary](../src/decoder_prefix_cache.cpp#L126)<!-- prefix: "int PrefixCa" --> · [Lookup](../src/decoder_prefix_cache.cpp#L130)<!-- prefix: "int PrefixCa" -->

### Checkpoint compatibility and resume

For a checkpoint of `P` prompt IDs, target KV covers positions `0` through `P-1`. The entry also stores GDN state and the target hidden and residual rows after position `P-1`. A paired checkpoint contains draft KV for positions `0` through `P-2`. Each draft position needs the next token's embedding, so the checkpoint leaves draft position `P-1` open. [Restore](../src/decoder_prefix_cache.cpp#L151)<!-- prefix: "int PrefixCa" --> · [Draft copy](../src/decoder_mtp.cpp#L511)<!-- prefix: "void MtpDraf" -->

K0 can use either a target-only or a paired entry. Depths 1–4 share paired entries, and lookup skips target-only entries even if they match a longer prefix. If no eligible pair matches, the lane prefills from the start. Publication can fill in the draft state of the same target-only key without an eviction. Publishing K0 state for a key that already has a pair preserves the pair. [Store](../src/decoder_prefix_cache.cpp#L185)<!-- prefix: "void PrefixC" -->

A partial hit copies state into the lane's private allocations. At positive depth, the lane seeds draft position `P-1` with the first token ID in the new suffix. It then prefills the suffix. The same delayed seed applies when a cold prefill continues past its checkpoint. [Delayed draft seed](../src/decoder_batch.cpp#L643)<!-- prefix: "if (rows[slo" -->

For native generation prompt scores, lookup also caps `P` at `logprob_start_len` so the model runs every row needed for scoring. An offset of zero, including OpenAI echo scoring, starts prefill at the beginning. [Pointwise scoring](serving/pointwise-score.md#runtime-path) always prefills from the beginning and leaves prefix cache counters and entries untouched.

On a full hit, the target computes logits for the first output token from the restored hidden and residual rows. The request's current [grammar](text-serving.md#finite-choice-constraints) and [sampler and seed](text-serving.md#sampling) control selection. Restore copies, draft seeding, and full-hit scoring stay outside the target row cap. Full-hit scoring stays outside the packed decode width and graph mask. Eligible peers can replay on the same step. If the first verification round runs, the draft seeds position `P-1` with the first committed output ID. A request that ends on this first output leaves the draft position open. Subsequent decode follows [MTP row accounting](mtp.md#runtime-path) and [batch graph eligibility](batch-graphs.md#runtime-path). [Full hit](../src/decoder_batch.cpp#L626)<!-- prefix: "if (lane.res" -->

### Tensor-core scheduling

With tensor-core prefill and reuse enabled, both batch advance methods use the same planner. With a positive cap, the tile size is the largest power of two at or below the smaller of 128 and the cap. The planner rotates priority across slots. It reserves a tile or tail if the first ready lane has prompt work, then assigns one row per ready decode lane while space remains. Spare rows can fund multiple whole tiles or tails per slot in one step. A tile stops at the checkpoint or prompt end. This preserves tile boundaries when peers change. The zero-cap C++ planner retains a separate 128-row prompt budget. Exact and cache-off execution use the [ordinary planner](batch-graphs.md#runtime-path). [Capped planner](../src/decoder_batch.cpp#L213)<!-- prefix: "StepPlan pla" --> · [Legacy planner](../src/decoder_batch.cpp#L185)<!-- prefix: "PrefillRows " -->

## Constraints and failure behavior

An entry stores four MXFP8 K/V payload and scale planes for each of 16 target attention layers. It stores BF16 convolution history and FP32 recurrent state for 48 GDN layers, plus BF16 hidden and residual rows. A full entry uses 50 logical pages. Each configured entry reserves 370,233,344 device bytes when the batch starts. An exact batch whose decoder loaded MTP weights reserves another 13,516,800 bytes per entry for one draft attention layer. These reservations precede the resident capacity calculation. Restoring a prefix leaves the lane's usual page-rounded [request reservation](batch-graphs.md#resident-memory-admission) in effect. [Allocation](../src/decoder_prefix_cache.cpp#L111)<!-- prefix: "PrefixCache:" --> · [Capacity](../src/decoder_batch.cpp#L293)<!-- prefix: "if (options." -->

Store gathers the target KV planes into logical page order. Restore scatters them into the destination request's physical pages. The copy validates the covered pages and rejects aliases into the suffix. A lane owns its restored copy and can continue after eviction. A store publishes the checkpoint after its target and any draft copies complete on the batch stream. A hit or repeated key refreshes recency. A new key evicts the least recently used entry when all entries are occupied. Cancellation can leave an already published checkpoint available. Media requests and zero-budget requests without prompt scoring bypass reuse and publication. [Page mapping](../src/decoder_prefix_cache.cpp#L17)<!-- prefix: "PageMapping " --> · [Completed copy](../src/decoder_prefix_cache.cpp#L218)<!-- prefix: "check(cudaSt" -->

Restore runs during [admission](batch-graphs.md#resident-memory-admission). A failed restore leaves the source checkpoint and hit counters intact. The decoder retires the new lane and preserves its peers if the failure permits recovery and the stream drains. A failed copy into an entry leaves that entry unpublished. A recoverable CUDA error during publication fails the step and retires the active lanes after the stream drains. New lanes can use the batch and reuse the free entry. A terminal CUDA error, failed drain, or checkpoint publication invariant error disables the batch. A tensor-core cache step that lacks multirow workspace fails before publication and disables the batch. An invalid sampled or constrained selection retires only its lane. [Admission recovery](../src/decoder_batch.cpp#L512)<!-- prefix: "} catch (con" --> · [Workspace guard](../src/decoder_prefill.cpp#L912)<!-- prefix: "if (tensor_c" --> · [Step recovery](../src/decoder_batch.cpp#L954)<!-- prefix: "} catch (con" -->

With `--trace-batches`, `entries` and `pages` count published state, and `device_bytes` counts the reservation. `hits`, `cached_tokens`, `prefill_tokens`, `stores`, and `evictions` are cumulative for the batch. `prefill_tokens` counts cache-eligible prompt rows processed on the batch path. Request memory and batch scratch have separate counters. [Retired request blocks](serving/request-memory.md) belong to the batch and are outside the prefix cache reservation. [Trace](../include/sglang/server/batch_trace.h#L73)<!-- prefix: "inline void " --> · [Counters](../src/decoder_prefix_cache.cpp#L239)<!-- prefix: "DecoderPrefi" -->

The draft counters separate reserved device bytes (`draft_device_bytes`, included in `device_bytes`) from restored rows (`draft_cached_rows`). `draft_prefill_rows` counts rows seeded during prompt prefill, including the delayed row on a partial hit. It excludes drafts seeded during verification. `draft_misses` counts positive-depth lookups that found a matching target-only entry and no eligible pair. `/metrics` exposes these counters as `sglang_cpp:prefix_cache_draft_device_bytes`, `sglang_cpp:prefix_cache_draft_cached_rows_total`, `sglang_cpp:prefix_cache_draft_prefill_rows_total`, and `sglang_cpp:prefix_cache_draft_misses_total`. [Draft series](../apps/server/loads.cpp#L228)<!-- prefix: "emit(\"sglang" -->

### Native HTTP controls

The metrics, reset, and profile routes require native prefix reuse. They return HTTP 404 with reuse off. [HTTP admission and shutdown](http-server.md#http-admission-and-shutdown) apply with reuse on or off.

| Route | Input | Result |
| --- | --- | --- |
| Health routes | Empty GET | Use the [liveness, readiness, and generation progress responses](http-server.md#health-and-readiness). |
| `POST /reset_prefix_cache` | `{}` | Returns `{"success":true}` after the idle decoder worker invalidates all checkpoints. Returns HTTP 409 for a waiting or active completion, an installed health job, or another reset in progress. |
| `GET /metrics` | Empty body | Returns Prometheus text with native provenance and `engine="0"`. |
| `POST /start_profile` | Exactly three fields. `profile_prefix` is a string of 1–64 bytes. `max_iterations` is an integer from 1 to 64. `profiler_kwargs` equals `{"with_stack":false}`. | Opens one capture and returns `{"success":true,"profiler":"native-batch-steps"}`. Returns HTTP 409 if a capture is open. |
| `POST /stop_profile` | `{}` | Closes the capture and adds `profile_prefix` and `steps` to the JSON response. Returns HTTP 409 if no capture is open. |

Successful controls return HTTP 200. Malformed POST control bodies return HTTP 400. Reset clears published entries and pages. It preserves the allocation and cumulative counters. During reset, new completions can enter the queue. Completion workers wait for the reset response write to finish before claiming queued requests. `/health` and `/ready` remain available while reset is pending. [Reset admission](../apps/server/prefix_controls.cpp#L373)<!-- prefix: "void Connect" --> · [Reset response](../apps/server/prefix_controls.cpp#L386)<!-- prefix: "void Connect" --> · [Invalidation](../src/decoder_prefix_cache.cpp#L233)<!-- prefix: "void PrefixC" --> · [Profile input](../apps/server/prefix_controls.cpp#L212)<!-- prefix: "void start_p" -->

The `vllm:` names support the frozen client. `num_requests_running` counts claimed completions through socket closure. `num_requests_waiting` counts queued completions after removal of disconnected clients. `kv_cache_usage_perc` is published prefix pages divided by reserved prefix pages. `prefix_cache_hits_total` counts restored prompt IDs, and `external_prefix_cache_hits_total` is zero. The first published user output adds the full prompt length to `prompt_tokens_total` and increments `request_prefill_kv_computed_tokens_count`. `request_prefill_kv_computed_tokens_sum` counts cache-eligible prompt rows. `generation_tokens_total` counts published user output IDs. The `sglang_cpp:` series report entries, pages, reserved bytes, hit requests, stores, and evictions. See [generation probe accounting](http-server.md#generation-probe-lifecycle). [Series](../apps/server/loads.cpp#L203)<!-- prefix: "std::string " --> · [Publication](../apps/server/batch_service.cpp#L329)<!-- prefix: "bool BatchSe" -->

A native capture records completed `advance_chunked` calls from nonempty iterations that start after profile start. Each sample contains `elapsed_ns` in host time, `width`, total `prefill_rows`, `decode_lanes`, and the `graph_slots` bit mask. The capture stays open after reaching its sample limit. [Step timing](../apps/server/batch_service.cpp#L432)<!-- prefix: "steps = batc" --> · [Sample recording](../apps/server/batch_service.cpp#L380)<!-- prefix: "void BatchSe" --> · [Response](../apps/server/prefix_controls.cpp#L231)<!-- prefix: "void stop_pr" -->

### Cache flush

`GET /flush_cache` and `POST /flush_cache` work with prefix reuse on or off. Both ignore a successfully framed body. Other methods return HTTP 405 with `Allow: GET, POST`. HEAD omits the body. The optional `timeout` query parameter is a nonnegative number in seconds. The reader form-decodes query fields (`+` means space) and uses the last `timeout`. The default is zero. Positive infinity allows an indefinite wait for idle. [Native input](../apps/server/flush_cache.cpp#L90)<!-- prefix: "FlushCacheRe" --> · [Pinned handler](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1060)<!-- prefix: "@app.api_rou" -->

| Condition | HTTP response |
| --- | --- |
| Idle, including with reuse off | 200, `Cache flushed.\nPlease check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)\n` |
| Busy at zero timeout | 400, `Flush cache failed.\n` |
| Still busy when a positive timeout expires | 400, `Timed out waiting for idle state.` |
| Empty, malformed, negative, or NaN timeout | 400 JSON with `object="error"`, `type="Bad Request"`, `param=null`, `code=400`, and a `message` identifying `query.timeout`. |

The first three bodies are plain text without a `Content-Type` header. Each `\n` represents one line feed. The reader validates the timeout before queueing. [Replies](../apps/server/flush_cache.cpp#L107)<!-- prefix: "void write_f" --> · [Validation error](../apps/server/flush_cache.cpp#L75)<!-- prefix: "void write_i" -->

A separate worker claims HTTP flushes in FIFO order. Each timeout starts when the worker handles that request. Queued or claimed completion, generation, and tokenization requests count as busy. The scheduler also checks for active generation, an installed health job, or another reset. During a deferred wait, other requests can enter and finish. [FIFO](../apps/server/prefix_controls.cpp#L436)<!-- prefix: "void Connect" --> · [Wait loop](../apps/server/prefix_controls.cpp#L406)<!-- prefix: "FlushCacheRe" --> · [Scheduler guard](../apps/server/batch_service.cpp#L842)<!-- prefix: "bool BatchSe" -->

The pinned single-tokenizer HTTP communicator holds its FIFO lock through the scheduler reply. A second direct scheduler call that bypasses the communicator can return `Another flush_cache is already in progress.` while a deferred request is pending. [Communicator](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/communicator.py#L41)<!-- prefix: "async def qu" --> · [Scheduler wrapper](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler_components/flush_wrapper.py#L22)<!-- prefix: "def handle(s" -->

Native flush invalidates all published target and draft prefixes. Entries and pages drop to zero. Reserved storage, cumulative counters, the loaded tokenizer, and compiled tool grammars persist. The pinned scheduler also resets memory pools, grammar state, and metrics. Its tokenizer clears the preprocessing cache when a multimodal processor is present. [Batch reset](../src/decoder_batch.cpp#L559)<!-- prefix: "void Decoder" --> · [Tokenizer](../apps/server/text_completion.cpp#L20)<!-- prefix: "const Tokeni" --> · [Tool grammars](../apps/server/chat_tool_constraints.cpp#L71)<!-- prefix: "std::shared_" --> · [Pinned scheduler](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler.py#L6285)<!-- prefix: "self.tree_ca" --> · [Pinned tokenizer](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_control_mixin.py#L330)<!-- prefix: "async def fl" -->

## Verification

Follow the [prefix cache CUDA and route checks](../tests/prefix/README.md). Cover identity and remapped page tables, transfer failures, all four KV planes, and the 6,400-token limit. Compare greedy cold, full-hit, and suffix-hit routes from the same binary with exact prefill. Check page boundaries, changed prefix IDs, changed salts, zero budget, cancellation, and eviction. Compare committed IDs and finish reasons. In usage, require the expected cached-token count and identical prompt and completion counts.

Follow the [MTP prefix checks](../tests/prefix/README.md#prefix-cache-with-mtp) for exact full and partial restores. Compare each warm request with its own cold control. Cover a changed suffix, target-only misses, paired restores at different depths, and paired restores into K0. Check the first target and draft rows, cache and step counters, finish reasons, and continued output. Check requests with budgets of zero and one output token. Include a tight target row cap, seeded sampling, and constrained selection. Exercise cancellation and remapped request arenas. Check the diagnostic for tensor-core reuse at positive depth.

For tensor-core prefill, enable reuse for both cold and warm checks. Use a fresh cache for the cold check and collect its own output and device state. Compare that state with a remapped restore and suffix prefill under concurrency. Check logical KV payload and scales, GDN state, hidden and residual rows, full BF16 logits, and selected IDs. Include prompts just below, at, and above a page boundary. Inject workspace allocation failure and require no checkpoint publication from the failed step.

Repeat full and partial hits with seeded sampling and a second seed or sampling policy. Change the finite-choice grammar while reusing the same prefix and salt. Compare each warm result with its own cold request, including the first output ID. Check that a full hit stays out of the replay mask while eligible peers replay. Inject a selection failure and check that the failed lane retires and its peers finish.

For the HTTP controls, reset a populated idle cache and check zero entries and pages. Check that reserved bytes and cumulative counters stay constant. Try reset during a stream and require HTTP 409. Check malformed bodies, duplicate profile start, and stop without an open capture. After a profiled cohort, check the sample cap and the step fields.

Run the shared [transport checks](http-server.md#verification) for queue saturation, disconnects, and slot release. Use the native decoder on a reserved B200 for CUDA correctness, output IDs, and frozen timing acceptance.

### Live control regression

Expose only the reserved B200 as CUDA device 0. Start the native server with these options and tracing off.

```bash
"$SGLANG_SERVER" --checkpoint "$CHECKPOINT" --device 0 --port "$port" \
    --max-active 2 --prefix-cache-entries 2 --fp8-prefill exact \
    --mtp-steps 0 --batch-cuda-graphs
```

Wait for `GET /ready` to return HTTP 200. Run the test from the repository root. Use a new evidence directory under an existing parent.

```bash
/data1/shared/dongwu.chen/.pixi/envs/default/bin/python -B \
    tests/prefix/live_controls.py --port "$port" --evidence "$run/live-controls"
```

Require exit zero and `passed.json`. The [live test](../tests/prefix/live_controls.py#L223)<!-- prefix: "def run(cont" --> checks malformed controls, idle and busy reset, and exact and partial hits. It also checks bounded profiles, cache eviction, queue overflow, canceled waiters, and survivor completion. Use the separate W3-prefix protocol for workload timing.

### Flush verification

At idle, try both flush methods with reuse on and off. Check the [status, body, and headers](#cache-flush), including `?timeout=invalid`. With reuse on, prime a cacheable prompt and repeat it to confirm a hit. Flush, check entries, pages, reservations, and cumulative counters, then repeat the prompt twice. Expect a miss followed by a hit.

Hold a stream active and send three overlapping HTTP flushes in order with short, short, and zero timeouts. Keep the stream active through both deadlines. Expect two timeout replies followed by the busy reply. Verify that a failed flush preserves a primed hit. Repeat with two longer waits and finish the stream. Expect both flushes to succeed in order. Check `/health` while the flush worker waits.

### W3-prefix

Use the [frozen variant fixtures](../benchmarks/variants/README.md#reproduce-on-the-baseline-host). Start the native server with `--max-active 4 --batch-cuda-graphs --fp8-prefill exact --mtp-steps 0 --prefix-cache-entries 4`. Wait for HTTP readiness. The frozen client resets the idle cache before each cohort. The cold control uses distinct salts and no prime. Before each warm cohort, the client submits the 6,401-ID off-branch prime with a one-ID output budget. Reset and prime are outside the timed interval. Each cohort submits four greedy 8,192-ID prompts with `ignore_eos: true`. Require 1,024 output IDs per request and finish reason `length`.

Run from the repository root. Set `endpoint` to the native loopback origin. Set `run` to the evidence directory containing `fixtures/` and `fixtures-frozen-utc.txt`. Keep `"$run/W3-prefix/capture"` absent before capture.

```bash
python=/data1/shared/dongwu.chen/.pixi/envs/default/bin/python
mkdir -p "$run/W3-prefix"
"$python" -B -m benchmarks.variants.client W3-prefix \
    "$endpoint" "$run/fixtures" "$run/W3-prefix/capture"
"$python" -B - "$run" <<'PY'
from pathlib import Path
from sys import argv
from benchmarks.variants.verify import verify_capture, verify_fixtures

run = Path(argv[1])
verify_fixtures(run / "fixtures")
verify_capture(run, "W3-prefix")
PY
```

Require cold and warm output IDs to match. Require zero restored IDs on the cold control. For each warm cohort, require four completed prefills, 25,600 restored IDs, 7,168 computed prompt rows, and zero external hits. In a separate traced warm run, require one 50-page entry and zero new evictions. [Capture sequence](../benchmarks/variants/client.py#L256)<!-- prefix: "def run_vllm" --> · [Capture verification](../benchmarks/variants/verify.py#L344)<!-- prefix: "def verify_c" -->

Keep two warmup cohorts, five measured cohorts, and one separate diagnostic cohort. Require stream overlap and a submission span of at most 50 milliseconds. Keep the 90-second socket, 15-second barrier, and 360-second cohort limits. Disable tracing and profiling for measured cohorts. Report cache reservation and peak device memory separately. Use the [frozen variant gates](../benchmarks/variants/README.md#evidence-and-comparison) for performance acceptance.

If the cold control times out, record the frozen run as failed. A separate longer-socket cold run can supply an output comparison with the warm cohort. Apply the original deadlines to the acceptance run.

For a separate tensor-core comparison, use a fresh evidence directory and change the native flag to `--fp8-prefill tensor-core`. Keep the frozen client, pinned vLLM reference, fixture bytes, deadlines, and counter checks. Report the native policy for each result. The frozen adoption gate applies to the explicit exact-prefill run. The exact-prefill qualification remains open.
