# Native prefix cache

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Configure prefix reuse, validate salts, and report usage. | [CLI](../apps/sglang-server.cpp#L82)<!-- prefix: "} else if (o" --> · [Salt parser](../apps/server/text_request.cpp#L298)<!-- prefix: "void set_cac" --> · [Usage writer](../apps/server/text_completion.cpp#L27)<!-- prefix: "void append_" --> |
| HTTP controls | Dispatch cache reset, metrics, and bounded scheduler profiling. | [Control routes](../apps/server/prefix_controls.cpp#L157)<!-- prefix: "void Connect" --> |
| Batch | Restore request state, schedule remaining prompt rows, and publish completed checkpoints. | [Admission](../src/decoder_batch.cpp#L322)<!-- prefix: "int DecoderB" --> · [Checkpoint publication](../src/decoder_batch.cpp#L703)<!-- prefix: "cache-\u003estore" --> |
| Cache | Match checkpoint keys and own the copies until eviction or batch teardown. | [Lookup](../src/decoder_prefix_cache.cpp#L119)<!-- prefix: "int PrefixCa" --> · [Restore](../src/decoder_prefix_cache.cpp#L138)<!-- prefix: "int PrefixCa" --> · [Store](../src/decoder_prefix_cache.cpp#L161)<!-- prefix: "void PrefixC" --> |

## Runtime path

Use `--prefix-cache-entries 1` through `4` for native prefix reuse. The default `0` disables reuse. Keep `--mtp-steps 0` and omit `--eager`. The default prefill policy is `exact`. Pass `--fp8-prefill tensor-core` to select tensor-core prefill. An enabled cache selects the batch route even at `--max-active 1`. Batch graph opt-in requires capacity two or higher. [CLI validation](../apps/sglang-server.cpp#L107)<!-- prefix: "if (argument" --> · [Route selection](../apps/sglang-server.cpp#L220)<!-- prefix: "if (argument" -->

Native text, chat, and benchmark ID requests accept an optional `cache_salt` string of up to 256 UTF-8 bytes. Omission uses an empty string. Invalid salts return HTTP 400 even when reuse is disabled. Native text and chat support greedy, sampled, and finite-choice requests with prefix reuse. The benchmark ID route keeps its greedy policy.

C++ callers set `DecoderBatchOptions::prefix_cache_entries` and `DecoderOptions::prefix_cache_salt`. `DecoderBatch::admit` rejects salts longer than 256 bytes and returns the restored token count on success. Tensor-core batches with prefix reuse reject positive MTP depth at admission. [Batch options](../include/sglang/decoder_batch.h#L43)<!-- prefix: "struct Decod" --> · [MTP guard](../src/decoder_batch.cpp#L332)<!-- prefix: "if (state_-\u003e" --> · [Token-ID parser](../apps/server/completions.cpp#L27)<!-- prefix: "CompletionRe" -->

JSON usage and SSE usage count the full prompt in `prompt_tokens`. `prompt_tokens_details.cached_tokens` counts restored prompt IDs. It is zero on a miss, with the cache off, or for a zero output budget. [Token-ID usage](../apps/server/completions.cpp#L96)<!-- prefix: "std::string " -->

The checkpoint boundary is the largest multiple of 128 within the prompt length and 6,400-token limit. Prompts shorter than 128 tokens have no checkpoint. Lookup chooses the longest stored checkpoint within this boundary with matching prefix IDs and salt. A miss prefills from the start. A partial hit restores state into private request allocations and prefills the suffix. [Boundary](../src/decoder_prefix_cache.cpp#L115)<!-- prefix: "int PrefixCa" -->

On a full hit, eager scoring uses the restored hidden and residual rows at output index zero. Selection uses the current request's [grammar](text-serving.md#finite-choice-constraints) and [sampling options](text-serving.md#sampling), including its seed. Its first score stays outside the packed decode width and graph mask. Eligible decode peers can replay on the same step. Subsequent decode follows [batch graph eligibility](batch-graphs.md#runtime-path). [Full hit](../src/decoder_batch.cpp#L492)<!-- prefix: "if (lane.res" --> · [Token selection](../src/decoder_batch.cpp#L453)<!-- prefix: "void Decoder" -->

### Tensor-core scheduling

With tensor-core prefill and reuse enabled, both batch advance methods schedule complete 128-row page tiles and whole final tails. Restore ends at a page boundary. The planner ends a tile at the request's checkpoint before it schedules the suffix. It rotates priority across slots and fits whole tiles into a 128-row prompt budget per step. Ready decode lanes also advance. The tile boundaries for a request stay fixed when its peers change. Steps with prompt rows run eagerly. Exact and cache-off scheduling keep their existing rules. [Planner](../src/decoder_batch.cpp#L213)<!-- prefix: "PrefillRows " --> · [Dispatch](../src/decoder_batch.cpp#L666)<!-- prefix: "if (tensor_c" -->

## Constraints and failure behavior

An entry stores four MXFP8 K/V payload and scale planes for each of 16 attention layers. It stores BF16 convolution history and FP32 recurrent state for 48 GDN layers, plus BF16 hidden and residual rows. A full entry uses 50 logical pages. Each configured entry reserves 370,233,344 device bytes when the batch starts. Four entries reserve 1,480,933,376 bytes in addition to request allocations. [Allocations](../src/decoder_prefix_cache.cpp#L104)<!-- prefix: "PrefixCache:" -->

Store gathers the four KV planes into logical page order. Restore scatters them into the destination request's physical pages. The copy validates the covered pages and rejects aliases into the suffix. A lane owns its restored copy and can continue after eviction. A store publishes the checkpoint after its copies complete on the batch stream. A hit or repeated key refreshes recency. A new key evicts the least recently used entry when all entries are occupied. Cancellation can leave an already published checkpoint available. Zero-budget requests skip reuse and publication. In exact C++ batches, positive-depth MTP lanes also skip reuse and publication. [Page mapping](../src/decoder_prefix_cache.cpp#L17)<!-- prefix: "PageMapping " --> · [Completed copy](../src/decoder_prefix_cache.cpp#L183)<!-- prefix: "check(cudaSt" -->

An error while restoring a K0 lane disables the batch. A publication error disables the batch on a K0-only step. If a tensor-core cache step cannot allocate its multirow workspace, it fails before publishing a checkpoint from that step. The batch becomes unavailable. An invalid sampled or constrained selection retires only its lane. [Admission recovery](../src/decoder_batch.cpp#L371)<!-- prefix: "} catch (..." --> · [Workspace guard](../src/decoder_batch.cpp#L531)<!-- prefix: "if (tensor_c" --> · [Step recovery](../src/decoder_batch.cpp#L722)<!-- prefix: "if (!contain" -->

With `--trace-batches`, `entries` and `pages` count published state, and `device_bytes` counts the reservation. `hits`, `cached_tokens`, `prefill_tokens`, `stores`, and `evictions` are cumulative for the batch. `prefill_tokens` counts cache-eligible prompt rows processed on the batch path. Request memory and batch scratch have separate counters. [Trace](../apps/server/batch_service.cpp#L296)<!-- prefix: "void trace_p" --> · [Counters](../src/decoder_prefix_cache.cpp#L202)<!-- prefix: "DecoderPrefi" -->

### Native HTTP controls

These routes are available on the loopback listener with native prefix reuse and K0 under either prefill policy. Let `A` be `--max-active`. The listener has `A + 1` workers that accept and parse HTTP requests and serve controls. This count includes the serving thread. Readers share one nonblocking listener with a backlog setting of 16 and a five-second HTTP read deadline. [Dispatch](../apps/sglang-server.cpp#L236)<!-- prefix: "if (argument" --> · [Reader loop](../apps/server/prefix_controls.cpp#L219)<!-- prefix: "void Connect" --> · [Worker setup](../apps/server/prefix_controls.cpp#L274)<!-- prefix: "void serve_p" --> · [Read limits](../apps/server/http.cpp#L24)<!-- prefix: "constexpr st" -->

After HTTP parsing, completions enter a FIFO in admission order. Up to `A` wait and up to `A` hold a completion worker. The queue prunes disconnected clients before checking capacity and returns HTTP 503 when full. The `3A + 1` accepted socket bound includes readers, queued completions, and claimed completions. At 2 MiB per body, raw body payload can reach `2A` MiB in the FIFO and `2(3A + 1)` MiB overall. Headers, decoded requests, allocator overhead, and kernel buffers are separate. Shutdown closes the queue, discards waiters, and joins readers and completion workers before propagating the first escaped worker error. [Queue](../apps/server/prefix_controls.cpp#L200)<!-- prefix: "void Connect" --> · [Shutdown](../apps/server/prefix_controls.cpp#L314)<!-- prefix: "queue.stop()" -->

| Route | Input | Result |
| --- | --- | --- |
| `POST /reset_prefix_cache` | `{}` | Returns `{"success":true}` after the idle decoder worker invalidates all checkpoints. Returns HTTP 409 while a completion is queued or holds a claimed decoder slot. |
| `GET /metrics` | Empty body | Returns Prometheus text with native provenance and `engine="0"`. |
| `POST /start_profile` | Exactly three fields. `profile_prefix` is a string of 1–64 bytes. `max_iterations` is an integer from 1 to 64. `profiler_kwargs` equals `{"with_stack":false}`. | Opens one capture and returns `{"success":true,"profiler":"native-batch-steps"}`. Returns HTTP 409 if a capture is open. |
| `POST /stop_profile` | `{}` | Closes the capture and adds `profile_prefix` and `steps` to the JSON response. Returns HTTP 409 if no capture is open. |

Successful controls return HTTP 200. Malformed POST control bodies return HTTP 400. Reset clears published entries and pages. It preserves the allocation and cumulative counters. A claimed slot counts as running during completion parsing and token callbacks. On cancellation, the completion worker waits for decoder retirement before releasing its slot. [Reset](../apps/server/batch_service.cpp#L489)<!-- prefix: "bool BatchSe" --> · [Invalidation](../src/decoder_prefix_cache.cpp#L196)<!-- prefix: "void PrefixC" --> · [Profile input](../apps/server/prefix_controls.cpp#L88)<!-- prefix: "void start_p" -->

The `vllm:` names support the frozen client. `num_requests_running` counts occupied slots, and `num_requests_waiting` counts queued completions. `kv_cache_usage_perc` is published prefix pages divided by reserved prefix pages. `prefix_cache_hits_total` counts restored prompt IDs, and `external_prefix_cache_hits_total` is zero. The first published output adds the full prompt length to `prompt_tokens_total` and increments `request_prefill_kv_computed_tokens_count`. `request_prefill_kv_computed_tokens_sum` counts cache-eligible prompt rows. `generation_tokens_total` counts published output IDs. The `sglang_cpp:` series report entries, pages, reserved bytes, hit requests, stores, and evictions. [Series](../apps/server/prefix_controls.cpp#L45)<!-- prefix: "std::string " --> · [Publication](../apps/server/batch_service.cpp#L256)<!-- prefix: "void BatchSe" -->

A native capture records completed `advance_chunked` calls from nonempty iterations that start after profile start. Each sample contains `elapsed_ns` in host time, `width`, total `prefill_rows`, `decode_lanes`, and the `graph_slots` bit mask. The capture stays open after reaching its sample limit. [Step timing](../apps/server/batch_service.cpp#L344)<!-- prefix: "steps = batc" --> · [Sample recording](../apps/server/batch_service.cpp#L280)<!-- prefix: "void BatchSe" --> · [Response](../apps/server/prefix_controls.cpp#L107)<!-- prefix: "void stop_pr" -->

## Verification

Follow the [prefix cache CUDA and route checks](../tests/prefix/README.md). Cover identity and remapped page tables, transfer failures, all four KV planes, and the 6,400-token limit. Compare greedy cold, full-hit, and suffix-hit routes from the same binary with exact prefill. Check page boundaries, changed prefix IDs, changed salts, zero budget, cancellation, and eviction. Compare committed IDs and finish reasons. In usage, require the expected cached-token count and identical prompt and completion counts.

For tensor-core prefill, enable reuse for both cold and warm checks. Use a fresh cache for the cold check and collect its own output and device state. Compare that state with a remapped restore and suffix prefill under concurrency. Check logical KV payload and scales, GDN state, hidden and residual rows, full BF16 logits, and selected IDs. Include prompts just below, at, and above a page boundary. Inject workspace allocation failure and require no checkpoint publication from the failed step.

Repeat full and partial hits with seeded sampling and a second seed or sampling policy. Change the finite-choice grammar while reusing the same prefix and salt. Compare each warm result with its own cold request, including the first output ID. Check that a full hit stays out of the replay mask while eligible peers replay. Inject a selection failure and check that the failed lane retires and its peers finish.

For the HTTP controls, reset a populated idle cache and check zero entries and pages. Check that reserved bytes and cumulative counters stay constant. Try reset during a stream and require HTTP 409. Check malformed bodies, duplicate profile start, and stop without an open capture. After a profiled cohort, check the sample cap and the step fields.

Fill the completion queue and require HTTP 503 for the next request. Disconnect a queued client and check that another request can take its place. A CPU decoder double can verify these HTTP and scheduling checks. Use the native decoder on a reserved B200 for CUDA correctness, output IDs, and frozen timing acceptance.

### Live control regression

Expose only the reserved B200 as CUDA device 0. Start the native server with these options and tracing off.

```bash
"$SGLANG_SERVER" --checkpoint "$CHECKPOINT" --device 0 --port "$port" \
    --max-active 2 --prefix-cache-entries 2 --fp8-prefill exact \
    --mtp-steps 0 --batch-cuda-graphs
```

After health is ready, run the test from the repository root. Use a new evidence directory under an existing parent.

```bash
/data1/shared/dongwu.chen/.pixi/envs/default/bin/python -B \
    tests/prefix/live_controls.py --port "$port" --evidence "$run/live-controls"
```

Require exit zero and `passed.json`. The [live test](../tests/prefix/live_controls.py#L222)<!-- prefix: "def run(cont" --> checks malformed controls, idle and busy reset, and exact and partial hits. It also checks bounded profiles, cache eviction, queue overflow, canceled waiters, and survivor completion. Use the separate W3-prefix protocol for workload timing.

When changing reader concurrency, pause one reader before dispatch at `--max-active 1`. Require complete health and completion requests to return HTTP 200 before releasing it. Also stop the server with a reader paused and require shutdown to join it and discard its request. A CPU decoder double can exercise both cases.

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
