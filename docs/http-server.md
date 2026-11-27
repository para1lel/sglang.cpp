# Native HTTP transport

The loopback server uses one HTTP reader and one batch scheduler at every capacity. [Launch options](../README.md#serve-token-ids-over-http) · [Startup](../apps/sglang-server.cpp#L420)<!-- prefix: "serve_connec" -->

## Request targets and redirects

The reader splits the HTTP/1.1 origin-form target at the first literal `?`, then percent-decodes the path once. It preserves `+` and replaces invalid UTF-8. `%2F` becomes `/`, `%252F` becomes `%2F`, and `%3F` becomes path data. Fixed routes match case-sensitively and allow one final decoded line feed. Framing checks use the same path. [Target split](../apps/server/http.cpp#L143)<!-- prefix: "const auto s" --> · [Decoder](../apps/server/http.cpp#L353)<!-- prefix: "std::string " --> · [Route matching](../apps/server/http.cpp#L386)<!-- prefix: "std::string_" -->

The reader checks exact routes before trying a slash redirect. A slash variant that matches an enabled route returns an empty HTTP 307. Both server-info aliases redirect when their callback is installed. Metrics, reset, and profile paths require prefix reuse. `/v1/models/` retrieves the empty ID, and `/v1/models/name/` retrieves `name/`. [Dispatch](../apps/server/prefix_controls.cpp#L617)<!-- prefix: "void Connect" --> · [Enabled routes](../apps/server/prefix_controls.cpp#L580)<!-- prefix: "ConnectionQu" -->

[Ollama](serving/ollama.md) covers model discovery, text generation, chat, and route overrides.

`Location` uses `http`, the decoded slash-trimmed path, and the original nonempty query with URL quoting. Requests require one nonempty `Host`. An unsuitable authority falls back to `127.0.0.1` with the local port if available and different from 80. Native framing and media-type limits apply. The reader rejects `Transfer-Encoding`, `Content-Encoding`, and `Expect`. [Headers](../apps/server/http.cpp#L150)<!-- prefix: "std::size_t " --> · [Authority](../apps/server/http.cpp#L593)<!-- prefix: "static std::" --> · [Location](../apps/server/http.cpp#L623)<!-- prefix: "void write_r" -->

[Cache flush](prefix-cache.md#cache-flush) covers methods, timeouts, and queueing. [Anthropic Messages](serving/anthropic-messages.md) and [token counting](native-tokenization.md#count-anthropic-prompt-tokens) describe the Anthropic endpoints. [OpenAI Responses](serving/openai-responses.md) covers text, reasoning, and state. [Decisions and System One](serving/decisions.md) covers zero-output scoring. [`/v1/loads`](serving/loads.md) reports local running and waiting requests.

## Health and readiness

Health checks accept empty GET requests and bypass the completion queue.

| Route | Responsibility | Response |
| --- | --- | --- |
| `/health` | Reports liveness and scheduler limits. | HTTP 200 with `max_active` and `max_num_batched_tokens`, even after permanent worker failure. [Health response](../apps/sglang-server.cpp#L219)<!-- prefix: "std::string " --> |
| `/ready` | Checks for permanent decoder worker failure. | HTTP 200 with `{"ready":true}` or HTTP 503 with `{"ready":false,"reason":"worker_failed"}`. [Readiness response](../apps/server/http.cpp#L668)<!-- prefix: "void write_r" --> · [Worker flag](../apps/server/batch_service.cpp#L719)<!-- prefix: "bool BatchSe" --> |
| `/health_generate` | A dedicated HTTP worker probes generation progress. | HTTP 200 for progress before timeout. HTTP 503 for queue overload, probe failure, timeout, permanent worker failure, or shutdown. Both carry `Content-Length: 0` and `Connection: close`. [Dispatch](../apps/server/prefix_controls.cpp#L483)<!-- prefix: "void Connect" --> · [Reply](../apps/server/http.cpp#L677)<!-- prefix: "void write_g" --> |

Permanent worker failure also makes new completions return HTTP 500. Generation health timeout leaves readiness unchanged. [Batch failure](../apps/server/batch_service.cpp#L71)<!-- prefix: "void BatchSe" -->

The defaults are eight active requests per participating GPU and 32,768 aggregate tokens per step. Override them with `--max-active` (1–32) and `--max-num-batched-tokens` (1–32,768). Verify both at `/health`. [Default capacity](../apps/sglang-server.cpp#L38)<!-- prefix: "const int de" --> · [Default budget](../include/sglang/decoder.h#L47)<!-- prefix: "inline const" -->

The default server attempts to connect to an LMCache daemon on the same GPU. Daemon availability is independent of liveness and worker readiness. Check [server information](native-model-info.md#server-information) before a warm-cache run. [Start the daemon](prefix-cache.md#lmcache-daemon)

### Generation probe lifecycle

The HTTP reader starts a 20-second monotonic deadline when it dispatches the parsed request. The deadline includes time in the health FIFO. The FIFO holds up to `--max-active` waiting probes in addition to the one claimed by its worker. The reader removes disconnected waiters before testing capacity. The health worker owns the claimed socket until it attempts the reply. [Deadline](../apps/server/prefix_controls.cpp#L33)<!-- prefix: "constexpr au" --> · [Health admission](../apps/server/prefix_controls.cpp#L850)<!-- prefix: "void Connect" --> · [Health worker](../apps/server/prefix_controls.cpp#L874)<!-- prefix: "void Connect" -->

The scheduler can use its latest successful token publication if it occurred after the probe started and by its deadline. Otherwise it registers one waiter for subsequent publication. While user jobs occupy decoder lanes, they can satisfy the probe without reserving a lane for health. Admission, prefill without a token, cancellation, and failed steps do not satisfy the probe. Publication is independent of when a client receives the token. [Waiter and timestamp](../apps/server/batch_service.cpp#L721)<!-- prefix: "bool BatchSe" --> · [Token publication](../apps/server/batch_service.cpp#L296)<!-- prefix: "void BatchSe" -->

When all decoder lanes become idle and no prefix reset is pending, the scheduler can install an internal job in the first lane. It submits prompt IDs `[0]` and generates at most one greedy token with MTP disabled. The scheduler removes the waiter on return and marks any internal job cancelled. It retains a still-installed job until the decoder worker can retire it. An installed health job makes prefix reset return HTTP 409. User request and token counters exclude the probe. Lower-level cache counters and native step profiles can include its decoder work. [Idle eligibility](../apps/server/batch_service.cpp#L139)<!-- prefix: "bool BatchSe" --> · [Job creation](../apps/server/batch_service.cpp#L735)<!-- prefix: "auto job = s" --> · [Probe options](../include/sglang/decoder.h#L58)<!-- prefix: "struct Decod" --> · [Waiter cleanup](../apps/server/batch_service.cpp#L759)<!-- prefix: "job->cancell" --> · [Lane retirement](../apps/server/batch_service.cpp#L207)<!-- prefix: "if (cancelle" --> · [Reset exclusion](../apps/server/batch_service.cpp#L795)<!-- prefix: "bool BatchSe" -->

### Request, failure, and shutdown behavior

After HTTP parsing, a GET with a nonempty body returns a JSON HTTP 404 error. Other methods receive HTTP 405, `Allow: GET`, and a JSON method error. `HEAD` omits the body. These requests do not enter the health FIFO. [Input checks](../apps/server/prefix_controls.cpp#L850)<!-- prefix: "void Connect" --> · [Method response](../apps/server/http.cpp#L633)<!-- prefix: "void write_g" -->

The scheduler wakes the health worker on progress and checks for a disconnect at intervals of up to 10 milliseconds while waiting. A disconnected client can lose the reply. On shutdown, the health worker leaves the active wait and drains the queued health checks. It attempts an empty HTTP 503 on each remaining socket before the transport joins the workers. The 20-second deadline bounds the progress window. The HTTP read and response write have separate I/O limits. [Wait loop](../apps/server/batch_service.cpp#L742)<!-- prefix: "while (!stat" --> · [Disconnect detection](../apps/server/http.cpp#L445)<!-- prefix: "bool disconn" --> · [Queue stop](../apps/server/prefix_controls.cpp#L892)<!-- prefix: "void Connect" --> · [Worker join](../apps/server/prefix_controls.cpp#L956)<!-- prefix: "queue.stop()" -->

### Differences from pinned SGLang

Pinned SGLang uses detokenizer or scheduler responses and the configurable wall-clock `SGLANG_HEALTH_CHECK_TIMEOUT` (20 seconds by default), and offers a diagnostic bypass. Native uses successful token publication within a fixed monotonic deadline and schedules a synthetic job only while idle. [Pinned health route and timeout comparison](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/http-server.md#differences-from-pinned-sglang)

## Model discovery

The reader handles `GET /v1/models` and `GET /v1/models/{model}` outside the completion queue regardless of prefix reuse. [Model discovery](native-model-info.md#model-discovery) · [Checkpoint capabilities and parsers](native-model-info.md)

`GET /server_info` and its alias expose effective launch settings and live scheduler counters. [Server information](native-model-info.md#server-information)

Use [`POST /separate_reasoning`](serving/separate-reasoning.md) to split reasoning from supplied text.

## HTTP admission and shutdown

With `A = --max-active`, the loopback reader has TCP backlog 16 and retains up to `A + 1` incomplete sockets. It processes ready and expired reads before acceptance. The five-second acceptance-to-request deadline covers 8 KiB headers and a 2 MiB body. Clients may half-close after the complete request. At capacity the oldest unreadable socket becomes replaceable after 250 ms without a read. Replacement briefly takes one extra socket. [Reader](../apps/server/http.cpp#L479)<!-- prefix: "void read_co" --> · [Read limits](../apps/server/http.cpp#L30)<!-- prefix: "constexpr st" --> · [Socket replacement](../apps/server/http.cpp#L454)<!-- prefix: "static void " -->

Parsed completions, [native generation](native-generate.md) · [pointwise scoring](serving/pointwise-score.md) · [tokenization](native-tokenization.md) use one FIFO with `A` waiting slots and `A` claiming workers. Remove disconnected waiters before checking the queue. A full waiting queue returns JSON 503 even with an idle worker. `/generate` reserves IDs after that check. A claimed request keeps its worker until the response socket closes. [Queue admission](../apps/server/prefix_controls.cpp#L617)<!-- prefix: "void Connect" --> · [Worker](../apps/server/prefix_controls.cpp#L784)<!-- prefix: "void Connect" -->

A batch uses one HTTP slot. Admitting an item moves its batch behind other ready waiters. Scalar and one-item requests wait for their worker's lane. Tokenization stays available after permanent decoder failure. Failed writes and resets cancel unfinished jobs and require retirement before slot reuse. Ready peers can advance while one writer or cache read waits. The cache owns armed GET destinations and submitted STORE sources through transfer or release. An empty slot can refresh its job snapshot once before admission, without waiting, when the original uncached job fills the step budget. A busy lock, shutdown or cache reset skips the refresh. Both selections share the smaller lane and row caps. [Batch cancellation](../apps/server/batch_service.cpp#L463)<!-- prefix: "void BatchSe" --> · [row grants and arrival order](batch-graphs.md#runtime-path) · [Direct cache scheduling](prefix-cache.md#constraints-and-failure-behavior)

[`POST /abort_request`](native-abort.md) runs on the reader after HTTP parsing. It bypasses the completion queue and works when every generation slot is occupied. It shares the same incomplete-request capacity, replacement policy, and deadlines as other controls. A separate CPU worker can remove a fully aborted generation request from the FIFO and send its result without a generation slot. Partially aborted batches stay in the FIFO. [Control route](../apps/server/prefix_controls.cpp#L496)<!-- prefix: "if (abort_ &" --> · [Abort worker](../apps/server/prefix_controls.cpp#L820)<!-- prefix: "void Connect" -->

On shutdown, the reader closes incomplete sockets. The queue discards pending work, and the server interrupts cache GET waits before joining HTTP workers. The [transfer lease](prefix-cache.md#constraints-and-failure-behavior) protects destinations through shutdown. After the HTTP workers join, the transport rethrows the first error from its reader or workers, if any. During batch service destruction, the cache waits for submitted stores to finish and joins its own workers. [HTTP shutdown](../apps/server/prefix_controls.cpp#L956)<!-- prefix: "queue.stop()" --> · [Cache worker shutdown](../apps/server/batch_cache.cpp#L104)<!-- prefix: "for (auto& w" -->

## Completion streams

Benchmark, text, chat, and `/generate` streams open HTTP 200 only after successful admission and before the first token or successful terminal result. Batch streams need at least one admission and no observed item error. A fully aborted `/generate` FIFO request may stream without admission. Abort streams end with one `[DONE]`. Nonstreaming text, chat, and generation return JSON. [Single-request callback order](../apps/server/batch_service.cpp#L424)<!-- prefix: "const bool a" --> · [Batch callback order](../apps/server/batch_service.cpp#L628)<!-- prefix: "bool BatchSe" -->

Before headers, use JSON errors (400 invalid request, 503 overload, 413 or 503 decoder capacity, 500 unexpected failure). After headers, an exception attempts one SSE error and closes without `[DONE]`, even with zero output IDs. The HTTP status remains 200. [decoder capacity failures](batch-graphs.md#resident-memory-admission) · [JSON error mapping](../apps/server/prefix_controls.cpp#L178)<!-- prefix: "void handle_" --> · [Text and chat error event](../apps/server/text_completion.cpp#L202)<!-- prefix: "void complet" -->

When waiting, the completion worker can send `:\n\n` after 15 seconds without a successful header, data, or comment write. Item frames share the batch timer. Buffered tokens do not reset it. Single-request polling yields to a pending error or finish. Batch polling runs once before consuming results. A failed write cancels the request. Measure [TTFT](performance.md#measurement) at the first output token, ignoring framing and metadata. [Write timestamp](../apps/server/http.cpp#L693)<!-- prefix: "bool EventSt" --> · [Comment deadline](../apps/server/http.cpp#L712)<!-- prefix: "bool EventSt" -->

## Decode scheduling

Ready batches rotate after each admission. A blocked poll or cancellation retains refill priority for one poll interval. If still blocked, other ready batches can claim free lanes. The original batch resumes with the same ticket. Ordinary readiness waits for mailbox acknowledgement outside the scheduler mutex. Ready lanes get up to 1 ms of grace. The K0 graph grace ends at 25 ready lanes when all 32 live jobs qualify. Recheck selected lanes before dispatch. Later acknowledgements can form another step. [Mailbox](../include/sglang/server/detail/output_mailbox.h#L10)<!-- prefix: "struct Outpu" --> · [Early dispatch conditions](../apps/server/batch_service.cpp#L101)<!-- prefix: "bool BatchSe" --> · [graph eligibility rules](batch-graphs.md#runtime-path)

### Bounded decode output

`DecoderCallbacks::buffer_decode_output` defaults to false. The benchmark ID stream on `/v1/completions` and scalar per-token SSE on `/generate` set it. Native generation uses step delivery for text stops, constraints, scores, and indexed requests. OpenAI text and chat use ordinary readiness. [Callback flag](../include/sglang/decoder.h#L132)<!-- prefix: "bool buffer_" --> · [Benchmark stream](../apps/server/completions.cpp#L135)<!-- prefix: "void stream_" --> · [Native dispatch](../apps/server/native_generate.cpp#L408)<!-- prefix: "} else if (!" -->

Positive-MTP buffering requires scalar per-token opt-in and greedy text without media, constraints, forced eager execution, top logits, label scoring, or target-only options such as penalties, logit bias, and minimum output length. The writer must acknowledge the first whole step. [Effective options](../apps/server/batch_service.cpp#L860)<!-- prefix: "job->options" --> · [Eligibility](../apps/server/detail/batch_state.cpp#L177)<!-- prefix: "bool Job::ca" --> · [Target-only options](../src/runtime/execution/decoder_options.cpp#L18)<!-- prefix: "bool Decoder" -->

The FIFO holds two whole decoder steps, including the step being delivered. A full mailbox blocks only its lane. The writer frees each entry after accepting all its IDs. On stop or length, it drains the FIFO before reporting completion. A terminal checkpoint can delay completion while it waits for store capacity. After submission the request can finish while the store worker retains the allocation views. [Readiness](../apps/server/detail/batch_state.cpp#L172)<!-- prefix: "bool Job::re" --> · [Publication](../apps/server/batch_service.cpp#L262)<!-- prefix: "bool BatchSe" --> · [FIFO delivery](../apps/server/batch_service.cpp#L439)<!-- prefix: "if (job.has_" --> · [Checkpoint submission](../src/runtime/execution/decoder_batch_prefix.cpp#L97)<!-- prefix: "bool Decoder" -->

At server depth zero, the separate cohort rule requires CUDA graphs, greedy prompts of at least 32,768 IDs, and first-output acknowledgement from every live job. Unindexed step-delivered `/generate` streams can opt in at K0. With 16–31 eligible lanes, all mailboxes having room ends grace early. After grace, any with room can advance. A full 32-lane cohort can dispatch at 25 ready mailboxes. Cancelled, aborted, or otherwise ineligible jobs prevent buffered dispatch. Under ordinary readiness, each lane drains its mailbox before decoding again. [Cohort guards](../apps/server/batch_service.cpp#L101)<!-- prefix: "bool BatchSe" --> · [Step-stream enrollment](../apps/server/batch_service.cpp#L520)<!-- prefix: "job->buffer_" -->

A failed write or disconnect cancels the request and waits for decoder retirement before slot reuse. Scalar writers drain deliverable output before reporting a decoder error. If native abort wins the state transition during a worker fault, an acknowledged, eligible positive-MTP lane still owned by the worker yields an abort result. Peers receive the error, and a permanent fault makes the worker unhealthy. [Disconnect retirement](../apps/server/batch_service.cpp#L408)<!-- prefix: "void BatchSe" --> · [Fault handling](../apps/server/batch_service.cpp#L71)<!-- prefix: "void BatchSe" --> · [Abort delivery](native-abort.md#runtime-path)

## Verification

Run the [CPU reader and queue checks](../tests/prefix/README.md#cpu-reader-queue) with prefix reuse off and on. Check framing, capacity, deadlines, replacement, cancellation, and shutdown. `/health` and `/ready` stay responsive under load and during reset acknowledgement. Reset rejects a claimed completion with 409. The [CPU placement probe](../tests/batch/run_indexed_admission.sh#L1)<!-- prefix: "#!/usr/bin/e" --> must show an indexed singleton waiting for its owner lane despite a free peer. Run from the repository root in managed Linux Pixi with `artifacts/` present. The grace check uses GNU linker wrapping and libstdc++.

```bash
bash tests/batch/run_service_mtp.sh "$PWD" "$PWD/artifacts/service-mtp-check"
bash tests/batch/run_service_decode_grace.sh "$PWD" "$PWD/artifacts/service-decode-grace-check"
bash tests/batch/run_service_mtp_mailbox.sh "$PWD" "$PWD/artifacts/service-mtp-mailbox"
```

Require exactly one admission callback before token delivery. A spurious wake and 24 acknowledgements stay in grace. At 25, dispatch before timeout, then dispatch the other seven separately. Check two ordered IDs and a length finish per item, reset rejection during streaming, and successful idle reset. The mailbox fixture covers first acknowledgement, the two-step bound, ready peers, stop and length, slot reuse, eligibility, late arrivals, blocked writers, and trace order. Check phase-dependent abort, failure after a completion claim, and recoverable and permanent faults for peers.

Use `curl -i --max-time 30 http://127.0.0.1:17384/health_generate`. Require an empty HTTP 200 when idle and after the next user token on busy lanes, with no synthetic admission. Deadline, FIFO overflow, and shutdown of writable probes return empty 503s. Check responsiveness while waiting, rerun after disconnect, and verify all three [health responses](#health-and-readiness) after a permanent worker failure.

On a reserved B200, run the [streaming checks](development.md#streaming-verification) and [traced prefill reset](../tests/prefill/README.md). Require early headers and reset after one traced chunk, retirement to zero request bytes, and successful replacement. With and without buffered decoder steps, require an SSE comment after 15 seconds of wire silence and measure TTFT at the first output ID frame. Check JSON rejection before headers and HTTP 200 with one SSE error and no `[DONE]` after headers but before the first token.

[Full verification appendix](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/docs/http-server.md#verification)
