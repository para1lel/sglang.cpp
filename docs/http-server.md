# Native HTTP transport

The loopback server uses one HTTP reader and one batch scheduler at every capacity. [Launch options](../README.md#serve-token-ids-over-http) · [Startup](../apps/sglang-server.cpp#L420)<!-- prefix: "serve_connec" -->

## Request targets and redirects

The reader splits the HTTP/1.1 origin-form target at the first literal `?`, then percent-decodes the path once. It preserves `+` and replaces invalid UTF-8. `%2F` becomes `/`, `%252F` becomes `%2F`, and `%3F` becomes path data. Fixed routes match case-sensitively and allow one final decoded line feed. Framing checks use the same path. [Target split](../apps/server/http.cpp#L143)<!-- prefix: "const auto s" --> · [Decoder](../apps/server/http.cpp#L353)<!-- prefix: "std::string " --> · [Route matching](../apps/server/http.cpp#L386)<!-- prefix: "std::string_" -->

The reader checks exact routes before trying a slash redirect. A slash variant that matches an enabled route returns an empty HTTP 307. Both server-info aliases redirect when their callback is installed. Metrics, reset, and profile paths require prefix reuse. `/v1/models/` retrieves the empty ID, and `/v1/models/name/` retrieves `name/`. [Dispatch](../apps/server/prefix_controls.cpp#L617)<!-- prefix: "void Connect" --> · [Enabled routes](../apps/server/prefix_controls.cpp#L580)<!-- prefix: "ConnectionQu" -->

[Ollama](serving/ollama.md) covers model discovery, text generation, chat, and route overrides.

Redirect `Location` uses `http`, the decoded slash-trimmed path, and the original nonempty query with URL quoting. Each request needs one nonempty `Host`. An unsuitable authority falls back to `127.0.0.1` with the local port if available and different from 80. The reader rejects `Transfer-Encoding`, `Content-Encoding`, and `Expect`. For `POST /v1/completions` and both `/generate` methods, require `Content-Length` and `Content-Type: application/json` (case-insensitive, without parameters). [Headers](../apps/server/http.cpp#L150)<!-- prefix: "std::size_t " --> · [Authority](../apps/server/http.cpp#L593)<!-- prefix: "static std::" --> · [Location](../apps/server/http.cpp#L623)<!-- prefix: "void write_r" -->

[Cache flush](prefix-cache.md#cache-flush) covers methods, timeouts, and queueing. [Anthropic Messages](serving/anthropic-messages.md) and [token counting](native-tokenization.md#count-anthropic-prompt-tokens) describe the Anthropic endpoints. [OpenAI Responses](serving/openai-responses.md) covers text, reasoning, and state. [Decisions and System One](serving/decisions.md) covers zero-output scoring. [`/v1/loads`](serving/loads.md) reports local running and waiting requests.

## Health and readiness

Health checks use empty GETs outside the completion queue.

| Route | Response |
| --- | --- |
| `/health` | 200 with `max_active` and `max_num_batched_tokens`, even after permanent decoder failure. |
| `/ready` | 200 with `{"ready":true}` or 503 with `{"ready":false,"reason":"worker_failed"}`. |
| `/health_generate` | A dedicated worker returns 200 on progress, or 503 on overload, probe failure, timeout, permanent worker failure, or shutdown. Both replies have `Content-Length: 0` and `Connection: close`. |

Permanent worker failure also makes new completions return 500. A generation probe timeout leaves readiness unchanged. Defaults are eight active requests per participating GPU and 32,768 tokens per step. Set `--max-active` (1–32) and `--max-num-batched-tokens` (1–32,768), then verify them at `/health`. By default the server attempts a same-GPU LMCache connection. Daemon availability is independent. Check [server information](native-model-info.md#server-information) before warming the [daemon](prefix-cache.md#lmcache-daemon). [Health response](../apps/sglang-server.cpp#L219)<!-- prefix: "std::string " --> · [Readiness response](../apps/server/http.cpp#L668)<!-- prefix: "void write_r" --> · [Worker flag](../apps/server/batch_service.cpp#L719)<!-- prefix: "bool BatchSe" --> · [Dispatch](../apps/server/prefix_controls.cpp#L483)<!-- prefix: "void Connect" --> · [Reply](../apps/server/http.cpp#L677)<!-- prefix: "void write_g" --> · [Batch failure](../apps/server/batch_service.cpp#L71)<!-- prefix: "void BatchSe" --> · [Default capacity](../apps/sglang-server.cpp#L38)<!-- prefix: "const int de" --> · [Default budget](../include/sglang/decoder.h#L47)<!-- prefix: "inline const" -->

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

With `A = --max-active`, TCP backlog is 16 and the reader retains up to `A+1` incomplete requests. It services ready and expired reads before acceptance. The five-second deadline from acceptance covers 8 KiB of headers (431) and 2 MiB of body (413). At capacity the oldest unreadable socket can be replaced after 250 ms without a read, briefly taking one extra socket. A client can half-close after sending a complete request.

Completions, [pointwise scoring](serving/pointwise-score.md), and [tokenization](native-tokenization.md) share one FIFO with `A` waiting slots and `A` workers. Drop disconnected waiters before admitting. A full FIFO returns JSON 503 even with an idle worker. [Native generation](native-generate.md) reserves IDs after admission. One request or batch holds its HTTP worker until socket closure. Multi-item batches rotate behind ready waiters after each admission. Scalars and one-item batches wait for the owning worker's lane. Tokenization survives permanent decoder failure. A failed write or reset cancels unfinished work and requires retirement before slot reuse. Peers can advance while a writer or cache read waits. See [row grants](batch-graphs.md#runtime-path) and [cache resource ownership](prefix-cache.md#constraints-and-failure-behavior).

If the original uncached job fills the step budget, an empty slot can refresh its job snapshot once before admission without waiting. A busy lock, shutdown, or cache reset skips refresh. Both selections share the smaller row and lane caps. [Reader](../apps/server/http.cpp#L479)<!-- prefix: "void read_co" --> · [Read limits](../apps/server/http.cpp#L30)<!-- prefix: "constexpr st" --> · [Socket replacement](../apps/server/http.cpp#L454)<!-- prefix: "static void " --> · [Queue admission](../apps/server/prefix_controls.cpp#L617)<!-- prefix: "void Connect" --> · [Worker](../apps/server/prefix_controls.cpp#L784)<!-- prefix: "void Connect" --> · [Batch cancellation](../apps/server/batch_service.cpp#L463)<!-- prefix: "void BatchSe" -->

[`POST /abort_request`](native-abort.md) runs on the reader after HTTP parsing. It bypasses the completion queue and works when every generation slot is occupied. It shares the same incomplete-request capacity, replacement policy, and deadlines as other controls. A separate CPU worker can remove a fully aborted generation request from the FIFO and send its result without a generation slot. Partially aborted batches stay in the FIFO. [Control route](../apps/server/prefix_controls.cpp#L496)<!-- prefix: "if (abort_ &" --> · [Abort worker](../apps/server/prefix_controls.cpp#L820)<!-- prefix: "void Connect" -->

On shutdown, the reader closes incomplete sockets. The queue discards pending work, and the server interrupts cache GET waits before joining HTTP workers. The [transfer lease](prefix-cache.md#constraints-and-failure-behavior) protects destinations through shutdown. After the HTTP workers join, the transport rethrows the first error from its reader or workers, if any. During batch service destruction, the cache waits for submitted stores to finish and joins its own workers. [HTTP shutdown](../apps/server/prefix_controls.cpp#L956)<!-- prefix: "queue.stop()" --> · [Cache worker shutdown](../apps/server/batch_cache.cpp#L103)<!-- prefix: "for (auto& w" -->

## Completion streams

Benchmark, text, chat, and `/generate` open SSE 200 after successful admission and before the first token or successful terminal result. Batches need at least one admission and no observed item error. A fully aborted `/generate` request can stream without admission. Successful or aborted streams end with one `[DONE]`. Text, chat, and native generation without streaming return JSON.

For these completion routes, pre-header errors return JSON with `error.message`. Invalid requests and duplicate live IDs return 400, [oversized inputs](#http-admission-and-shutdown) 413 or 431, and [decoder capacity failures](batch-graphs.md#resident-memory-admission) 413 or 503. Overload returns 503 and unexpected failure returns 500. After headers the status stays 200, even before the first output ID. The worker attempts one SSE error event and closes without `[DONE]`.

After 15 seconds without a successful header, data, or comment write, the worker can send `:\n\n`. All batch frames update one timer. Buffered output leaves it unchanged. Single-request polling yields to pending errors and finishes. Batch polling runs once before reading results. Failed writes cancel the request. Measure [TTFT](performance.md#measurement) at the first output ID for each item. Zero-output items have no sample. [Single-request callback order](../apps/server/batch_service.cpp#L424)<!-- prefix: "const bool a" --> · [Batch callback order](../apps/server/batch_service.cpp#L628)<!-- prefix: "bool BatchSe" --> · [JSON error mapping](../apps/server/prefix_controls.cpp#L178)<!-- prefix: "void handle_" --> · [Text and chat error event](../apps/server/text_completion.cpp#L200)<!-- prefix: "void complet" --> · [Write timestamp](../apps/server/http.cpp#L693)<!-- prefix: "bool EventSt" --> · [Comment deadline](../apps/server/http.cpp#L712)<!-- prefix: "bool EventSt" -->

## Decode scheduling

After a blocked poll or cancellation, a batch holds refill priority for one poll interval. If still blocked, peers can claim free lanes. The original ticket stays valid. Ordinary readiness waits for mailbox acknowledgement outside the scheduler mutex. Ready lanes receive up to 1 ms of grace. Recheck selected lanes before dispatch. Later acknowledgements can form a separate step. [K0 cohort threshold](#bounded-decode-output) · [Graph eligibility](batch-graphs.md#runtime-path) [Mailbox](../include/sglang/server/detail/output_mailbox.h#L10)<!-- prefix: "struct Outpu" --> · [Early dispatch conditions](../apps/server/batch_service.cpp#L101)<!-- prefix: "bool BatchSe" -->

### Bounded decode output

`DecoderCallbacks::buffer_decode_output` defaults to false. The benchmark ID stream and eligible scalar `/generate` SSE opt in. OpenAI text, chat, and indexed `/generate` use ordinary readiness. See [native delivery modes](native-generate.md#text-stops).

Positive MTP requires a per-token writer and greedy text without media, constraints, eager mode, top logits, label scoring, or [target-only options](mtp.md#sampled-mtp). The first whole step must be acknowledged. The FIFO holds two whole steps including the one being delivered. A full mailbox blocks its lane. The writer releases each entry after accepting all IDs and drains before reporting stop or length. A terminal checkpoint can wait for [store capacity](prefix-cache.md#constraints-and-failure-behavior). After submission, the STORE worker retains the views.

K0 can also buffer unindexed native step streams. It requires CUDA graphs, greedy prompts of at least 32,768 IDs, and first-output acknowledgement from every live job. At 16–31 eligible lanes, room in all mailboxes ends grace early. With a full 32-lane cohort, 25 ready lanes suffice. After grace, any eligible lane with room can advance. Cancelled, aborted, or ineligible jobs prevent buffered dispatch. Ordinary readiness waits for a drained mailbox.

Scalar writers drain deliverable output before reporting a decoder error. If native abort wins during a fault, an acknowledged eligible positive-MTP lane still owned by the worker yields an [abort result](native-abort.md#runtime-path). Peers receive the error and permanent faults make the worker unhealthy. [Callback flag](../include/sglang/decoder.h#L132)<!-- prefix: "bool buffer_" --> · [Benchmark stream](../apps/server/completions.cpp#L135)<!-- prefix: "void stream_" --> · [Native dispatch](../apps/server/native_generate.cpp#L405)<!-- prefix: "} else if (!" --> · [Effective options](../apps/server/batch_service.cpp#L860)<!-- prefix: "job->options" --> · [Eligibility](../apps/server/detail/batch_state.cpp#L177)<!-- prefix: "bool Job::ca" --> · [Target-only options](../src/runtime/execution/decoder_options.cpp#L18)<!-- prefix: "bool Decoder" --> · [Readiness](../apps/server/detail/batch_state.cpp#L172)<!-- prefix: "bool Job::re" --> · [Publication](../apps/server/batch_service.cpp#L262)<!-- prefix: "bool BatchSe" --> · [FIFO delivery](../apps/server/batch_service.cpp#L439)<!-- prefix: "if (job.has_" --> · [Checkpoint submission](../src/runtime/execution/decoder_batch_prefix.cpp#L97)<!-- prefix: "bool Decoder" --> · [Cohort guards](../apps/server/batch_service.cpp#L101)<!-- prefix: "bool BatchSe" --> · [Step-stream enrollment](../apps/server/batch_service.cpp#L520)<!-- prefix: "job->buffer_" --> · [Disconnect retirement](../apps/server/batch_service.cpp#L408)<!-- prefix: "void BatchSe" --> · [Fault handling](../apps/server/batch_service.cpp#L71)<!-- prefix: "void BatchSe" -->

## Verification

Run the [CPU reader and queue check](../tests/prefix/README.md#cpu-reader-queue) with reuse off and on for framing, capacity, deadlines, replacement, cancellation, and shutdown. Health and readiness stay responsive during load and a held reset acknowledgement. Reset returns 409 while a completion holds its worker. The [CPU placement probe](../tests/batch/run_indexed_admission.sh#L1)<!-- prefix: "#!/usr/bin/e" --> verifies an indexed singleton waits for its owning lane despite a free peer.

Run from the repository root in managed Linux Pixi with `artifacts/` present. The grace check needs GNU linker wrapping and libstdc++.

```bash
bash tests/batch/run_service_mtp.sh "$PWD" "$PWD/artifacts/service-mtp-check"
bash tests/batch/run_service_decode_grace.sh "$PWD" "$PWD/artifacts/service-decode-grace-check"
bash tests/batch/run_service_mtp_mailbox.sh "$PWD" "$PWD/artifacts/service-mtp-mailbox"
```

Require exactly one admission callback before tokens. A spurious wake and 24 acknowledgements stay in grace. At 25, dispatch before timeout, then dispatch the remaining seven separately. Require two ordered IDs and length per item, reset rejection while streaming and success at idle. Check the [mailbox rules](#bounded-decode-output), late arrivals, blocked writes, trace order, phase-dependent abort, post-claim failures, and peer outcomes after recoverable and permanent faults.

Run `curl -i --max-time 30 http://127.0.0.1:17384/health_generate`. Require empty 200 at idle and on the next busy user token without synthetic admission. Require empty 503 on timeout, FIFO overflow, and shutdown of writable probes. Check other routes respond while a probe waits. Retry after disconnect and check all [health responses](#health-and-readiness) after permanent failure.

On a reserved B200, run [streaming](development.md#streaming-verification) and [traced prefill reset](../tests/prefill/README.md). Require headers before tokens, reset after one traced chunk, zero request bytes after retirement, and successful replacement. With and without bounded output, check the [15-second comment, TTFT, and pre/post-header failure rules](#completion-streams), including an error after headers but before tokens.

[Full verification appendix](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/docs/http-server.md#verification)
