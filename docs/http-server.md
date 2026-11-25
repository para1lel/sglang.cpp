# Native HTTP transport

The loopback server uses one HTTP reader and one batch scheduler at every capacity. [Launch options](../README.md#serve-token-ids-over-http) · [Startup](../apps/sglang-server.cpp#L446)<!-- prefix: "serve_connec" -->

## Request targets and redirects

The reader splits the HTTP/1.1 origin-form target at the first literal `?`, then percent-decodes the path once. It preserves `+` and replaces invalid UTF-8. `%2F` becomes `/`, `%252F` becomes `%2F`, and `%3F` becomes path data. Fixed routes match case-sensitively and allow one final decoded line feed. Framing checks use the same path. [Target split](../apps/server/http.cpp#L143)<!-- prefix: "const auto s" --> · [Decoder](../apps/server/http.cpp#L346)<!-- prefix: "std::string " --> · [Route matching](../apps/server/http.cpp#L372)<!-- prefix: "std::string_" -->

The reader checks exact routes before trying a slash redirect. A slash variant that matches an enabled route returns an empty HTTP 307. Both server-info aliases redirect when their callback is installed. Metrics, reset, and profile paths require prefix reuse. `/v1/models/` retrieves the empty ID, and `/v1/models/name/` retrieves `name/`. [Dispatch](../apps/server/prefix_controls.cpp#L617)<!-- prefix: "void Connect" --> · [Enabled routes](../apps/server/prefix_controls.cpp#L580)<!-- prefix: "ConnectionQu" -->

[Ollama](serving/ollama.md) covers model discovery, text generation, chat, and route overrides.

`Location` uses `http`, the decoded slash-trimmed path, and the original nonempty query with URL quoting. Requests require one nonempty `Host`. An unsuitable authority falls back to `127.0.0.1` with the local port if available and different from 80. Native framing and media-type limits apply. The reader rejects `Transfer-Encoding`, `Content-Encoding`, and `Expect`. [Headers](../apps/server/http.cpp#L150)<!-- prefix: "std::size_t " --> · [Authority](../apps/server/http.cpp#L587)<!-- prefix: "static std::" --> · [Location](../apps/server/http.cpp#L617)<!-- prefix: "void write_r" -->

[Cache flush](prefix-cache.md#cache-flush) covers methods, timeouts, and queueing. [Anthropic Messages](serving/anthropic-messages.md) and [token counting](native-tokenization.md#count-anthropic-prompt-tokens) describe the Anthropic endpoints. [OpenAI Responses](serving/openai-responses.md) covers text, reasoning, and state. [Decisions and System One](serving/decisions.md) covers zero-output scoring. [`/v1/loads`](serving/loads.md) reports local running and waiting requests.

## Health and readiness

Health checks accept empty GET requests and bypass the completion queue.

| Route | Responsibility | Response |
| --- | --- | --- |
| `/health` | Reports liveness and scheduler limits. | HTTP 200 with `max_active` and `max_num_batched_tokens`, even after permanent worker failure. [Health response](../apps/sglang-server.cpp#L248)<!-- prefix: "std::string " --> |
| `/ready` | Checks for permanent decoder worker failure. | HTTP 200 with `{"ready":true}` or HTTP 503 with `{"ready":false,"reason":"worker_failed"}`. [Readiness response](../apps/server/http.cpp#L662)<!-- prefix: "void write_r" --> · [Worker flag](../apps/server/batch_service.cpp#L720)<!-- prefix: "bool BatchSe" --> |
| `/health_generate` | A dedicated HTTP worker probes generation progress. | HTTP 200 for progress before timeout. HTTP 503 for queue overload, probe failure, timeout, permanent worker failure, or shutdown. Both carry `Content-Length: 0` and `Connection: close`. [Dispatch](../apps/server/prefix_controls.cpp#L483)<!-- prefix: "void Connect" --> · [Reply](../apps/server/http.cpp#L671)<!-- prefix: "void write_g" --> |

Permanent worker failure also makes new completions return HTTP 500. Generation health timeout leaves readiness unchanged. [Batch failure](../apps/server/batch_service.cpp#L71)<!-- prefix: "void BatchSe" -->

The defaults are eight active requests per participating GPU and 32,768 aggregate tokens per step. Override them with `--max-active` (1–32) and `--max-num-batched-tokens` (1–32,768). Verify both at `/health`. [Default capacity](../apps/sglang-server.cpp#L39)<!-- prefix: "constexpr in" --> · [Default budget](../include/sglang/decoder.h#L47)<!-- prefix: "inline const" -->

The default server attempts to connect to an LMCache daemon on the same GPU. Daemon availability is independent of liveness and worker readiness. Check [server information](native-model-info.md#server-information) before a warm-cache run. [Start the daemon](prefix-cache.md#lmcache-daemon)

### Generation probe lifecycle

The HTTP reader starts a 20-second monotonic deadline when it dispatches the parsed request. The deadline includes time in the health FIFO. The FIFO holds up to `--max-active` waiting probes in addition to the one claimed by its worker. The reader removes disconnected waiters before testing capacity. The health worker owns the claimed socket until it attempts the reply. [Deadline](../apps/server/prefix_controls.cpp#L33)<!-- prefix: "constexpr au" --> · [Health admission](../apps/server/prefix_controls.cpp#L851)<!-- prefix: "void Connect" --> · [Health worker](../apps/server/prefix_controls.cpp#L875)<!-- prefix: "void Connect" -->

The scheduler can use its latest successful token publication if it occurred after the probe started and by its deadline. Otherwise it registers one waiter for subsequent publication. While user jobs occupy decoder lanes, they can satisfy the probe without reserving a lane for health. Admission, prefill without a token, cancellation, and failed steps do not satisfy the probe. Publication is independent of when a client receives the token. [Waiter and timestamp](../apps/server/batch_service.cpp#L722)<!-- prefix: "bool BatchSe" --> · [Token publication](../apps/server/batch_service.cpp#L296)<!-- prefix: "void BatchSe" -->

When all decoder lanes become idle and no prefix reset is pending, the scheduler can install an internal job in the first lane. It submits prompt IDs `[0]` and generates at most one greedy token with MTP disabled. The scheduler removes the waiter on return and marks any internal job cancelled. It retains a still-installed job until the decoder worker can retire it. An installed health job makes prefix reset return HTTP 409. User request and token counters exclude the probe. Lower-level cache counters and native step profiles can include its decoder work. [Idle eligibility](../apps/server/batch_service.cpp#L139)<!-- prefix: "bool BatchSe" --> · [Job creation](../apps/server/batch_service.cpp#L736)<!-- prefix: "auto job = s" --> · [Probe options](../include/sglang/decoder.h#L58)<!-- prefix: "struct Decod" --> · [Waiter cleanup](../apps/server/batch_service.cpp#L760)<!-- prefix: "job->cancell" --> · [Lane retirement](../apps/server/batch_service.cpp#L207)<!-- prefix: "if (cancelle" --> · [Reset exclusion](../apps/server/batch_service.cpp#L791)<!-- prefix: "bool BatchSe" -->

### Request, failure, and shutdown behavior

After HTTP parsing, a GET with a nonempty body returns a JSON HTTP 404 error. Other methods receive HTTP 405, `Allow: GET`, and a JSON method error. `HEAD` omits the body. These requests do not enter the health FIFO. [Input checks](../apps/server/prefix_controls.cpp#L851)<!-- prefix: "void Connect" --> · [Method response](../apps/server/http.cpp#L627)<!-- prefix: "void write_g" -->

The scheduler wakes the health worker on progress and checks for a disconnect at intervals of up to 10 milliseconds while waiting. A disconnected client can lose the reply. On shutdown, the health worker leaves the active wait and drains the queued health checks. It attempts an empty HTTP 503 on each remaining socket before the transport joins the workers. The 20-second deadline bounds the progress window. The HTTP read and response write have separate I/O limits. [Wait loop](../apps/server/batch_service.cpp#L743)<!-- prefix: "while (!stat" --> · [Disconnect detection](../apps/server/http.cpp#L438)<!-- prefix: "bool disconn" --> · [Queue stop](../apps/server/prefix_controls.cpp#L893)<!-- prefix: "void Connect" --> · [Worker join](../apps/server/prefix_controls.cpp#L972)<!-- prefix: "queue.stop()" -->

### Differences from pinned SGLang

Both generation checks use empty HTTP 200 or 503 replies and a maximum of one greedy output token from `[0]`. Pinned SGLang creates a probe task per request and tests for any newer detokenizer or scheduler response. It checks wall-clock time after creating the task, polls once per second, and defaults to 20 seconds through `SGLANG_HEALTH_CHECK_TIMEOUT`. The native route uses a fixed 20 seconds from HTTP dispatch, scheduler notification on successful token publication, and a 10-millisecond disconnect poll. It has one active health worker and `--max-active` waiting slots. It installs a synthetic job only when the scheduler is idle. [Pinned SGLang timeout](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L228) · [Pinned SGLang probe](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L765-L843)

Pinned SGLang can enable generation on `/health` and can bypass generation on both health paths with its diagnostic setting. It sets its server status to `Up` on probe success and `UnHealthy` on timeout. Its `/ready` returns an empty body from the tokenizer manager's readiness check. The native `/health` always returns the limits, and `/ready` reports the worker flag with JSON. The native generation check has no diagnostic bypass. [Pinned SGLang readiness and probe](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L758-L843)

## Model discovery

The reader handles `GET /v1/models` and `GET /v1/models/{model}` outside the completion queue regardless of prefix reuse. [Model discovery](native-model-info.md#model-discovery) · [Checkpoint capabilities and parsers](native-model-info.md)

`GET /server_info` and its alias expose effective launch settings and live scheduler counters. [Server information](native-model-info.md#server-information)

Use [`POST /separate_reasoning`](serving/separate-reasoning.md) to split reasoning from supplied text.

## HTTP admission and shutdown

Let `A` be `--max-active`. The nonblocking loopback listener uses a TCP backlog setting of 16. The reader polls all retained sockets and advances each request incrementally. It retains up to `A + 1` incomplete requests. The reader services ready and expired reads before accepting a waiting connection. The five-second read deadline starts at acceptance and covers receipt of the HTTP headers and body. The header and body limits are 8 KiB and 2 MiB, respectively. Clients can use `shutdown(SHUT_WR)` after a complete request. [Listener](../apps/server/http.cpp#L405)<!-- prefix: "Socket liste" --> · [Reader](../apps/server/http.cpp#L472)<!-- prefix: "void read_co" --> · [Read limits](../apps/server/http.cpp#L30)<!-- prefix: "constexpr st" -->

When the reader is full, it defers acceptance unless an incomplete socket has no readable data and has been idle at least 250 milliseconds since acceptance or its last successful read. The reader accepts the new connection, then closes the oldest eligible socket. The count can briefly exceed the limit by one. [Socket replacement](../apps/server/http.cpp#L447)<!-- prefix: "static void " -->

After HTTP parsing, completions, [native generation](native-generate.md), [pointwise scoring](serving/pointwise-score.md), and [tokenization](native-tokenization.md) enter one FIFO. Up to `A` requests can wait, and up to `A` workers can claim requests. The queue removes disconnected waiters before checking capacity. It returns a JSON HTTP 503 response when `A` requests are still waiting, even if some workers are idle. The reader checks the shape of `/generate` and reserves its IDs after this capacity check. Workers validate the prompts and options. A normally claimed request holds its HTTP worker slot until its response socket closes. [Queue admission](../apps/server/prefix_controls.cpp#L617)<!-- prefix: "void Connect" --> · [Worker](../apps/server/prefix_controls.cpp#L784)<!-- prefix: "void Connect" -->

A [text completion batch](text-serving.md#send-a-request) or [native generation batch](native-generate.md#runtime-path) counts as one request for HTTP admission. Multi-item batches share eligible decoder lanes and give the longest-waiting ready batch first use of a free lane. Admitting an item moves that batch behind other waiters. Scalar requests and one-item batches reserve their worker's lane. Tokenization uses an HTTP worker and runs on the CPU, including after permanent decoder worker failure. For generation, a TCP reset or failed write cancels unfinished items. The batch retires active lanes before slot reuse, and ready peers can advance while a writer blocks. [Single-request cancellation](../apps/server/batch_service.cpp#L409)<!-- prefix: "void BatchSe" --> · [Batch cancellation](../apps/server/batch_service.cpp#L464)<!-- prefix: "void BatchSe" -->

Before the first long prefill grant, the worker can refresh its job snapshot once without waiting. It needs an empty slot, no previously admitted job, and an original job with a full uncached step budget. Original selections and newly published jobs share the smaller of the active-lane limit and row budget. A busy lock, shutdown, or cache reset skips the refresh. [Refresh trigger](../apps/server/batch_service.cpp#L362)<!-- prefix: "if (cold && " --> · [Roster refresh](../apps/server/batch_service.cpp#L180)<!-- prefix: "std::uint32_" --> See [row grants and arrival order](batch-graphs.md#runtime-path).

A lookup delays its own admission and can be abandoned on abort or disconnect. An armed GET holds its destination slot and allocations until transfer or lease release. A submitted STORE keeps its source allocations and releases the producing lane to continue or retire. Ready peers can advance in either case. [Direct cache scheduling](prefix-cache.md#constraints-and-failure-behavior)

[`POST /abort_request`](native-abort.md) runs on the reader after HTTP parsing. It bypasses the completion queue and works when every generation slot is occupied. It shares the same incomplete-request capacity, replacement policy, and deadlines as other controls. A separate CPU worker can remove a fully aborted generation request from the FIFO and send its result without a generation slot. Partially aborted batches stay in the FIFO. [Control route](../apps/server/prefix_controls.cpp#L496)<!-- prefix: "if (abort_ &" --> · [Abort worker](../apps/server/prefix_controls.cpp#L821)<!-- prefix: "void Connect" -->

On shutdown, the reader closes incomplete sockets. The queue discards pending work, and the server interrupts cache GET waits before joining HTTP workers. The [transfer lease](prefix-cache.md#constraints-and-failure-behavior) protects destinations through shutdown. After the HTTP workers join, the transport rethrows the first error from its reader or workers, if any. During batch service destruction, the cache waits for submitted stores to finish and joins its own workers. [HTTP shutdown](../apps/server/prefix_controls.cpp#L972)<!-- prefix: "queue.stop()" --> · [Cache worker shutdown](../apps/server/batch_cache.cpp#L104)<!-- prefix: "for (auto& w" -->

## Completion streams

For benchmark, text, chat, and `/generate` streams, the completion worker opens chunked HTTP 200 SSE after successful decoder admission. The scheduler signals admission after `admit` or `admit_multimodal` succeeds. The worker calls `on_admitted` before delivering a pending token or successful terminal result. A text completion batch or `/generate` batch opens one stream when at least one item has been admitted and the worker has observed no item error. Headers can arrive while prefill is still running. Nonstreaming text, chat, and `/generate` return JSON when generation finishes. [Decoder admission](../apps/server/batch_service.cpp#L239)<!-- prefix: "job->admitte" --> · [Single-request callback order](../apps/server/batch_service.cpp#L425)<!-- prefix: "const bool a" --> · [Batch callback order](../apps/server/batch_service.cpp#L629)<!-- prefix: "bool BatchSe" --> · [SSE headers](../apps/server/http.cpp#L690)<!-- prefix: "bool EventSt" -->

A fully aborted `/generate` request can open SSE to deliver its abort results without decoder admission. A stream that completes with abort results sends one `[DONE]`. [Waiting response](../apps/server/native_generate.cpp#L366)<!-- prefix: "void complet" -->

Before SSE opens, failures use JSON HTTP responses. Invalid completion requests return 400, queue overload returns 503, and [decoder capacity failures](batch-graphs.md#resident-memory-admission) return 413 or 503. Unexpected completion failures before headers return 500. Once SSE opens, the HTTP status stays 200. If generation throws, the worker attempts an SSE error event and ends the stream without `[DONE]`, even if it has sent zero tokens. [JSON error mapping](../apps/server/prefix_controls.cpp#L178)<!-- prefix: "void handle_" --> · [Benchmark error event](../apps/server/completions.cpp#L155)<!-- prefix: "} catch (con" --> · [Text and chat error event](../apps/server/text_completion.cpp#L202)<!-- prefix: "void complet" -->

After admission, the worker polls the SSE writer while waiting for decoder results. For a single request, it skips the poll when an error or finish is pending. For a text completion batch or `/generate` batch, it polls once per consumption loop before checking for failures and ready results. At 15 seconds of wire silence, the next eligible poll sends `:\n\n`. The writer uses a monotonic clock and restarts the interval after a successful header, data, or comment write. Any item event restarts the shared batch clock. Buffered tokens that produce no SSE bytes leave the timer unchanged. A failed write cancels the request. Clients skip comments and measure [TTFT](performance.md#measurement) at the first actual output token. Headers, comments, role events, and usage carry no output tokens.
[Single-request polling](../apps/server/batch_service.cpp#L427)<!-- prefix: "const bool p" --> · [Batch polling](../apps/server/batch_service.cpp#L638)<!-- prefix: "const bool p" --> · [Write timestamp](../apps/server/http.cpp#L682)<!-- prefix: "bool EventSt" --> · [Comment deadline](../apps/server/http.cpp#L701)<!-- prefix: "bool EventSt" --> · [Client parsing](../benchmarks/streaming/client.py#L156)<!-- prefix: "def record_s" -->

## Decode scheduling

Ready multi-item batches rotate refill priority after each admission. A cancellation or stream poll retains its ticket's eligibility for one poll interval. If the callback remains blocked, other ready batches can claim free lanes. Returning to enqueue restores eligibility using the same ticket.

A decoder step occupies a FIFO mailbox entry until its callback accepts it. The callback runs outside the scheduler mutex. Ordinary scheduling waits for acknowledgement. [Mailbox](../include/sglang/server/output_mailbox.h#L10)<!-- prefix: "struct Outpu" --> · [Single-request acknowledgement](../apps/server/batch_service.cpp#L450)<!-- prefix: "job.pending." --> · [Batch acknowledgement](../apps/server/batch_service.cpp#L617)<!-- prefix: "job.pending." -->

Ready lanes wait up to 1 millisecond for peers. At K0 with CUDA graphs, 25 ready lanes end grace once all 32 live jobs qualify. [Grace interval](../include/sglang/runtime/server/batch_service_state.h#L105)<!-- prefix: "inline const" --> · [Early dispatch conditions](../apps/server/batch_service.cpp#L101)<!-- prefix: "bool BatchSe" --> · [Wait](../apps/server/batch_service.cpp#L168)<!-- prefix: "changed.wait" --> · [Publication](../apps/server/batch_service.cpp#L262)<!-- prefix: "bool BatchSe" -->

The worker rechecks selected lanes before dispatch. Later acknowledgements
can produce a separate step. See the [graph eligibility rules](batch-graphs.md#runtime-path)
for the selected width. [Readiness check](../apps/server/batch_service.cpp#L202)<!-- prefix: "if (!job || " -->

### Bounded decode output

`DecoderCallbacks::buffer_decode_output` defaults to false. The benchmark ID stream on `/v1/completions` and scalar per-token SSE on `/generate` set it. Native generation uses step delivery for text stops, constraints, scores, and indexed requests. OpenAI text and chat use ordinary readiness. [Callback flag](../include/sglang/decoder.h#L152)<!-- prefix: "bool buffer_" --> · [Benchmark stream](../apps/server/completions.cpp#L135)<!-- prefix: "void stream_" --> · [Native dispatch](../apps/server/native_generate.cpp#L408)<!-- prefix: "} else if (!" -->

Positive-MTP buffering requires scalar per-token opt-in and greedy text without media, constraints, forced eager execution, top logits, label scoring, or target-only options such as penalties, logit bias, and minimum output length. The writer must acknowledge the first whole step. [Effective options](../apps/server/batch_service.cpp#L856)<!-- prefix: "job->options" --> · [Eligibility](../include/sglang/runtime/server/batch_service_state.h#L74)<!-- prefix: "bool can_buf" --> · [Target-only options](../include/sglang/decoder.h#L109)<!-- prefix: "bool require" -->

The FIFO holds two whole decoder steps, including the step being delivered. A full mailbox blocks only its lane. The writer frees each entry after accepting all its IDs. On stop or length, it drains the FIFO before reporting completion. A terminal checkpoint can delay completion while it waits for store capacity. After submission the request can finish while the store worker retains the allocation views. [Readiness](../include/sglang/runtime/server/batch_service_state.h#L70)<!-- prefix: "bool ready(b" --> · [Publication](../apps/server/batch_service.cpp#L262)<!-- prefix: "bool BatchSe" --> · [FIFO delivery](../apps/server/batch_service.cpp#L440)<!-- prefix: "if (job.has_" --> · [Checkpoint submission](../src/runtime/execution/decoder_batch_prefix.cpp#L99)<!-- prefix: "bool Decoder" -->

At server depth zero, the separate cohort rule requires CUDA graphs, greedy prompts of at least 32,768 IDs, and first-output acknowledgement from every live job. Unindexed step-delivered `/generate` streams can opt in at K0. With 16–31 eligible lanes, all mailboxes having room ends grace early. After grace, any with room can advance. A full 32-lane cohort can dispatch at 25 ready mailboxes. Cancelled, aborted, or otherwise ineligible jobs prevent buffered dispatch. Under ordinary readiness, each lane drains its mailbox before decoding again. [Cohort guards](../apps/server/batch_service.cpp#L101)<!-- prefix: "bool BatchSe" --> · [Step-stream enrollment](../apps/server/batch_service.cpp#L521)<!-- prefix: "job->buffer_" -->

A failed write or disconnect cancels the request and waits for decoder retirement before slot reuse. Scalar writers drain deliverable output before reporting a decoder error. If native abort wins the state transition during a worker fault, an acknowledged, eligible positive-MTP lane still owned by the worker yields an abort result. Peers receive the error, and a permanent fault makes the worker unhealthy. [Disconnect retirement](../apps/server/batch_service.cpp#L409)<!-- prefix: "void BatchSe" --> · [Fault handling](../apps/server/batch_service.cpp#L71)<!-- prefix: "void BatchSe" --> · [Abort delivery](native-abort.md#runtime-path)

## Verification

Run the [CPU reader and queue checks](../tests/prefix/README.md#cpu-reader-queue) with prefix reuse off and on. Exercise partial headers and bodies at the reader limit and absolute deadline, large buffered requests, overload, cancellation, FIFO replacement, and shutdown. Require responsive `/health` and `/ready` while readers or completions occupy their slots and during reset acknowledgement. With reuse enabled, reset during a claimed completion must return HTTP 409.

Run the [CPU placement probe](../tests/batch/run_indexed_admission.sh#L1)<!-- prefix: "#!/usr/bin/e" --> with the source directory and binary output path. A one-item batch must wait for its busy owner lane even when another lane is free.

Probe generation on the default port.

```bash
curl -i --max-time 30 http://127.0.0.1:17384/health_generate
```

On an idle healthy server, require HTTP 200 with an empty body. Hold publication on busy lanes, then release a user token. The waiting probe must return 200 without synthetic admission. Require empty 503 responses on deadline, FIFO overflow, and shutdown of writable active or queued probes. Check `/health` and `/ready` while the probe waits. Disconnect it and require the next probe to run. After permanent worker failure, check all three [health responses](#health-and-readiness).

Run `bash tests/batch/run_service_mtp.sh "$PWD" "$PWD/artifacts/service-mtp-check"` from the repository root in the managed environment with `artifacts/` present. Require one admission callback before token delivery. [Callback assertions](../tests/batch/service_mtp.cpp#L41)<!-- prefix: "callbacks.on" -->

Run the host decode grace regression from the repository root in the default Linux Pixi environment. Create `artifacts/` first. The stub decoder uses GNU linker wrapping and libstdc++.

```bash
bash tests/batch/run_service_decode_grace.sh "$PWD" "$PWD/artifacts/service-decode-grace-check"
```

With all outputs pending, a spurious wake must leave the worker waiting. At 24 acknowledgements it stays in grace. At 25 it dispatches before timeout. Release the remaining seven for a separate dispatch. Require two ordered IDs and a length finish per item, reset rejection while streaming, and successful idle reset. [Threshold assertions](../tests/batch/service_decode_grace.cpp#L145)<!-- prefix: "fixture.awai" -->

Run `bash tests/batch/run_service_mtp_mailbox.sh "$PWD" "$PWD/artifacts/service-mtp-mailbox"` in the same environment. The fixture checks first acknowledgement, the two-step bound, ready peers, ordered IDs, stop and length, and slot reuse. It covers opt-out, target-only options, and step delivery. A winning native abort drops unclaimed overlap, and peers still receive recoverable and permanent faults. It also checks abort before and after a completion claim and fault after a claim. [Scalar fixture](../tests/batch/service_mtp_mailbox.cpp#L107)<!-- prefix: "void check_s" --> · [Abort and fault fixture](../tests/batch/service_mtp_mailbox.cpp#L280)<!-- prefix: "void check_a" --> · [Completion claim fixture](../tests/batch/service_mtp_mailbox.cpp#L351)<!-- prefix: "void check_c" -->

The roster cases check late arrival, abort, blocked or disconnected writers, and trace order. The order fixture covers already-published jobs and replacement after retirement. [Cold-roster fixture](../tests/batch/service_mtp_mailbox.cpp#L425)<!-- prefix: "void check_p" --> · [Publication order fixture](../tests/batch/service_mtp_mailbox.cpp#L511)<!-- prefix: "void check_p" --> · [Replacement fixture](../tests/batch/service_mtp_mailbox.cpp#L546)<!-- prefix: "void check_p" --> · [Trace control](../tests/batch/service_mtp_mailbox.cpp#L611)<!-- prefix: "void check_p" -->

Run the [streaming verification](development.md#streaming-verification) on a reserved B200. The [native prefill reset check](../tests/prefill/README.md) requires headers before prefill finishes. It resets after a traced chunk, checks retirement to zero request bytes, and verifies a replacement. [Reset assertions](../tests/prefill/run_native.py#L145)<!-- prefix: "def check_re" -->

Check wire silence with no decoder steps and with steps that [buffer tokens](lessons.md#l008--buffered-tokens-suppress-sse-keepalives). Require a comment after 15 seconds and measure TTFT at the first output ID frame. Exercise rejection before headers and generation failure after headers but before the first token. Require JSON status for rejection and an SSE error without `[DONE]` for the open stream.
