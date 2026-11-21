# Native HTTP transport

The loopback server uses one HTTP reader and one batch scheduler at every capacity. See the [launch procedure](../README.md#serve-token-ids-over-http) for server options. [Startup](../apps/sglang-server.cpp#L219)<!-- prefix: "serve_connec" -->

## Health and readiness

Send an empty GET request for each check. All three checks bypass the completion queue and work with prefix reuse off or on.

| Route | Responsibility | Response |
| --- | --- | --- |
| `/health` | The HTTP reader reports liveness and the configured scheduler limits. | HTTP 200 with JSON fields `max_active` and `max_num_batched_tokens`, including after permanent worker failure. [Health response](../apps/sglang-server.cpp#L141)<!-- prefix: "std::string " --> |
| `/ready` | The HTTP reader checks the permanent decoder worker failure flag. | HTTP 200 with `{"ready":true}` while the worker has no permanent failure. After failure, HTTP 503 with `{"ready":false,"reason":"worker_failed"}`. [Readiness response](../apps/server/http.cpp#L467)<!-- prefix: "void write_r" --> · [Worker flag](../apps/server/batch_service.cpp#L764)<!-- prefix: "bool BatchSe" --> |
| `/health_generate` | A dedicated HTTP worker asks the batch scheduler for generation progress. | HTTP 200 for progress within the probe window. HTTP 503 for queue overload, probe failure, timeout, permanent worker failure, or shutdown. Both statuses carry `Content-Length: 0` and `Connection: close`. [Dispatch](../apps/server/prefix_controls.cpp#L252)<!-- prefix: "void Connect" --> · [Reply](../apps/server/http.cpp#L476)<!-- prefix: "void write_g" --> |

A permanent worker failure also causes new completions to return HTTP 500. A generation health timeout leaves the readiness flag unchanged. [Batch failure](../apps/server/batch_service.cpp#L186)<!-- prefix: "void BatchSe" -->

The scheduler defaults to 32 active requests and an aggregate budget of 8,192 tokens per step. Change the budget with `--max-num-batched-tokens` (1–32,768) and use `/health` to verify the configured value. [Default budget](../include/sglang/decoder.h#L22)<!-- prefix: "inline const" -->

### Generation probe lifecycle

The HTTP reader starts a 20-second monotonic deadline when it dispatches the parsed request. The deadline includes time in the health FIFO. The FIFO holds up to `--max-active` waiting probes in addition to the one claimed by its worker. The reader removes disconnected waiters before testing capacity. The health worker owns the claimed socket until it attempts the reply. [Deadline](../apps/server/prefix_controls.cpp#L23)<!-- prefix: "using Clock " --> · [Health admission](../apps/server/prefix_controls.cpp#L456)<!-- prefix: "void Connect" --> · [Health worker](../apps/server/prefix_controls.cpp#L480)<!-- prefix: "void Connect" -->

The scheduler can use its latest successful token publication if it occurred after the probe started and by its deadline. Otherwise it registers one waiter for subsequent publication. While user jobs occupy decoder lanes, they can satisfy the probe without reserving a lane for health. Admission, prefill without a token, cancellation, and failed steps do not satisfy the probe. Publication is independent of when a client receives the token. [Waiter and timestamp](../apps/server/batch_service.cpp#L767)<!-- prefix: "bool BatchSe" --> · [Token publication](../apps/server/batch_service.cpp#L339)<!-- prefix: "void BatchSe" -->

When all decoder lanes become idle and no prefix reset is pending, the scheduler can install an internal job in the first lane. It submits prompt IDs `[0]` and generates at most one greedy token with MTP disabled. The scheduler removes the waiter on return and marks any internal job cancelled. It retains a still-installed job until the decoder worker can retire it. An installed health job makes prefix reset return HTTP 409. User request and token counters exclude the probe. Lower-level cache counters and native step profiles can include its decoder work. [Idle eligibility](../apps/server/batch_service.cpp#L222)<!-- prefix: "bool BatchSe" --> · [Job creation and cleanup](../apps/server/batch_service.cpp#L781)<!-- prefix: "auto job = s" --> · [Probe options](../include/sglang/decoder.h#L31)<!-- prefix: "struct Decod" --> · [Lane retirement](../apps/server/batch_service.cpp#L266)<!-- prefix: "if (cancelle" --> · [Reset exclusion](../apps/server/batch_service.cpp#L823)<!-- prefix: "bool BatchSe" -->

### Request, failure, and shutdown behavior

After HTTP parsing, a GET with a nonempty body returns a JSON HTTP 404 error. Other methods receive HTTP 405, `Allow: GET`, and a JSON method error. `HEAD` omits the body. These requests do not enter the health FIFO. [Input checks](../apps/server/prefix_controls.cpp#L456)<!-- prefix: "void Connect" --> · [Method response](../apps/server/http.cpp#L450)<!-- prefix: "void write_g" -->

The scheduler wakes the health worker on progress and checks for a disconnect at intervals of up to 10 milliseconds while waiting. A disconnected client can lose the reply. On shutdown, the health worker leaves the active wait and drains the queued health checks. It attempts an empty HTTP 503 on each remaining socket before the transport joins the workers. The 20-second deadline bounds the progress window. The HTTP read and response write have separate I/O limits. [Wait loop](../apps/server/batch_service.cpp#L788)<!-- prefix: "while (!stat" --> · [Disconnect detection](../apps/server/http.cpp#L342)<!-- prefix: "bool disconn" --> · [Queue stop](../apps/server/prefix_controls.cpp#L498)<!-- prefix: "void Connect" --> · [Worker join](../apps/server/prefix_controls.cpp#L563)<!-- prefix: "queue.stop()" -->

### Differences from pinned SGLang

Both generation checks use empty HTTP 200 or 503 replies and a maximum of one greedy output token from `[0]`. Pinned SGLang creates a probe task per request and tests for any newer detokenizer or scheduler response. It checks wall-clock time after creating the task, polls once per second, and defaults to 20 seconds through `SGLANG_HEALTH_CHECK_TIMEOUT`. The native route uses a fixed 20 seconds from HTTP dispatch, scheduler notification on successful token publication, and a 10-millisecond disconnect poll. It has one active health worker and `--max-active` waiting slots. It installs a synthetic job only when the scheduler is idle. [Pinned SGLang timeout](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L228) · [Pinned SGLang probe](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L765-L843)

Pinned SGLang can enable generation on `/health` and can bypass generation on both health paths with its diagnostic setting. It sets its server status to `Up` on probe success and `UnHealthy` on timeout. Its `/ready` returns an empty body from the tokenizer manager's readiness check. The native `/health` always returns the limits, and `/ready` reports the worker flag with JSON. The native generation check has no diagnostic bypass. [Pinned SGLang readiness and probe](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L758-L843)

## Model discovery

Send `GET /v1/models` with an empty body. The route bypasses the completion queue and works with prefix reuse off or on. The server returns HTTP 200 and a JSON object with `object: "list"` and one card in `data`. [Control route](../apps/server/prefix_controls.cpp#L252)<!-- prefix: "void Connect" --> · [Model card writer](../apps/server/prefix_controls.cpp#L242)<!-- prefix: "void write_m" -->

The card uses `Qwen3.8-27B-FP8` for `id` and `root`. Use this ID for native chat and text completions. The card sets `object` to `model`, `owned_by` to `sglang`, and `parent` to `null`. The `created` field gives the response creation time in integer Unix seconds. The `max_model_len` field gives the 262,144-token context shared by prompt and output. Available GPU memory can reduce the admitted capacity. The `benchmark` alias belongs to the [token-ID API](../benchmarks/streaming/README.md#wire-contract).

After successful HTTP parsing, a POST or a GET with a body to `/v1/models` returns HTTP 404.

Use [native model information](native-model-info.md) for the checkpoint paths, model capabilities, and configured chat parsers.

## HTTP admission and shutdown

Let `A` be `--max-active`. The nonblocking loopback listener uses a TCP backlog setting of 16. The reader polls all retained sockets and advances each request incrementally. It retains up to `A + 1` incomplete requests. The reader services ready and expired reads before accepting a waiting connection. The five-second read deadline starts at acceptance and covers receipt of the HTTP headers and body. The header and body limits are 8 KiB and 2 MiB, respectively. Clients can use `shutdown(SHUT_WR)` after a complete request. [Listener](../apps/server/http.cpp#L309)<!-- prefix: "Socket liste" --> · [Reader](../apps/server/http.cpp#L376)<!-- prefix: "void read_co" --> · [Read limits](../apps/server/http.cpp#L29)<!-- prefix: "constexpr st" -->

When a connection is waiting and the reader is full, the reader looks for the oldest eligible incomplete socket. A socket is eligible if it has no readable data and has been idle for at least 250 milliseconds since acceptance or its last successful read. The reader defers acceptance if no socket is eligible. It closes the eligible socket after accepting the new one. During replacement, the socket count can briefly exceed the reader limit by one. [Socket replacement](../apps/server/http.cpp#L351)<!-- prefix: "static void " -->

After HTTP parsing, completions, [`POST /generate`](native-generate.md), and the [tokenization routes](native-tokenization.md) enter one FIFO. Up to `A` requests can wait, and up to `A` workers can claim requests. The queue removes disconnected waiters before checking capacity. It returns a JSON HTTP 503 response when `A` requests are still waiting, even if some workers are idle. The reader checks the shape of `/generate` and reserves its IDs after this capacity check. Workers validate the prompts and options. A normally claimed request holds its HTTP worker slot until its response socket closes. [Queue admission](../apps/server/prefix_controls.cpp#L337)<!-- prefix: "void Connect" --> · [Worker](../apps/server/prefix_controls.cpp#L402)<!-- prefix: "void Connect" -->

A [text completion batch](text-serving.md#send-a-request) or [native generation batch](native-generate.md#runtime-path) counts as one request for HTTP admission. Multi-item batches share eligible decoder lanes and give the longest-waiting ready batch first use of a free lane. Admitting an item moves that batch behind other waiters. Scalar requests and one-item batches reserve their worker's lane. Tokenization uses an HTTP worker and runs on the CPU, including after permanent decoder worker failure. For generation, a TCP reset or failed write cancels unfinished items. The batch retires active lanes before slot reuse, and ready peers can advance while a writer blocks. [Cancellation](../apps/server/batch_service.cpp#L486)<!-- prefix: "void BatchSe" -->

[`POST /abort_request`](native-abort.md) runs on the reader after HTTP parsing. It bypasses the completion queue and works when every generation slot is occupied. It shares the same incomplete-request capacity, replacement policy, and deadlines as other controls. A separate CPU worker can remove a fully aborted generation request from the FIFO and send its result without a generation slot. Partially aborted batches stay in the FIFO. [Control route](../apps/server/prefix_controls.cpp#L252)<!-- prefix: "void Connect" --> · [Abort worker](../apps/server/prefix_controls.cpp#L429)<!-- prefix: "void Connect" -->

On shutdown, the reader closes incomplete sockets. The queue discards pending work and joins the HTTP workers before the batch service is destroyed. The server then propagates the first error that escapes the transport reader or workers. [Shutdown](../apps/server/prefix_controls.cpp#L563)<!-- prefix: "queue.stop()" -->

## Completion streams

For benchmark, text, chat, and `/generate` streams, the completion worker opens chunked HTTP 200 SSE after successful decoder admission. The scheduler signals admission after `admit` or `admit_multimodal` succeeds. The worker calls `on_admitted` before delivering a pending token or successful terminal result. A text completion batch or `/generate` batch opens one stream when at least one item has been admitted and the worker has observed no item error. Headers can arrive while prefill is still running. Nonstreaming text, chat, and `/generate` return JSON when generation finishes. [Decoder admission](../apps/server/batch_service.cpp#L284)<!-- prefix: "job-\u003eadmitte" --> · [Single-request callback order](../apps/server/batch_service.cpp#L500)<!-- prefix: "const bool a" --> · [Batch callback order](../apps/server/batch_service.cpp#L693)<!-- prefix: "bool BatchSe" --> · [SSE headers](../apps/server/http.cpp#L495)<!-- prefix: "bool EventSt" -->

A fully aborted `/generate` request can open SSE to deliver its abort results without decoder admission. A stream that completes with abort results sends one `[DONE]`. [Waiting response](../apps/server/native_generate.cpp#L312)<!-- prefix: "void complet" -->

Before SSE opens, failures use JSON HTTP responses. Invalid completion requests return 400, queue overload returns 503, and [decoder capacity failures](batch-graphs.md#resident-memory-admission) return 413 or 503. Unexpected completion failures before headers return 500. Once SSE opens, the HTTP status stays 200. If generation throws, the worker attempts an SSE error event and ends the stream without `[DONE]`, even if it has sent zero tokens. [JSON error mapping](../apps/server/prefix_controls.cpp#L45)<!-- prefix: "void handle_" --> · [Benchmark error event](../apps/server/completions.cpp#L155)<!-- prefix: "} catch (con" --> · [Text and chat error event](../apps/server/text_completion.cpp#L161)<!-- prefix: "void complet" -->

After admission, the worker polls the SSE writer while waiting for decoder results. For a single request, it skips the poll when an error or finish is pending. For a text completion batch or `/generate` batch, it polls once per consumption loop before checking for failures and ready results. At 15 seconds of wire silence, the next eligible poll sends `:\n\n`. The writer uses a monotonic clock and restarts the interval after a successful header, data, or comment write. Any item event restarts the shared batch clock. Buffered tokens that produce no SSE bytes leave the timer unchanged. A failed write cancels the request. Clients skip comments and measure [TTFT](performance.md#measurement) at the first actual output token. Headers, comments, role events, and usage carry no output tokens. [Single-request polling](../apps/server/batch_service.cpp#L501)<!-- prefix: "const bool p" --> · [Batch polling](../apps/server/batch_service.cpp#L693)<!-- prefix: "bool BatchSe" --> · [Write timestamp](../apps/server/http.cpp#L487)<!-- prefix: "bool EventSt" --> · [Comment deadline](../apps/server/http.cpp#L501)<!-- prefix: "bool EventSt" --> · [Client parsing](../benchmarks/streaming/client.py#L127)<!-- prefix: "def record_s" -->

## Decode scheduling

Ready multi-item batches rotate refill priority after each admission. A cancellation or stream poll retains its ticket's eligibility for one poll interval. If the callback remains blocked, other ready batches can claim free lanes. Returning to enqueue restores eligibility using the same ticket.

A job keeps a decoder step in its FIFO mailbox until its completion worker accepts delivery. The entry remains charged while the callback runs outside the scheduler mutex. In ordinary scheduling, a lane with pending output waits for this acknowledgement. A cancelled lane wakes the decoder worker for retirement. [Mailbox](../include/sglang/server/output_mailbox.h#L10)<!-- prefix: "struct Outpu" --> · [Single-request acknowledgement](../apps/server/batch_service.cpp#L526)<!-- prefix: "job.pending." --> · [Batch acknowledgement](../apps/server/batch_service.cpp#L681)<!-- prefix: "job.pending." -->

When any lane is ready, the decoder worker gives other lanes up to 1 millisecond to get ready. With `--max-active 32`, `--batch-cuda-graphs`, and MTP disabled, this grace can end as soon as at least 25 lanes are ready. All 32 slots must hold uncancelled jobs that have published their first token. First-token publication counts with prefix reuse off or on. At 24 ready lanes, the normal wait conditions still apply. [Grace interval](../apps/server/batch_service.cpp#L74)<!-- prefix: "constexpr au" --> · [Early dispatch conditions](../apps/server/batch_service.cpp#L205)<!-- prefix: "bool BatchSe" --> · [Wait](../apps/server/batch_service.cpp#L229)<!-- prefix: "std::optiona" --> · [Publication](../apps/server/batch_service.cpp#L305)<!-- prefix: "bool BatchSe" -->

The worker rechecks each selected job before dispatch. It can select every ready lane in the snapshot. Lanes that acknowledge later can run in a subsequent step. If 25 to 32 selected lanes all qualify for [packed decode](batch-graphs.md#runtime-path), the decoder uses the same 32-row bucket. A step with 25 ready lanes can therefore start sooner, with a separate step for the other seven if they acknowledge later. [Readiness check](../apps/server/batch_service.cpp#L256)<!-- prefix: "bool BatchSe" --> · [Bucket selection](../src/text/decoder_graph_packed.cpp#L223)<!-- prefix: "bool PackedD" -->

### Bounded decode output

The benchmark ID stream on `/v1/completions` and eligible single-prompt `/generate` streams opt in through `DecoderCallbacks::buffer_decode_output`. The callback flag defaults to false. OpenAI text completions in all prompt forms, chat completions, step-delivered or batch `/generate`, and nonstreaming generation use ordinary readiness. [Callback flag](../include/sglang/decoder.h#L107)<!-- prefix: "bool buffer_" --> · [Benchmark stream](../apps/server/completions.cpp#L135)<!-- prefix: "void stream_" --> · [Single-prompt stream](../apps/server/native_generate.cpp#L210)<!-- prefix: "void complet" --> · [Native dispatch](../apps/server/native_generate.cpp#L341)<!-- prefix: "void complet" -->

Buffering requires `--max-active 32`, `--batch-cuda-graphs`, and MTP disabled. Each of the 32 slots must hold a live, opted-in job whose first output callback has returned successfully. Each prompt must contain at least 32,768 IDs and use greedy selection, with no media, constraint, forced eager execution, or top-logit request. At least 25 mailboxes must have room when the scheduler takes its snapshot. It rechecks readiness and cancellation before dispatching each lane. [Cohort eligibility](../apps/server/batch_service.cpp#L205)<!-- prefix: "bool BatchSe" --> · [Job eligibility](../apps/server/batch_service.cpp#L48)<!-- prefix: "bool can_buf" --> · [Greedy selection](../include/sglang/decoder.h#L55)<!-- prefix: "bool samples" -->

The mailbox holds at most two decode steps, including any step whose callback is running. A full mailbox blocks that lane and bounds lookahead to one K0 token beyond the outstanding callback. When the cohort falls back to ordinary scheduling, each lane drains its mailbox before decoding again. The decoder decides stop and length and retires the lane at the terminal step. The writer drains accepted output in FIFO order before serializing the terminal result, using the route's existing token, stop, metadata, and SSE formats. [Readiness](../apps/server/batch_service.cpp#L45)<!-- prefix: "bool ready(b" --> · [Publication and retirement](../apps/server/batch_service.cpp#L305)<!-- prefix: "bool BatchSe" --> · [FIFO delivery](../apps/server/batch_service.cpp#L514)<!-- prefix: "const bool c" -->

A failed write or disconnect cancels the job and retires its decoder work before slot reuse. The job records whether steps have ever overlapped. That history determines [native abort delivery](native-abort.md#runtime-path) even after ordinary scheduling resumes. An already dispatched decoder step can finish, and publication suppresses its output if cancellation has won. [Disconnect retirement](../apps/server/batch_service.cpp#L486)<!-- prefix: "void BatchSe" --> · [Overlap history](../apps/server/batch_service.cpp#L323)<!-- prefix: "job.output_o" -->

## Verification

With the server running on the default port, request its model list.

```bash
curl --fail-with-body -i http://127.0.0.1:17384/v1/models
```

Require HTTP 200 and one card with the fields specified in [Model discovery](#model-discovery).

Run the [CPU reader and queue checks](../tests/prefix/README.md#cpu-reader-queue) with prefix reuse off and on. Check the reader limit and absolute deadline for partial headers and bodies. Check that large buffered requests reach the completion queue. Check overload, FIFO replacement after cancellation, and worker shutdown. Require `/health` and `/ready` responses while partial reads or completions occupy their respective slots. With prefix reuse on, send a reset while a completion holds its slot. Require HTTP 409. Require `/health` and `/ready` responses while reset acknowledgement is pending.

Run the [CPU placement probe](../tests/batch/run_indexed_admission.sh#L1)<!-- prefix: "#!/usr/bin/e" --> with the source directory and a binary output path. The probe holds a one-item batch's owner lane busy while another lane is free and checks that the item waits for its owner lane.

With the server running on the default port, send an empty generation check.

```bash
curl -i --max-time 30 http://127.0.0.1:17384/health_generate
```

Require HTTP 200 with an empty body on an idle healthy server. With a controlled decoder, hold publication on busy lanes and require the probe to wait. Release a live user token and require HTTP 200 without a synthetic admission. Withhold progress for the full deadline and require an empty HTTP 503. Fill the health FIFO and require an empty HTTP 503 for the extra probe. Check `/health` and `/ready` while the probe worker waits. Disconnect the active probe and verify that the worker can claim the next check. On shutdown, check that writable active and queued probe sockets receive empty HTTP 503 replies. After permanent worker failure, verify the three distinct responses in [Health and readiness](#health-and-readiness).

Run `bash tests/batch/run_service_mtp.sh "$PWD" "$PWD/artifacts/service-mtp-check"` from the repository root in the managed environment with `artifacts/` present. Require one admission callback before token delivery. [Callback assertions](../tests/batch/service_mtp.cpp#L37)<!-- prefix: "callbacks.on" -->

Run the host decode grace regression from the repository root in the default Linux Pixi environment. Create `artifacts/` before the run. The driver uses GNU linker wrapping and libstdc++ with a stub decoder.

```bash
bash tests/batch/run_service_decode_grace.sh "$PWD" "$PWD/artifacts/service-decode-grace-check"
```

With all outputs pending, require the worker to wait again after a spurious wake. Require 24 acknowledgements to remain in grace. Require 25 to dispatch those lanes before the timer expires. Release the last seven and require a separate dispatch. Require two ordered token IDs and a length finish for every item. Check that reset is rejected during streaming and succeeds when idle. [Threshold assertions](../tests/batch/service_decode_grace.cpp#L142)<!-- prefix: "fixture.rele" -->

For bounded decode output, block a writer after its first callback and verify one additional K0 step can run for its lane. Hold all writers until mailboxes are full and verify decoding waits. Release writers and check ordered IDs, stop and length retirement, and fallback when the cohort loses eligibility. Race native abort against normal terminal delivery with one claimed step and one queued step. When abort wins, check that the claimed callback completes, the queued step is dropped, and the slot becomes reusable. See the [CPU scheduler regression](../tests/batch/README.md).

The focused B200 replay checked greedy IDs on frozen histories for four full-length W5 prompts and 32 decode steps. Full W5 parity still requires 32 full-length prompts and 4,096 free-running output tokens per lane. CPU callback tests and modeled receipt delays leave socket-writer latency, GPU scheduler throughput, and end-to-end W5 speedup unmeasured.

Run the [streaming verification](development.md#streaming-verification) on a reserved B200. The [native prefill reset check](../tests/prefill/README.md) requires headers before prefill finishes. It resets after a traced chunk, checks retirement to zero request bytes, and verifies a replacement. [Reset assertions](../tests/prefill/run_native.py#L139)<!-- prefix: "def check_re" -->

Check wire silence with no decoder steps and with steps that [buffer tokens](lessons.md#l008--buffered-tokens-suppress-sse-keepalives). Require a comment after 15 seconds and measure TTFT at the first output ID frame. Exercise rejection before headers and generation failure after headers but before the first token. Require JSON status for rejection and an SSE error without `[DONE]` for the open stream.
