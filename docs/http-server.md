# Native HTTP transport

The loopback server uses one HTTP reader and one batch scheduler at every capacity. See the [launch procedure](../README.md#serve-token-ids-over-http) for server options. [Startup](../apps/sglang-server.cpp#L191)<!-- prefix: "serve_connec" -->

Send empty GET requests to `/health` and `/ready`. `/health` returns HTTP 200 with the limits. `/ready` returns HTTP 200 with `{"ready":true}`. After permanent decoder worker failure, it returns HTTP 503 with `{"ready":false,"reason":"worker_failed"}`. Both routes bypass the completion queue. New completions return HTTP 500 after permanent decoder worker failure. The [cache controls](prefix-cache.md#native-http-controls) require prefix reuse. [Health response](../apps/sglang-server.cpp#L138)<!-- prefix: "std::string " --> · [Readiness response](../apps/server/http.cpp#L421)<!-- prefix: "void write_r" --> · [Batch failure](../apps/server/batch_service.cpp#L182)<!-- prefix: "void BatchSe" -->

## Model discovery

Send `GET /v1/models` with an empty body. The route bypasses the completion queue and works with prefix reuse off or on. The server returns HTTP 200 and a JSON object with `object: "list"` and one card in `data`. [Control route](../apps/server/prefix_controls.cpp#L229)<!-- prefix: "void Connect" --> · [Model card writer](../apps/server/prefix_controls.cpp#L219)<!-- prefix: "void write_m" -->

The card uses `Qwen3.8-27B-FP8` for `id` and `root`. Use this ID for native chat and text completions. The card sets `object` to `model`, `owned_by` to `sglang`, and `parent` to `null`. The `created` field gives the response creation time in integer Unix seconds. The `max_model_len` field gives the 262,144-token context shared by prompt and output. Available GPU memory can reduce the admitted capacity. The `benchmark` alias belongs to the [token-ID API](../benchmarks/streaming/README.md#wire-contract).

After successful HTTP parsing, a POST or a GET with a body to `/v1/models` returns HTTP 404.

## HTTP admission and shutdown

Let `A` be `--max-active`. The nonblocking loopback listener uses a TCP backlog setting of 16. The reader polls all retained sockets and advances each request incrementally. It retains up to `A + 1` incomplete requests. The reader services ready and expired reads before accepting a waiting connection. The five-second read deadline starts at acceptance and covers receipt of the HTTP headers and body. The header and body limits are 8 KiB and 2 MiB, respectively. Clients can use `shutdown(SHUT_WR)` after a complete request. [Listener](../apps/server/http.cpp#L284)<!-- prefix: "Socket liste" --> · [Reader](../apps/server/http.cpp#L351)<!-- prefix: "void read_co" --> · [Read limits](../apps/server/http.cpp#L27)<!-- prefix: "constexpr st" -->

When a connection is waiting and the reader is full, the reader looks for the oldest eligible incomplete socket. A socket is eligible if it has no readable data and has been idle for at least 250 milliseconds since acceptance or its last successful read. The reader defers acceptance if no socket is eligible. It closes the eligible socket after accepting the new one. During replacement, the socket count can briefly exceed the reader limit by one. [Socket replacement](../apps/server/http.cpp#L326)<!-- prefix: "static void " -->

After HTTP parsing, `POST /v1/completions`, `POST /v1/chat/completions`, and [`POST /generate`](native-generate.md) enter one FIFO. Up to `A` requests can wait, and up to `A` workers can claim requests. Completion workers parse the request JSON after they claim a request. The queue removes disconnected waiters before checking capacity. It returns a JSON HTTP 503 response when `A` requests are still waiting, even if some workers are idle. A claimed completion keeps its slot until its response socket closes. A TCP reset or failed write cancels active decoder work. The batch retires that work before slot reuse, and ready peers can advance while a writer blocks. [Queue admission](../apps/server/prefix_controls.cpp#L279)<!-- prefix: "void Connect" --> · [Worker](../apps/server/prefix_controls.cpp#L331)<!-- prefix: "void Connect" --> · [Cancellation](../apps/server/batch_service.cpp#L432)<!-- prefix: "void BatchSe" -->

On shutdown, the reader closes incomplete sockets. The queue discards pending work and joins the completion workers before the batch service is destroyed. The server then propagates the first error that escapes the transport reader or workers. [Shutdown](../apps/server/prefix_controls.cpp#L398)<!-- prefix: "queue.stop()" -->

## Completion streams

For benchmark, text, chat, and `/generate` streams, the completion worker opens chunked HTTP 200 SSE after successful decoder admission. The batch signals admission after `admit` or `admit_multimodal` succeeds. The worker calls `on_admitted` before delivering a pending token or successful terminal result. Headers can arrive while prefill is still running. Nonstreaming text, chat, and `/generate` return JSON when generation finishes. [Decoder admission](../apps/server/batch_service.cpp#L249)<!-- prefix: "job-\u003eadmitte" --> · [Callback order](../apps/server/batch_service.cpp#L446)<!-- prefix: "const bool a" --> · [SSE headers](../apps/server/http.cpp#L442)<!-- prefix: "bool EventSt" -->

Before SSE opens, failures use JSON HTTP responses. Invalid completion requests return 400, queue overload returns 503, and [decoder capacity failures](batch-graphs.md#resident-memory-admission) return 413 or 503. Unexpected completion failures before headers return 500. Once SSE opens, the HTTP status stays 200. If generation throws, the worker attempts an SSE error event and ends the stream without `[DONE]`, even if it has sent zero tokens. [JSON error mapping](../apps/server/prefix_controls.cpp#L34)<!-- prefix: "void handle_" --> · [Benchmark error event](../apps/server/completions.cpp#L154)<!-- prefix: "} catch (con" --> · [Text and chat error event](../apps/server/text_completion.cpp#L409)<!-- prefix: "} catch (con" -->

After admission, the worker polls the SSE writer during pending decoder steps and timed waits. It skips the poll when an error or finish is pending. At 15 seconds of wire silence, the next eligible poll sends `:\n\n`. The writer uses a monotonic clock and restarts the interval after a successful header, data, or comment write. Buffered tokens that produce no SSE bytes leave the timer unchanged. A failed write cancels the request. Clients skip comments and measure [TTFT](performance.md#measurement) at the first actual output token. Headers, comments, role events, and usage carry no output tokens. [Poll scheduling](../apps/server/batch_service.cpp#L447)<!-- prefix: "const bool p" --> · [Write timestamp](../apps/server/http.cpp#L434)<!-- prefix: "bool EventSt" --> · [Comment deadline](../apps/server/http.cpp#L448)<!-- prefix: "bool EventSt" --> · [Client parsing](../benchmarks/streaming/client.py#L127)<!-- prefix: "def record_s" -->

## Verification

With the server running on the default port, request its model list.

```bash
curl --fail-with-body -i http://127.0.0.1:17384/v1/models
```

Require HTTP 200 and one card with the fields specified in [Model discovery](#model-discovery).

Run the [CPU reader and queue checks](../tests/prefix/README.md#cpu-reader-queue) with prefix reuse off and on. Check the reader limit and absolute deadline for partial headers and bodies. Check that large buffered requests reach the completion queue. Check overload, FIFO replacement after cancellation, and worker shutdown. Require health and readiness responses while partial reads or completions occupy their respective slots. With prefix reuse on, send a reset while a completion holds its slot. Require HTTP 409. Require health and readiness responses while reset acknowledgement is pending.

Run `bash tests/batch/run_service_mtp.sh "$PWD" "$PWD/artifacts/service-mtp-check"` from the repository root in the managed environment with `artifacts/` present. Require one admission callback before token delivery. [Callback assertions](../tests/batch/service_mtp.cpp#L37)<!-- prefix: "callbacks.on" -->

Run the [streaming verification](development.md#streaming-verification) on a reserved B200. The [native prefill reset check](../tests/prefill/README.md) requires headers before prefill finishes. It resets after a traced chunk, checks retirement to zero request bytes, and verifies a replacement. [Reset assertions](../tests/prefill/run_native.py#L139)<!-- prefix: "def check_re" -->

Check wire silence with no decoder steps and with steps that [buffer tokens](lessons.md#l008--buffered-tokens-suppress-sse-keepalives). Require a comment after 15 seconds and measure TTFT at the first output ID frame. Exercise rejection before headers and generation failure after headers but before the first token. Require JSON status for rejection and an SSE error without `[DONE]` for the open stream.
