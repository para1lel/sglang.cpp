# Native request abort

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Native generation | Assign one request ID per prompt and track which item owns each ID. | [ID input](../apps/server/native_generate.cpp#L362)<!-- prefix: "std::vector\u003c" --> · [Registration](../apps/server/native_generate.cpp#L410)<!-- prefix: "NativeGenera" --> |
| HTTP control | Dispatch `POST /abort_request` on the reader and release fully aborted queue entries. | [Control route](../apps/server/prefix_controls.cpp#L252)<!-- prefix: "void Connect" --> · [Queue cleanup](../apps/server/prefix_controls.cpp#L429)<!-- prefix: "void Connect" --> |
| Decoder service | Skip waiting items, retire running items, and apply abort delivery according to the job's overlap history. | [ID state](../include/sglang/server/batch_service.h#L34)<!-- prefix: "class BatchR" --> · [Retirement](../apps/server/batch_service.cpp#L301)<!-- prefix: "bool BatchSe" --> · [Waiting items](../apps/server/batch_service.cpp#L638)<!-- prefix: "bool BatchSe" --> · [Abort delivery](../apps/server/batch_service.cpp#L42)<!-- prefix: "bool has_out" --> |
| Native response | Report each item's terminal abort on its original generation connection. | [Finish metadata](../apps/server/native_generate.cpp#L118)<!-- prefix: "if (terminal" --> · [Pre-admission response](../apps/server/native_generate.cpp#L306)<!-- prefix: "void complet" --> |

## Runtime path

Supply `rid` with [`POST /generate`](native-generate.md) if you need to abort before the first result. The response reports the effective ID in `meta_info.id`.

| `rid` in `/generate` | Effective IDs |
| --- | --- |
| Omitted or null | The server generates an ID for each prompt. |
| String with one scalar prompt | The item uses that string. The server preserves whitespace and UTF-8. |
| String with a batch | Each item uses the string followed by `_` and its unpadded zero-based index. This also applies to a one-item batch. |
| Array of strings with a batch | Each item uses the string at its input position. The array length must equal the prompt count. |

Once the HTTP queue has capacity, the reader checks the prompt shape and reserves all effective IDs before it queues the request. The worker validates the remaining generation options and prepares all prompts before decoder submission.

Send the following request from a second connection while `job-42` is live.

```bash
curl --fail-with-body -i http://127.0.0.1:17384/abort_request \
  -H 'Content-Type: application/json' \
  --data-binary '{"rid":"job-42"}'
```

HTTP 200 has an empty body. It acknowledges receipt of the control request. Use the original `/generate` response to observe completion.

- A named abort targets an exact live ID. If it changes that item's state, it also marks already-dispatched items whose IDs start with the same string. A batch string such as `job` produces IDs such as `job_0`. Aborting `job` has no effect unless `job` itself is also a live item.
- `abort_all: true` marks all currently registered native generation items, including items with server-assigned IDs and items still waiting. The `rid` value has no effect in this mode.
- An item aborted before decoder submission produces empty `output_ids` and a finish reason of `{"type":"abort","message":"Abort in waiting queue"}`.
- An item aborted after submission returns `{"type":"abort","message":"Aborted","status_code":null,"err_type":null}` as its finish reason. Its output IDs follow the delivery rules below.

The scheduler observes the abort at its next cancellation check. Work already in flight can finish its current step. A normal terminal result, including a [native text stop](native-generate.md#text-stops), stays final if it wins the state transition. [Matching](../apps/server/native_generate.cpp#L464)<!-- prefix: "bool changed" --> · [Publish](../apps/server/batch_service.cpp#L350)<!-- prefix: "bool BatchSe" -->

If a job has never published a step while another is pending, an abort still delivers any single already-published step before the terminal result. Once a job has published overlapping steps through [bounded decode output](http-server.md#bounded-decode-output), an abort winner discards every unclaimed step, even after ordinary scheduling resumes. A callback already claimed can finish. The abort terminal includes the IDs accepted by the writer. [Overlap history](../apps/server/batch_service.cpp#L368)<!-- prefix: "job.output_o" --> · [Single-item delivery](../apps/server/batch_service.cpp#L42)<!-- prefix: "bool has_out" --> · [Batch delivery](../apps/server/batch_service.cpp#L679)<!-- prefix: "const bool h" -->

An abort terminal uses the regular JSON or SSE result shape. Batch JSON retains input order, and batch SSE retains each item's index. A stream that completes normally with abort results sends one `[DONE]`. A fully aborted request can open SSE before it has admitted any decoder work. A partially aborted batch keeps its remaining items and opens SSE when one is admitted.

### ID reuse

Duplicate effective IDs within a batch or against live items return HTTP 400. A failed registration releases its reservations. A finished item can reuse its ID while siblings from its previous batch are still running. Cleanup removes a registration only when it still owns that ID.

If a request disconnects or fails, reuse its IDs after the associated work and queue entry have been cleaned up. A late named abort targets whichever live item currently owns that ID. Use distinct caller IDs when late control requests can overlap a replacement. [Cleanup](../apps/server/native_generate.cpp#L395)<!-- prefix: "~Registratio" --> · [Terminal state](../include/sglang/server/batch_service.h#L67)<!-- prefix: "bool finishe" -->

### Capacity

The control route uses the [shared HTTP reader](http-server.md#http-admission-and-shutdown). It can run with every generation slot occupied and the completion queue full. It still needs the reader to receive a complete HTTP request. If all read positions are occupied, a connection waits for read progress, an idle-socket replacement, or a read timeout.

With `--max-active A`, the completion queue holds up to `A` waiters and the reader retains up to `A + 1` incomplete requests. An idle incomplete request becomes eligible for replacement after 250 milliseconds. The read deadline is five seconds from acceptance.

One additional CPU worker removes a fully aborted pending request from the completion queue. It validates its prompts and writes the terminal response without claiming a generation slot. Invalid prompts or options still return the usual generation error. A partially aborted pending batch stays in FIFO order so its remaining items can use a normal slot. [Queue admission](../apps/server/prefix_controls.cpp#L337)<!-- prefix: "void Connect" --> · [Abort worker](../apps/server/prefix_controls.cpp#L444)<!-- prefix: "auto work = " -->

## Constraints and failure behavior

- Use `POST` with the exact `/abort_request` target. Aliases, query strings, trailing slashes, and other methods return HTTP 404 after successful HTTP parsing.
- Send a JSON object. `rid` accepts a string or null. `abort_all` accepts a boolean and defaults to `false` when omitted. A null `abort_all` or invalid field type returns HTTP 400.
- An empty, missing, null, unknown, or finished `rid` with `abort_all: false` returns HTTP 200 without changing any item. An empty scalar generation ID therefore requires `abort_all` or disconnect for cancellation.
- The control uses the shared header and body size limits. Send `Content-Length` for the JSON body. The route accepts `Content-Type: application/json` and has no media-type requirement of its own.
- The control is available with `--prefix-cache-entries` at `0` or any supported positive value. Published prefix-cache effects can remain. There is no separate launch flag for the route.
- Benchmark generation, OpenAI text and chat completions, and tokenization are outside the abort scope. Upstream IPC fields, `abort_message`, and `finished_reason` have no effect on this route.

### Pinned SGLang reference

SGLang commit `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8` defines `POST /abort_request` and an empty HTTP 200 response. It expands a batch string ID with the same unpadded suffix. With one tokenizer worker, a named abort needs a known exact ID before dispatch. Its scheduler then searches dispatched requests by prefix.

[HTTP route](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1727)<!-- prefix: "@app.post(\"/" --> · [Batch IDs](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/io_struct.py#L884)<!-- prefix: "def _normali" --> · [Tokenizer dispatch](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_manager.py#L2544)<!-- prefix: "def abort_re" --> · [Scheduler matching](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/io_struct.py#L2538)<!-- prefix: "def matches(" -->

The native server applies the exact-ID gate locally. It has no distributed tokenizer, IPC abort forwarding, or disaggregated abort control.

## Verification

Run the [CPU generation fixture](../tests/text/README.md#native-generate-on-cpu) with the checkpoint tokenizer and scripted decoder.

1. Check scalar, generated, and per-item IDs. Check that invalid types and duplicate live IDs return 400.
2. Fill the generation slots and HTTP queue. Abort an active stream and check its cumulative IDs, abort terminal, and single `[DONE]`.
3. Abort a fully queued request while the active stream is stalled. Check its zero-ID terminal and verify that a replacement can queue.
4. Abort one item in a waiting batch. Check that its live sibling can still run and that the aborted item never gets a lane.
5. Reuse a finished item's ID while its original batch has a live sibling. Finish the old batch and abort the replacement by that ID.
6. Run the [reader capacity checks](http-server.md#verification). Use stalled partial requests to exercise idle-socket replacement and the absolute read deadline.

Use the [standard launch procedure](../README.md#serve-token-ids-over-http) for release checks on a reserved B200. Compare the terminal result on the generation connection with the empty response on the control connection. The CPU fixture verifies HTTP and ownership behavior with scripted decoder steps.
