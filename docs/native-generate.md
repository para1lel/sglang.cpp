# Native generation

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP transport | Validate the request framing and route `POST /generate` through the shared completion queue. | [Framing](../apps/server/http.cpp#L183)<!-- prefix: "if (request." --> · [Route](../apps/server/prefix_controls.cpp#L320)<!-- prefix: "void Connect" --> · [Admission](http-server.md#http-admission-and-shutdown) |
| Request parser | Check the request shape for ID registration, then prepare every prompt and its generation options. | [Shape](../apps/server/text_request.cpp#L680)<!-- prefix: "GenerateInpu" --> · [Prompts](../apps/server/text_request.cpp#L711)<!-- prefix: "std::vector\u003c" --> |
| Completion worker | Dispatch the prepared request to its response mode. | [Dispatch](../apps/server/native_generate.cpp#L251)<!-- prefix: "void complet" --> |
| Batch service | Submit items to available decoder lanes and collect results by input position. | [Batch entry](../apps/server/batch_service.cpp#L857)<!-- prefix: "BatchService" --> · [Result collection](../apps/server/batch_service.cpp#L610)<!-- prefix: "bool BatchSe" --> |
| Response writer | Encode final JSON or cumulative SSE. | [Result object](../apps/server/native_generate.cpp#L51)<!-- prefix: "std::string " --> · [Batch JSON](../apps/server/native_generate.cpp#L163)<!-- prefix: "void write_b" --> · [Batch SSE](../apps/server/native_generate.cpp#L193)<!-- prefix: "void stream_" --> |

## Runtime path

Start the server with the [standard launch procedure](../README.md#serve-token-ids-over-http).
The server uses its configured checkpoint and tokenizer. This route takes raw text or prompt IDs.
Use [chat completions](chat-serving.md) to render conversation history with the chat template.

Send a text prompt for a JSON response.

```bash
curl --fail-with-body http://127.0.0.1:17384/generate \
  -H 'Content-Type: application/json' \
  --data-binary '{"text":"The capital of France is","sampling_params":{"max_new_tokens":8,"temperature":0}}'
```

Send prompt IDs with `stream: true` for an SSE response.

```bash
curl --fail-with-body -N http://127.0.0.1:17384/generate \
  -H 'Content-Type: application/json' \
  --data-binary '{"input_ids":[220],"sampling_params":{"max_new_tokens":2,"temperature":0},"stream":true}'
```

Send an array of text prompts with one sampling object for all items.

```bash
curl --fail-with-body http://127.0.0.1:17384/generate \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "text": ["The capital of France is", "The capital of Spain is"],
    "sampling_params": {"max_new_tokens": 8, "temperature": 0}
  }'
```

Send nested prompt IDs with separate sampling objects and cache salts. Set `stream: true` to receive indexed SSE events on one connection.

```bash
curl --fail-with-body -N http://127.0.0.1:17384/generate \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "input_ids": [[220], [32, 220]],
    "sampling_params": [{"max_new_tokens": 0}, {"max_new_tokens": 2, "temperature": 0}],
    "cache_salt": ["", "example"],
    "stream": true
  }'
```

The reader parses the request shape and reserves its effective IDs when the queue has capacity.
It validates all items and encodes all text prompts before it submits any decoder work.
Both input forms use the checkpoint tokenizer to render generated IDs.

A batch occupies one HTTP queue entry and one completion worker. Its items share the [decoder scheduler](http-server.md#decode-scheduling) with other completions.
An item that decodes alone can use [score-backed singleton attention](singleton-gqa-decode.md).
The batch submits items to its worker's lane and can use available lanes without a reservation.
If a new completion reserves a borrowed lane, it can use the lane after the current item retires.
The worker submits remaining items as lanes become available. A batch can therefore contain more items than `--max-active`.
[Lane selection](../apps/server/batch_service.cpp#L557)<!-- prefix: "void BatchSe" -->

### JSON and SSE

With `stream` unset or false, HTTP 200 carries one final object for a single prompt or an array for a batch.
The batch response contains one final object per item in input order, even when items finish in a different order.
A one-item batch still returns a JSON array. Each item has its own ID and metadata.
With `stream: true`, one connection opens [chunked HTTP 200](http-server.md#completion-streams) after successful decoder admission or when all items have been aborted before admission.
A single prompt sends result events without `index`. Every batch result event includes a top-level `index`, the zero-based position of its prompt in the request.
Items can interleave and finish in any order. A one-item batch still sends indexed events.
[Indexed result](../apps/server/native_generate.cpp#L106)<!-- prefix: "std::string " --> · [Batch admission and polling](../apps/server/batch_service.cpp#L658)<!-- prefix: "bool BatchSe" -->

| Field | Meaning |
| --- | --- |
| `index` | Original prompt position, present on every batch result event. |
| `text` | Cumulative decoded text. The writer skips tokenizer special tokens and buffers incomplete UTF-8. |
| `output_ids` | Cumulative array of generated IDs. |
| `meta_info.id` | Effective item ID, provided by `rid` or generated by the server. It stays constant across that item's events. |
| `meta_info.prompt_tokens` | Number of prompt IDs. |
| `meta_info.completion_tokens` | Number of IDs in `output_ids`. |
| `meta_info.finish_reason` | `null` on intermediate events. The final value is `{"type":"length","length":N}`, `{"type":"stop","matched":ID}`, or an [abort result](native-abort.md#runtime-path). |
| `meta_info.cached_tokens` | Number of reused prompt IDs. Present on every batch result event and in every final object. Single-prompt intermediate events omit it. |

For a batch, keep separate state by `index` and replace that item's text and ID array with each event's values. Text can stay unchanged while IDs grow, and one event can contain several new IDs from a decoder step.
Each event repeats its item's full prefix, so streamed bytes can grow quadratically with output length.
The final object carries the budget-ending ID or the matched stop ID. The matched stop ID contributes to `completion_tokens`. Its decoded text is omitted, even when the ID is an explicit stop.
Zero budget returns empty text and IDs with `{"type":"length","length":0}`.
Each item sends one final object on success. After all items finish, the response sends a single `data: [DONE]` and closes.
The [keepalive timer](http-server.md#completion-streams) applies to the whole connection. Traffic for any item resets it.
Skip comments and measure each item's TTFT at its first event with output IDs. A zero-budget item has no first-token sample.
The completion worker writes outside the scheduler mutex. A blocked write can hold the batch's pending results while other ready completions advance.
[Batch result delivery](../apps/server/batch_service.cpp#L610)<!-- prefix: "bool BatchSe" -->

## Constraints and failure behavior

Send HTTP/1.1 with `Content-Length` and `Content-Type: application/json`. The media type accepts case variation and requires an exact value without parameters.
The shared reader enforces the [header and body limits](http-server.md#http-admission-and-shutdown).
Send one JSON object. The parser rejects duplicate keys, nesting beyond 32 containers, and floating-point range errors.

| Request field | Accepted values |
| --- | --- |
| `text` | A nonempty UTF-8 string or an array of such strings. Each string is at most 1,048,576 bytes. |
| `input_ids` | A nonempty flat array of IDs or an array of such arrays. Each ID is an integer in `[0, 248320)`. |
| `stream` | Boolean. Omission or null uses `false`. `true` enables SSE for a single prompt or batch. |
| `rid` | A string or, for a batch, one string per item in input order. A single batch string gets a zero-based suffix for each item. Omission or null generates IDs. See [abort and ID reuse](native-abort.md). |
| `sampling_params` | One object or, for a batch, one object per item in input order. A single object applies to every item. Omission or top-level null uses the defaults below. |
| `cache_salt` | One string or, for a batch, one string per item in input order. A single string applies to every item. Each string is at most 256 UTF-8 bytes. Omission or top-level null uses the empty salt. See [prefix reuse](prefix-cache.md#runtime-path). |

Supply exactly one non-null prompt field. A batch contains 1–32 items of the same prompt form.
If you send a parameter array, its length must equal the number of prompts. Each entry must have the type given in the table.
Missing sampling keys in each object use the native defaults.
The sum of batch output budgets must be at most 262,144 tokens. Each prompt and its own budget must fit the model context.
[Batch form](../apps/server/text_request.cpp#L700)<!-- prefix: "input.batch " --> · [Parameter arrays](../apps/server/text_request.cpp#L665)<!-- prefix: "const json::" --> · [Batch budget](../apps/server/text_request.cpp#L726)<!-- prefix: "int remainin" -->

| Sampling field | Accepted values and default |
| --- | --- |
| `max_new_tokens` | Nonnegative integer. Omission uses 128. Null is invalid. The prompt length plus this budget must fit 262,144 tokens. |
| `temperature`, `top_p`, `top_k`, `min_p` | Use the [shared sampling limits](text-serving.md#sampling). Defaults are `1`, `1`, `-1`, and `0`, respectively. |
| `sampling_seed` | Signed 64-bit integer, with no seed by default. |
| `n` | Integer `1`, the default. |
| `ignore_eos` | Boolean, default `false`. Set `true` to disable both model and explicit stop IDs. |
| `stop_token_ids` | Array of at most 248,320 integer IDs in `[0, 248320)`. Without explicit IDs, the model stops on `248044` or `248046` unless `ignore_eos` is true. |

Null uses the default for every sampling field except `max_new_tokens`. The server validates explicit stop IDs even with `ignore_eos: true`.
Client compatibility fields accept only disabled values.
Use `false` or null for `return_logprob` and `return_routed_experts`.
Use `-1` or null for `logprob_start_len`, and null for `lora_path`.
Other fields return HTTP 400, including `model`, `messages`, string stops, and output constraints.

| Result | Behavior |
| --- | --- |
| Invalid headers, JSON, request options, or duplicate live IDs | HTTP 400. |
| Missing length or media type | HTTP 411 for missing `Content-Length`. HTTP 415 for missing or unsupported `Content-Type`. |
| Oversized request | HTTP 413 for an oversized body. HTTP 431 for oversized headers. |
| Decoder capacity failure | HTTP 413 if the request exceeds total [resident capacity](batch-graphs.md#resident-memory-admission). HTTP 503 if capacity is temporarily unavailable. |
| Queue overload | HTTP 503. |
| Unexpected failure before response headers | HTTP 500. This includes a tokenizer that cannot load. |

Before response headers, an error returns the mapped HTTP status and one JSON body with `error.message`. A nonstreaming batch failure returns this error in place of the result array.
After SSE opens, HTTP stays at 200. The worker attempts one SSE error event with `error.message: "Native decoder failed"`, `error.type: "server_error"`, `error.code: 500`, and `error.retryable: false`.
The error event applies to the stream and has no item index. Previously delivered events remain visible, and the response closes without `[DONE]`.
On a generation failure, disconnect, or failed write, the worker discards unsubmitted items, cancels the active jobs still owned by the batch, and waits for their retirement before releasing the slot.
Decoder effects already published, such as prefix-cache entries, can remain.
The [cache reset exclusion](prefix-cache.md#native-http-controls) covers the whole claimed completion.
[Batch cancellation](../apps/server/batch_service.cpp#L536)<!-- prefix: "void BatchSe" --> · [HTTP error mapping](../apps/server/prefix_controls.cpp#L36)<!-- prefix: "void handle_" --> · [SSE error writer](../apps/server/native_generate.cpp#L118)<!-- prefix: "void write_s" -->
After successful HTTP parsing, a method other than POST at `/generate` returns HTTP 404.
A disconnected client can cancel queued or active work through the [shared transport](http-server.md#http-admission-and-shutdown).
Send [`POST /abort_request`](native-abort.md) from a second connection to cancel by ID.

## Verification

Start the release server with its pinned checkpoint on a reserved B200. Run the single-prompt examples from [Runtime path](#runtime-path).
Repeat each single-prompt request with the other response mode. Keep its prompt and sampling options fixed.
Compare final text, output IDs, token counts, and finish reason across JSON and SSE.
Require every intermediate ID array to be a prefix of the final array.
Require exactly one final object and one `[DONE]` on a successful stream.

Repeat the ID request with `max_new_tokens: 0`. Require the zero-budget result in both response modes.
Use the first ID from the greedy request as an explicit stop ID. Require `stop`, the matched ID in `output_ids`, and empty text.
Send both prompt fields with `stream: true`. Require a JSON HTTP 400 before SSE opens.
Use the [shared transport checks](http-server.md#verification) for keepalives, cancellation, and failures after admission.

Run both batch examples in JSON and SSE with `--max-active 1`. Repeat with `--max-active 2` to allow concurrent items.
Compare each terminal event with the corresponding final JSON object using the same greedy options and cache state. Event IDs are new for each request.
Check each index's cumulative ID prefixes, stable ID, terminal event, and token counts. Require one `[DONE]` after both items finish.
Require zero output IDs for the first nested-ID item. Repeat with a one-item batch and require a one-element JSON array and indexed SSE events.
Send a streaming batch with a mismatched sampling array or an invalid later prompt. Require JSON HTTP 400 before SSE opens.

Follow the [CPU fixture setup](../tests/text/README.md#native-generate-on-cpu) to exercise stream framing and lifecycle faults with a scripted decoder.
Inject a capacity failure before admission and check the mapped JSON error. Inject a failure after an item event and check HTTP 200, one SSE error, and closure without `[DONE]`.
Use a stalled item to check the 15-second comment and exclude it from TTFT. Disconnect during a batch with an unsubmitted item, check cancellation and retirement of its active jobs, and verify successful reuse by a replacement request.
