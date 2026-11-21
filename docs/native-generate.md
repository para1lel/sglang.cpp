# Native generation

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP transport | Validate the request framing and route `POST /generate` through the shared completion queue. | [Framing](../apps/server/http.cpp#L181)<!-- prefix: "if (request." --> · [Route](../apps/server/prefix_controls.cpp#L279)<!-- prefix: "void Connect" --> · [Admission](http-server.md#http-admission-and-shutdown) |
| Request parser | Validate and prepare every prompt and its generation options. | [Parser](../apps/server/text_request.cpp#L622)<!-- prefix: "GenerateRequ" --> |
| Completion worker | Dispatch the prepared request to the single-prompt or batch path. | [Dispatch](../apps/server/native_generate.cpp#L164)<!-- prefix: "void complet" --> |
| Batch service | Submit items to available decoder lanes and collect results by input position. | [Batch entry](../apps/server/batch_service.cpp#L715)<!-- prefix: "BatchService" --> · [Result collection](../apps/server/batch_service.cpp#L534)<!-- prefix: "bool BatchSe" --> |
| Response writer | Encode one result as JSON or cumulative SSE. Encode a batch as a JSON array. | [Result object](../apps/server/native_generate.cpp#L57)<!-- prefix: "std::string " --> · [Batch writer](../apps/server/native_generate.cpp#L135)<!-- prefix: "void complet" --> |

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

Send nested prompt IDs with separate sampling objects and cache salts.

```bash
curl --fail-with-body http://127.0.0.1:17384/generate \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "input_ids": [[220], [32, 220]],
    "sampling_params": [{"max_new_tokens": 0}, {"max_new_tokens": 2, "temperature": 0}],
    "cache_salt": ["", "example"]
  }'
```

The completion worker parses the body after it claims a queue slot.
It validates all items and encodes all text prompts before it submits any decoder work.
Both input forms use the checkpoint tokenizer to render generated IDs.

A batch occupies one HTTP queue entry and one completion worker. Its items share the decoder scheduler with other completions.
An item that decodes alone can use [score-backed singleton attention](singleton-gqa-decode.md).
The batch submits items to its worker's lane and can use available lanes without a reservation.
If a new completion reserves a borrowed lane, it can use the lane after the current item retires.
The worker submits remaining items as lanes become available. A batch can therefore contain more items than `--max-active`.
[Lane selection](../apps/server/batch_service.cpp#L517)<!-- prefix: "void BatchSe" -->

### JSON and SSE

With `stream` unset or false, HTTP 200 carries one final object for a single prompt or an array for a batch.
The batch response contains one final object per item in input order, even when items finish in a different order.
A one-item batch still returns a JSON array. Each item has its own ID and metadata.
For a single prompt, `stream: true` opens [chunked HTTP 200 after decoder admission](http-server.md#completion-streams).
Each successful `data:` event before `[DONE]` carries the result object.

| Field | Meaning |
| --- | --- |
| `text` | Cumulative decoded text. The writer skips tokenizer special tokens and buffers incomplete UTF-8. |
| `output_ids` | Cumulative array of generated IDs. |
| `meta_info.id` | Server-assigned ID for the result. It stays constant across the events of a stream. |
| `meta_info.prompt_tokens` | Number of prompt IDs. |
| `meta_info.completion_tokens` | Number of IDs in `output_ids`. |
| `meta_info.finish_reason` | `null` on intermediate events. The final value is `{"type":"length","length":N}` or `{"type":"stop","matched":ID}`. |
| `meta_info.cached_tokens` | Number of reused prompt IDs. Only the final object includes this field. |

Replace the previous text and ID array with each event's values. Text can stay unchanged when the ID array grows.
Since each event carries the full prefix, total streamed bytes grow quadratically with output length.
The final object carries the budget-ending ID or the matched stop ID. The matched stop ID contributes to `completion_tokens`. Its decoded text is omitted, even when the ID is an explicit stop.
Zero budget returns empty text and IDs with `{"type":"length","length":0}`.
After the final object, a successful SSE response sends `data: [DONE]` and closes.
Skip SSE comments and measure TTFT at the first event with output IDs.

## Constraints and failure behavior

Send HTTP/1.1 with `Content-Length` and `Content-Type: application/json`. The media type accepts case variation and requires an exact value without parameters.
The shared reader enforces the [header and body limits](http-server.md#http-admission-and-shutdown).
Send one JSON object. The parser rejects duplicate keys, nesting beyond 32 containers, and floating-point range errors.

| Request field | Accepted values |
| --- | --- |
| `text` | A nonempty UTF-8 string or an array of such strings. Each string is at most 1,048,576 bytes. |
| `input_ids` | A nonempty flat array of IDs or an array of such arrays. Each ID is an integer in `[0, 248320)`. |
| `stream` | Boolean. Omission or null uses `false`. `true` requires a text string or flat ID array. |
| `sampling_params` | One object or, for a batch, one object per item in input order. A single object applies to every item. Omission or top-level null uses the defaults below. |
| `cache_salt` | One string or, for a batch, one string per item in input order. A single string applies to every item. Each string is at most 256 UTF-8 bytes. Omission or top-level null uses the empty salt. See [prefix reuse](prefix-cache.md#runtime-path). |

Supply exactly one non-null prompt field. A batch contains 1–32 items of the same prompt form.
If you send a parameter array, its length must equal the number of prompts. Each entry must have the type given in the table.
Missing sampling keys in each object use the native defaults.
The sum of batch output budgets must be at most 262,144 tokens. Each prompt and its own budget must fit the model context.
[Batch form](../apps/server/text_request.cpp#L643)<!-- prefix: "requests.bat" --> · [Parameter arrays](../apps/server/text_request.cpp#L607)<!-- prefix: "const json::" --> · [Batch budget](../apps/server/text_request.cpp#L660)<!-- prefix: "int remainin" -->

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
| Invalid headers, JSON, or request options | HTTP 400. This includes `stream: true` with a batch. |
| Missing length or media type | HTTP 411 for missing `Content-Length`. HTTP 415 for missing or unsupported `Content-Type`. |
| Oversized request | HTTP 413 for an oversized body. HTTP 431 for oversized headers. |
| Decoder capacity failure | HTTP 413 if the request exceeds total [resident capacity](batch-graphs.md#resident-memory-admission). HTTP 503 if capacity is temporarily unavailable. |
| Queue overload | HTTP 503. |
| Unexpected failure before response headers | HTTP 500. This includes a tokenizer that cannot load. |

Before response headers, errors use JSON with `error.message`.
If a batch item fails during generation, the batch returns one error response with the mapped HTTP status.
On a generation failure or disconnect, the worker discards unsubmitted items and waits for its active jobs to retire.
It cancels only jobs still owned by that batch. A disconnect abandons the response.
The client receives no partial result array. Decoder effects already published, such as prefix-cache entries, can remain.
The [cache reset exclusion](prefix-cache.md#native-http-controls) covers the whole claimed completion.
[Batch cancellation](../apps/server/batch_service.cpp#L496)<!-- prefix: "void BatchSe" --> · [HTTP error mapping](../apps/server/prefix_controls.cpp#L34)<!-- prefix: "void handle_" -->

After SSE opens, HTTP stays at 200. The worker attempts an SSE error event with `error.type: "server_error"`, `error.code: 500`, and `error.retryable: false`.
It ends the stream without `[DONE]`. [SSE error writer](../apps/server/native_generate.cpp#L128)<!-- prefix: "if (response" -->
After successful HTTP parsing, a method other than POST at `/generate` returns HTTP 404.
A disconnected client can cancel queued or active work through the [shared transport](http-server.md#http-admission-and-shutdown).

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

Run both batch examples with `--max-active 1` so each request must reuse one decoder lane for two items.
Require a two-element array in input order. Compare each greedy item with its single-prompt request under the same options.
Require the first nested-ID result to contain zero output IDs. Repeat with one batch item and require a one-element array.
Send a batch with a mismatched sampling array, an invalid later prompt, or `stream: true`. Require one JSON HTTP 400 for each.
During failure checks, inject a capacity failure after an earlier item completes. Require one mapped HTTP error and verify lane reuse.
Disconnect a client during a batch with a pending item. Check retirement of its active jobs and successful admission of a replacement.
