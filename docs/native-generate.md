# Native generation

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP transport | Validate the request framing and route `POST /generate` through the shared completion queue. | [Framing](../apps/server/http.cpp#L181)<!-- prefix: "if (request." --> · [Route](../apps/server/prefix_controls.cpp#L279)<!-- prefix: "void Connect" --> · [Admission](http-server.md#http-admission-and-shutdown) |
| Request parser | Validate one prompt and its generation options. | [Parser](../apps/server/text_request.cpp#L515)<!-- prefix: "GenerateRequ" --> |
| Completion worker | Use the checkpoint tokenizer and submit work to the batch scheduler. | [Worker](../apps/server/native_generate.cpp#L102)<!-- prefix: "void complet" --> |
| Response writer | Encode the result as JSON or cumulative SSE events. | [Response fields](../apps/server/native_generate.cpp#L56)<!-- prefix: "std::string " --> |

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

The completion worker parses the body after it claims a queue slot.
It encodes `text` or uses `input_ids` directly. Both inputs need the tokenizer to render the output.
The worker submits the prompt to the same batch service as text and chat completions.

### JSON and SSE

With `stream` unset or false, HTTP 200 carries one final JSON object. An SSE request opens [chunked HTTP 200 after decoder admission](http-server.md#completion-streams). Each successful `data:` event before `[DONE]` carries this object shape.

| Field | Meaning |
| --- | --- |
| `text` | Cumulative decoded text. The writer skips tokenizer special tokens and buffers incomplete UTF-8. |
| `output_ids` | Cumulative array of generated IDs. |
| `meta_info.id` | Server-assigned request ID. It stays constant across the response. |
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
| `text` | One nonempty UTF-8 string of at most 1,048,576 bytes. Supply exactly one non-null prompt field. |
| `input_ids` | One nonempty flat array of integer IDs in `[0, 248320)`. |
| `stream` | Boolean. Omission or null uses `false`. |
| `sampling_params` | One object. Omission or null uses the defaults below. |
| `cache_salt` | String of at most 256 UTF-8 bytes. Omission or null uses the empty salt. See [prefix reuse](prefix-cache.md#runtime-path). |

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
| Invalid headers, JSON, or request options | HTTP 400. |
| Missing length or media type | HTTP 411 for missing `Content-Length`. HTTP 415 for missing or unsupported `Content-Type`. |
| Oversized request | HTTP 413 for an oversized body. HTTP 431 for oversized headers. |
| Decoder capacity failure | HTTP 413 if the request exceeds total [resident capacity](batch-graphs.md#resident-memory-admission). HTTP 503 if capacity is temporarily unavailable. |
| Queue overload | HTTP 503. |
| Unexpected failure before SSE headers | HTTP 500. This includes a tokenizer that cannot load. |

Before SSE opens, errors use JSON with `error.message`.
After SSE opens, HTTP stays at 200. The worker attempts an SSE error event with `error.type: "server_error"`, `error.code: 500`, and `error.retryable: false`.
It ends the stream without `[DONE]`. [SSE error writer](../apps/server/native_generate.cpp#L132)<!-- prefix: "if (response" -->
After successful HTTP parsing, a method other than POST at `/generate` returns HTTP 404.
A disconnected client can cancel queued or active work through the [shared transport](http-server.md#http-admission-and-shutdown).

## Verification

Start the release server with its pinned checkpoint on a reserved B200. Run both examples from [Runtime path](#runtime-path).
Repeat each request with the other response mode. Keep its prompt and sampling options fixed.
Compare final text, output IDs, token counts, and finish reason across JSON and SSE.
Require every intermediate ID array to be a prefix of the final array.
Require exactly one final object and one `[DONE]` on a successful stream.

Repeat the ID request with `max_new_tokens: 0`. Require the zero-budget result in both response modes.
Use the first ID from the greedy request as an explicit stop ID. Require `stop`, the matched ID in `output_ids`, and empty text.
Send both prompt fields with `stream: true`. Require a JSON HTTP 400 before SSE opens.
Use the [shared transport checks](http-server.md#verification) for keepalives, cancellation, and failures after admission.
