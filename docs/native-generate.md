# Native generation

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP transport | Validate the request framing and route `POST /generate` through the shared completion queue. | [Framing](../apps/server/http.cpp#L183)<!-- prefix: "if (request." --> · [Route](../apps/server/prefix_controls.cpp#L337)<!-- prefix: "void Connect" --> · [Admission](http-server.md#http-admission-and-shutdown) |
| Request parser | Check the request shape for ID registration, then prepare every prompt and its generation options. | [Shape](../apps/server/text_request.cpp#L883)<!-- prefix: "GenerateInpu" --> · [Prompts](../apps/server/text_request.cpp#L914)<!-- prefix: "std::vector\u003c" --> |
| Completion worker | Dispatch the prepared request to its response mode. | [Dispatch](../apps/server/native_generate.cpp#L339)<!-- prefix: "void complet" --> |
| Batch service | Submit items to available decoder lanes and collect results by input position. | [Batch entry](../apps/server/batch_service.cpp#L944)<!-- prefix: "BatchService" --> · [Result collection](../apps/server/batch_service.cpp#L638)<!-- prefix: "bool BatchSe" --> |
| Response writer | Encode final JSON or cumulative SSE. | [Result object](../apps/server/native_generate.cpp#L110)<!-- prefix: "std::string " --> · [Batch JSON](../apps/server/native_generate.cpp#L246)<!-- prefix: "void write_b" --> · [Batch SSE](../apps/server/native_generate.cpp#L278)<!-- prefix: "void stream_" --> |
| Text stops | Select the matching stop before ID publication and hold ambiguous text in the response writer. | [Worker match](../include/sglang/server/text_stop.h#L150)<!-- prefix: "bool apply(D" --> · [Response frame](../apps/server/native_generate.cpp#L175)<!-- prefix: "std::string " --> |

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
    "sampling_params": {"max_new_tokens": 8, "temperature": 0, "stop": ["\n", "."]}
  }'
```

Send nested prompt IDs with separate sampling objects and cache salts. Set `stream: true` to receive indexed SSE events on one connection.

```bash
curl --fail-with-body -N http://127.0.0.1:17384/generate \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "input_ids": [[220], [32, 220]],
    "sampling_params": [{"max_new_tokens": 0}, {"max_new_tokens": 2, "temperature": 0, "stop": "."}],
    "cache_salt": ["", "example"],
    "stream": true
  }'
```

The reader parses the request shape and reserves its effective IDs when the queue has capacity.
It validates all items and encodes all text prompts before it submits any decoder work.
Both input forms use the checkpoint tokenizer to render generated IDs.

A batch occupies one HTTP queue entry and one completion worker. Its items share the [decoder scheduler](http-server.md#decode-scheduling) with other completions.
An item that decodes alone can use [score-backed singleton attention](singleton-gqa-decode.md).
Multi-item batches submit in input order to available lanes, scanning from lane zero.
See [indexed refill priority](http-server.md#decode-scheduling).
A scalar request or one-item batch uses only its worker's lane. If another batch occupies it, the request waits there even when another lane is free.
A batch can contain more items than `--max-active`.
[Lane selection](../apps/server/batch_service.cpp#L561)<!-- prefix: "void BatchSe" -->

### JSON and SSE

With `stream` unset or false, HTTP 200 carries one final object for a single prompt or an array for a batch.
The batch response contains one final object per item in input order, even when items finish in a different order.
A one-item batch still returns a JSON array. Each item has its own ID and metadata.
With `stream: true`, one connection opens [chunked HTTP 200](http-server.md#completion-streams) after successful decoder admission or when all items have been aborted before admission.
A single prompt sends result events without `index`. Every batch result event includes a top-level `index`, the zero-based position of its prompt in the request.
Items can interleave and finish in any order. A one-item batch still sends indexed events.
[Indexed result](../apps/server/native_generate.cpp#L175)<!-- prefix: "std::string " --> · [Batch admission and polling](../apps/server/batch_service.cpp#L693)<!-- prefix: "bool BatchSe" -->

| Field | Meaning |
| --- | --- |
| `index` | Original prompt position, present on every batch result event. |
| `text` | Cumulative decoded text. The writer skips tokenizer special tokens and buffers incomplete UTF-8. |
| `output_ids` | Cumulative array of generated IDs. |
| `meta_info.id` | Effective item ID, provided by `rid` or generated by the server. It stays constant across that item's events. |
| `meta_info.prompt_tokens` | Number of prompt IDs. |
| `meta_info.completion_tokens` | Number of IDs in `output_ids`. |
| `meta_info.finish_reason` | `null` on intermediate events. The final value records `length`, `stop` with a string or integer `matched`, or an [abort result](native-abort.md#runtime-path). |
| `meta_info.cached_tokens` | Number of reused prompt IDs. Present on every batch result event and in every final object. Single-prompt intermediate events omit it. |

For a batch, keep separate state by `index` and replace that item's text and ID array with each event's values. Text can stay unchanged while IDs grow, and one event can contain several new IDs from a decoder step.
Each event repeats its item's full prefix, so streamed bytes can grow quadratically with output length.
The budget finish is `{"type":"length","length":N}`. A token-ID stop gives `{"type":"stop","matched":ID}`. Its final ID counts toward `completion_tokens`, and the writer omits its decoded text.
Zero budget returns empty text and IDs with `{"type":"length","length":0}`.
Each item sends one final object on success. After all items finish, the response sends a single `data: [DONE]` and closes.
The [keepalive timer](http-server.md#completion-streams) applies to the whole connection. Traffic for any item resets it.
Skip comments and measure each item's TTFT at its first event with output IDs. A zero-budget item has no first-token sample.
The completion worker writes outside the scheduler mutex. A blocked write can hold the batch's pending results while other ready completions advance.
Single-prompt streams can use [bounded decode output](http-server.md#bounded-decode-output) on an eligible cohort. Cumulative IDs, terminal metadata, and keepalives follow the same response contract.
[Batch result delivery](../apps/server/batch_service.cpp#L638)<!-- prefix: "bool BatchSe" -->

### Text stops

`sampling_params.stop` searches generated text. A string is one stop, and a list gives the stops for one item. A single sampling object broadcasts that list to every batch item. Use a `sampling_params` array of objects to set different stops per item.

The worker checks each decoder step for a match. If several strings match within that step, the first matching string in configured order wins. The worker uses that string's first occurrence, including a match that began in an earlier step. A text match takes precedence over a token-ID stop or length finish in the same step. Text stops still apply when `ignore_eos` is true.
[Match order](../include/sglang/server/text_stop.h#L67)<!-- prefix: "std::optiona" --> · [Step finish](../include/sglang/server/text_stop.h#L166)<!-- prefix: "matched = ma" -->

For a text stop, the final object contains `{"type":"stop","matched":"…"}` with the exact configured string. It contains text before the match and IDs through the token that completed it. That token stays in `output_ids` and counts toward `completion_tokens`, even if the match begins inside it. If MTP returned later IDs in the same step, the worker excludes those IDs and their usage before publication. The worker cancels the lane if it is still live. Other items keep running.
[Finish serialization](../apps/server/native_generate.cpp#L130)<!-- prefix: "if (type == " --> · [Clipping](../include/sglang/server/text_stop.h#L170)<!-- prefix: "const auto e" --> · [Lane retirement](../apps/server/batch_service.cpp#L458)<!-- prefix: "if (selected" -->

The tokenizer skips special tokens and can yield byte fallback pieces. A split UTF-8 character enters the matcher after its last byte arrives. The decoder of these bytes emits U+FFFD for invalid bytes or for an incomplete character when it flushes. Stop strings match the resulting UTF-8 text, including replacement characters. The writer holds any trailing text that could complete a stop in a later step. An event can add IDs while its text stays unchanged. When an item ends without a text match, the terminal event releases the held text.
[UTF-8 assembly](../src/utf8.cpp#L63)<!-- prefix: "std::string " --> · [UTF-8 flush](../src/utf8.cpp#L96)<!-- prefix: "std::string " --> · [Text release](../include/sglang/server/text_stop.h#L118)<!-- prefix: "std::string " -->

An unscored and unconstrained single prompt without text stops uses ID-based callbacks. Text stops, output constraints, and scores use step delivery with the same unindexed shape. An event can carry several MTP IDs, including on the terminal step. Eligible single-prompt SSE can use bounded delivery. Indexed batches, including one-item lists, use ordinary scheduling.
[Dispatch](../apps/server/native_generate.cpp#L357)<!-- prefix: "} else if (!" --> · [Bounded delivery](http-server.md#bounded-decode-output)

### Generated-token logprobs

Set top-level `return_logprob: true` and `sampling_params.temperature: 0`. The server requires MTP disabled. Every scored item uses unconstrained greedy decoding. A batch can mix scored and unscored items.
[Validation](../apps/server/text_request.cpp#L850)<!-- prefix: "void generat" --> · [MTP check](../apps/server/native_generate.cpp#L349)<!-- prefix: "if (scored &" -->

| Top-level field | Accepted values |
| --- | --- |
| `return_logprob` | Boolean, with omission or null using `false`. |
| `logprob_start_len` | `-1` or null, including omission. Scores cover generated tokens only. |
| `top_logprobs_num` | Integer `0`–`20`, with omission or null using `0`. |
| `return_text_in_logprobs` | Request-wide boolean, with omission or null using `false`. |
| `token_ids_logprob` | Null or omission. |

For the first three fields, a scalar broadcasts and an array supplies one value per batch item in input order. Scalar prompts take scalar values.
Each scored object includes `meta_info.output_token_logprobs`, with one `[logprob, token_id, text]` triple per output ID, and `output_token_logprobs_length`.
The text element is null by default. With text enabled, the tokenizer decodes each piece independently, preserves special-token pieces, and replaces invalid or incomplete UTF-8.
When the top count is positive, `output_top_logprobs` contains one ranked array of triples per output ID. Equal scores rank by ascending ID. `input_token_logprobs` and, when present, `input_top_logprobs` are empty.
Scores accumulate in SSE alongside the IDs. Stop and abort results include scores for the IDs they retain, even if display text was trimmed. Zero output yields empty arrays and count zero. Unscored items omit these fields.
[Serialization](../apps/server/native_generate.cpp#L53)<!-- prefix: "void append_" --> · [Frame](../apps/server/native_generate.cpp#L148)<!-- prefix: "if (result &" -->

Each scored row adds an FP32 log-softmax reduction outside graph replay. It reads BF16 target logits or FP32 scores adjusted by [output penalties](output-penalties.md) or logit bias. The argmax supplies the selected ID. The score readback transfers the selected record and requested alternatives to pinned host memory. Scratch and readback allocate on first scored admission.
[Scoring](../src/decoder_batch.cpp#L611)<!-- prefix: "if (lane.opt" --> · [Readback](../src/text/decoder_logprobs.cu#L190)<!-- prefix: "cudaError_t " --> · [Allocation](../src/decoder_batch.cpp#L529)<!-- prefix: "if (top \u003e= 0" -->

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
[Batch form](../apps/server/text_request.cpp#L903)<!-- prefix: "input.batch " --> · [Parameter arrays](../apps/server/text_request.cpp#L825)<!-- prefix: "const json::" --> · [Batch budget](../apps/server/text_request.cpp#L926)<!-- prefix: "int remainin" -->

| Sampling field | Accepted values and default |
| --- | --- |
| `max_new_tokens` | Nonnegative integer. Omission uses 128. Null is invalid. The prompt length plus this budget must fit 262,144 tokens. |
| `min_new_tokens` | [Minimum output](output-penalties.md#minimum-output), default zero. |
| `temperature`, `top_p`, `top_k`, `min_p` | Use the [shared sampling limits](text-serving.md#sampling). Defaults are `1`, `1`, `-1`, and `0`, respectively. |
| `sampling_seed` | Signed 64-bit integer, with no seed by default. |
| `frequency_penalty`, `presence_penalty` | [Output penalties](output-penalties.md). Omission or null uses zero. |
| `repetition_penalty` | [Output repetition penalty](output-penalties.md). Omission or null uses one. |
| `logit_bias` | Optional [per-token logit offsets](openai-logprobs.md#logit-bias). Each item uses its own sampling object. |
| `n` | Integer `1`, the default. |
| `ignore_eos` | Boolean, default `false`. Set `true` to disable both model and explicit stop IDs. |
| `stop_token_ids` | Array of at most 248,320 integer IDs in `[0, 248320)`. Without explicit IDs, the model stops on `248044` or `248046` unless `ignore_eos` is true. |
| `stop` | One UTF-8 string or an array of up to 32 UTF-8 strings. Omission, null, and `[]` disable text matching. The empty string `""` is a stop pattern. |
| `regex` | Optional [whole-output regular expression](output-constraints.md#native-generate). |
| `json_schema` | Optional [JSON-encoded schema string](output-constraints.md#native-json-schema). |
| `ebnf` | Optional [finite-choice grammar string](output-constraints.md#native-ebnf). |

Null uses the sampling defaults except for `max_new_tokens`. Explicit stop IDs are always validated.
`stop` belongs inside `sampling_params`. A nested list or a non-string list element returns HTTP 400.
The empty stop matches at the first generated step if selected by the configured order. It retains the first generated ID and returns empty text. Zero output budget still yields a length finish.
[Stop parser](../apps/server/text_request.cpp#L79)<!-- prefix: "std::vector\u003c" --> · [Batch options](../apps/server/text_request.cpp#L923)<!-- prefix: "const auto b" -->
Client compatibility fields accept only disabled values.
Use `false` or null for `return_routed_experts`, and null for `lora_path`.
Other fields return HTTP 400, including `model`, `messages`, and top-level sampling fields.

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
[Batch cancellation](../apps/server/batch_service.cpp#L540)<!-- prefix: "void BatchSe" --> · [HTTP error mapping](../apps/server/prefix_controls.cpp#L45)<!-- prefix: "void handle_" --> · [SSE error writer](../apps/server/native_generate.cpp#L200)<!-- prefix: "void write_s" -->
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

With MTP disabled, repeat the ID request with `temperature: 0`, `return_logprob: true`, `top_logprobs_num: 3`, and both text flag values. Compare JSON with final SSE. Check score prefixes and ID alignment. Repeat with top zero, zero budget, a matching stop ID, and a mixed batch. Require HTTP 400 for prompt rescoring, requested-ID scoring, active sampling, or server MTP enabled.

In the managed Pixi environment, build the score check from the repository root. Run it on a reserved B200.

```bash
checkpoint=/data0/shared/Qwen3.8-27B-FP8
xmake f -P . -o build -p linux -a x86_64 -m release --generated-logprobs-check=y
xmake build -P . generated-logprobs-check
./build/bin/generated-logprobs-check "$checkpoint"
```

Require exit status zero for reducer and checkpoint cases. On captured target rows, compare selected and ranked scores against an independent full-vocabulary log-softmax. [Build target](../xmake.lua#L354)<!-- prefix: "target(\"gene" --> · [Checkpoint check](../tests/text/generated_logprobs_check.cpp#L248)<!-- prefix: "int main(int" -->

Build the [CPU fixture](../tests/text/README.md#native-generate-on-cpu). Run the score suite against that binary.

```bash
out="$PWD/artifacts/native-generate-cpu"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
CUDA_VISIBLE_DEVICES='' /data1/shared/dongwu.chen/.pixi/envs/default/bin/python -B \
    tests/text/test_native_generated_logprobs_cpu.py \
    --binary "$out/server" --checkpoint "$checkpoint" --output "$out/logprob-evidence"
```

Require exit status zero. The suite checks cumulative scores, mixed opt-in, token text, stops, abort, and validation. It checks that `top_k: 1` stays greedy with positive temperature. [CPU suite](../tests/text/test_native_generated_logprobs_cpu.py#L1)<!-- prefix: "\"\"\"Check nat" -->

Run both batch examples in JSON and SSE with `--max-active 1`. Repeat with `--max-active 2` to allow concurrent items.
Compare each terminal event with the corresponding final JSON object using the same greedy options and cache state. Event IDs are new for each request.
Check each index's cumulative ID prefixes, stable ID, terminal event, and token counts. Require one `[DONE]` after both items finish.
Require zero output IDs for the first nested-ID item. Repeat with a one-item batch and require a one-element JSON array and indexed SSE events.
Send a streaming batch with a mismatched sampling array or an invalid later prompt. Require JSON HTTP 400 before SSE opens.

Build the [CPU fixture](../tests/text/README.md#native-generate-on-cpu), then run the text-stop suite against that same binary from the repository root.

```bash
out="$PWD/artifacts/native-generate-cpu"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
CUDA_VISIBLE_DEVICES='' python3 -B tests/text/test_native_text_stops_cpu.py \
    --binary "$out/server" --checkpoint "$checkpoint" --output "$out/text-stop-evidence"
```

Script a stop inside one token, across two tokens, and across the byte pieces of a UTF-8 character. Compare JSON with the SSE terminal. Check that earlier events never expose a pending stop prefix. Verify the matched string, trimmed text, retained IDs, and completion count.
Script competing patterns in one decoder step, then reverse their configured order. Test an unmatched partial suffix and an incomplete UTF-8 character at length. Require the suffix and replacement character in the appropriate terminal results.
Run two indexed items with different text stops. Repeat with a broadcast object, `ignore_eos: true`, and a nested stop array. Check item independence and HTTP 400 for the invalid array.
With a scripted MTP step, return additional IDs after the match. Require those IDs to be absent from the result and usage. Run a step that reaches the length budget without a match. Require all retained text and IDs in the terminal event and compare it with JSON. Verify lane reuse after the stop and after a client abort.
Inject a capacity failure before admission and check the mapped JSON error. Inject a failure after an item event and check HTTP 200, one SSE error, and closure without `[DONE]`.
Use a stalled item to check the 15-second comment and exclude it from TTFT. Disconnect during a batch with an unsubmitted item, check cancellation and retirement of its active jobs, and verify successful reuse by a replacement request.
