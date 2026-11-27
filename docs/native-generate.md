# Native generation

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP transport | Validate the framing and route `POST` and `PUT /generate` through the shared completion queue. | [Framing](../apps/server/http.cpp#L199)<!-- prefix: "if ((request" --> · [Route](../apps/server/prefix_controls.cpp#L617)<!-- prefix: "void Connect" --> · [Admission](http-server.md#http-admission-and-shutdown) |
| Request parser | Check the request shape for ID registration, then prepare every prompt and its generation options. | [Shape](../apps/server/text_request.cpp#L786)<!-- prefix: "GenerateInpu" --> · [Prompts](../apps/server/text_request.cpp#L823)<!-- prefix: "std::vector<" --> |
| Completion worker | Dispatch the prepared request to its response mode. | [Dispatch](../apps/server/native_generate.cpp#L395)<!-- prefix: "void complet" --> |
| Batch service | Submit items to available decoder lanes and collect results by input position. | [Batch entry](../apps/server/batch_service.cpp#L929)<!-- prefix: "BatchService" --> · [Result collection](../apps/server/batch_service.cpp#L573)<!-- prefix: "bool BatchSe" --> |
| Response writer | Encode final JSON or cumulative SSE. | [Result object](../apps/server/native_generate.cpp#L144)<!-- prefix: "std::string " --> · [Batch JSON](../apps/server/native_generate.cpp#L294)<!-- prefix: "void write_b" --> · [Batch SSE](../apps/server/native_generate.cpp#L330)<!-- prefix: "void stream_" --> |
| Text stops | Match bounded candidate IDs before commit and hold ambiguous text. | [Worker match](../apps/server/detail/text_stop.cpp#L127)<!-- prefix: "int TokenTex" --> · [Response frame](../apps/server/native_generate.cpp#L232)<!-- prefix: "std::string " --> |

## Runtime path

[Launch the server](../README.md#serve-token-ids-over-http). For history, use [chat](chat-serving.md) or [native sessions](serving/native-sessions.md).

`POST` and `PUT` accept the same request fields, validation, queue admission, and JSON or SSE response behavior. [Native route matcher](../apps/server/http.cpp#L388)<!-- prefix: "GenerateRout" --> · [Pinned Python methods](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L963)<!-- prefix: "@app.api_rou" -->

Send a text prompt for a JSON response.

```bash
curl --fail-with-body http://127.0.0.1:17384/generate \
  -H 'Content-Type: application/json' \
  --data-binary '{"text":"The capital of France is","sampling_params":{"max_new_tokens":8,"temperature":0}}'
```

Send prompt IDs with `stream: true` for an SSE response.

```bash
curl --fail-with-body -N -X PUT http://127.0.0.1:17384/generate \
  -H 'Content-Type: application/json' \
  --data-binary '{"input_ids":[220],"sampling_params":{"max_new_tokens":2,"temperature":0},"stream":true}'
```

Broadcast one sampling object across text prompts.

```bash
curl --fail-with-body http://127.0.0.1:17384/generate \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "text": ["The capital of France is", "The capital of Spain is"],
    "sampling_params": {"max_new_tokens": 8, "temperature": 0, "stop": ["\n", "."]}
  }'
```

Stream nested prompt IDs with per-prompt options and salts.

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

The reader checks shape and reserves IDs when the queue has capacity. The worker validates every item and encodes text before decoder admission.

A batch shares one HTTP queue entry, worker and [decoder scheduler](http-server.md#decode-scheduling).
An item that decodes alone can use [score-backed singleton attention](singleton-gqa-decode.md).
Multi-item batches submit in input order, scanning from lane zero. Items beyond `--max-active` wait.
A one-item request waits for its worker's lane even if another lane is free.
[Lane selection](../apps/server/batch_service.cpp#L486)<!-- prefix: "void BatchSe" -->

### Parallel sampling

`sampling_params.n > 1` samples each prompt in any of the four input forms. Per-prompt option objects must agree on `n`. Each sample inherits options, cache salt and any explicit seed. With omitted or null seeds, each lane seeds independently.

The worker prepares each prompt once and submits samples together without warmup jobs. Each owns its decode state, output and cancellation.

Results are prompt-major. SSE uses `index = prompt_position * n + sample_position`, starting at zero. Every sample receives a fresh UUID in `meta_info.id`, including when a caller supplies `rid`. Use observed IDs for [named aborts](native-abort.md#runtime-path).

### JSON and SSE

HTTP 200 JSON returns one object for scalar `n: 1` and an ordered array for indexed requests, including one-item batches.
SSE opens [chunked HTTP 200](http-server.md#completion-streams) after admission or if all items were aborted while waiting. Indexed events can interleave and carry their zero-based result positions. Scalar `n: 1` omits `index`.
[Indexed result](../apps/server/native_generate.cpp#L232)<!-- prefix: "std::string " --> · [Batch admission and polling](../apps/server/batch_service.cpp#L628)<!-- prefix: "bool BatchSe" -->

| Field | Meaning |
| --- | --- |
| `index` | Expanded item position, present on every indexed result event. |
| `text` | Cumulative decoded text. The writer skips tokenizer special tokens and buffers incomplete UTF-8. |
| `output_ids` | Cumulative array of generated IDs. |
| `meta_info.id` | Effective item ID, provided by `rid` or generated by the server. It stays constant across that item's events. |
| `meta_info.prompt_tokens` | Number of prompt IDs. |
| `meta_info.completion_tokens` | Number of IDs in `output_ids`. |
| `meta_info.finish_reason` | `null` on intermediate events. The final value records `length`, `stop` with a string or integer `matched`, or an [abort result](native-abort.md#runtime-path). |
| `meta_info.cached_tokens` | Number of reused prompt IDs. Present on every batch result event and in every final object. Single-prompt intermediate events omit it. |

Keep state by `index`. Replace text and IDs with each event. Text can stay unchanged while IDs grow. Cumulative events can use quadratically many bytes.
The budget finish is `{"type":"length","length":N}`. A token-ID stop gives `{"type":"stop","matched":ID}`. Its final ID counts toward `completion_tokens`. The writer omits its decoded text by default.
Zero budget returns empty text and IDs with `{"type":"length","length":0}`.
Successful items send one terminal object. The stream sends `data: [DONE]` once and closes.
Any item traffic resets the connection's [keepalive timer](http-server.md#completion-streams).
Measure TTFT at each item's first event with output IDs. Zero-budget items have no first-token sample.
The completion worker writes outside the scheduler mutex. A blocked write can hold the batch's pending results while other ready completions advance.
Eligible unindexed streams can use [bounded decode output](http-server.md#bounded-decode-output). Cumulative IDs, terminal metadata, and keepalives follow the same response contract.
[Batch result delivery](../apps/server/batch_service.cpp#L573)<!-- prefix: "bool BatchSe" -->

### Text stops

`sampling_params.stop` searches generated text including special token pieces, regardless of `skip_special_tokens`. A string is one stop, and a list gives the stops for one item. A single sampling object broadcasts that list to every batch item. Use a `sampling_params` array of objects to set different stops per item.

The worker checks the IDs eligible to commit in the current decoder step. If several stops match, configured string order wins across the step. The worker uses that string's first occurrence, which can start in an earlier step. A text match takes precedence over a token-ID stop or length finish in that step. Text stops also apply when `ignore_eos` is true.
[Match order](../apps/server/detail/text_stop.cpp#L127)<!-- prefix: "int TokenTex" --> · [Worker binding](../apps/server/batch_service.cpp#L525)<!-- prefix: "job->options" -->

The final object contains `{"type":"stop","matched":"…"}` with the configured string. IDs and usage include the token that completes it, even if the match starts inside that token. The decoder ends the lane at that ID before later candidate IDs can commit. [Checkpoint coverage](prefix-cache.md#checkpoint-boundaries) follows processed target state. Other items keep running.
[Finish serialization](../apps/server/native_generate.cpp#L164)<!-- prefix: "if (type == " --> · [Commit cutoff](../apps/server/detail/text_stop.cpp#L136)<!-- prefix: "const auto e" -->

The response searches rendered `text` for the matched string. By default it trims from the first occurrence. Set `sampling_params.no_stop_trim` to `true` to keep text through the end of that occurrence. For example, stopping on `ll` in the token `hello` returns `hell`. If the matched string is absent after special token filtering, both settings return all rendered text through the matching token. IDs, usage, finish metadata, and optional token logprobs follow the same rules with either setting.

Set `sampling_params.skip_special_tokens` to `false` to show generated special tokens in `text`. A matched stop ID contributes visible text when `no_stop_trim` is true and its piece is visible. Prompt tokens stay outside the output and the stop matcher. Generated-token logprob text uses its independent piece decoding.

A split UTF-8 character enters the matcher after its last byte arrives. The decoder emits U+FFFD for invalid bytes or for an incomplete character when it flushes. Stop strings match the resulting UTF-8 text, including replacement characters. The writer holds trailing stop prefixes and text starting at a rendered occurrence until it knows which bytes to release. This keeps text cumulative even when hiding a special token joins visible pieces into a stop string. An event can add IDs while its text stays unchanged. When an item ends without a text match, the terminal event releases the held text.
[UTF-8 assembly](../src/utf8.cpp#L77)<!-- prefix: "std::string " --> · [UTF-8 flush](../src/utf8.cpp#L110)<!-- prefix: "std::string " --> · [Text release](../apps/server/detail/text_stop.cpp#L102)<!-- prefix: "std::string " -->

An unscored and unconstrained scalar prompt with `n: 1` and no text stops uses ID-based callbacks. Text stops, output constraints, and scores use step delivery with the same unindexed shape. A step-delivered event can carry several MTP IDs, including on the terminal step. Positive-MTP buffering requires the per-token path. Unindexed step streams can use the K0 cohort rule. Indexed requests use ordinary scheduling.
[Dispatch](../apps/server/native_generate.cpp#L408)<!-- prefix: "} else if (!" --> · [Bounded delivery](http-server.md#bounded-decode-output)

### Token logprobs

Set top-level `return_logprob: true` and `sampling_params.temperature: 0`. Scoring uses unconstrained greedy decoding. Batches can mix scored and unscored items. See the [server MTP policy](mtp.md#runtime-path) for scored items.
[Validation](../apps/server/text_request.cpp#L746)<!-- prefix: "void generat" -->

| Top-level field | Accepted values |
| --- | --- |
| `return_logprob` | Boolean, with omission or null using `false`. |
| `logprob_start_len` | `-1` or null for output only. `0` through prompt length for input scores from that offset. Ignored if `return_logprob` is false. |
| `top_logprobs_num` | Integer `0`–`20`, with omission or null using `0`. |
| `return_text_in_logprobs` | Request-wide boolean, with omission or null using `false`. |
| `token_ids_logprob` | Null or omission. |

For the first three fields, a scalar broadcasts. Arrays require a prompt array and `n: 1`.
`meta_info.output_token_logprobs` contains a `[logprob, token_id, text]` triple per output ID. `output_token_logprobs_length` counts them.
Text defaults to null. When enabled, each piece decodes independently, preserving special tokens and replacing invalid UTF-8.
`input_token_logprobs` starts with `[null, token_id, text]` at the offset, then scores later tokens from their preceding context. An offset at prompt length yields empty input arrays.
A positive top count adds ranked triples in `input_top_logprobs` and `output_top_logprobs`. The first input entry is null. Ties rank by ascending ID.
SSE repeats completed input scores and accumulates output scores. An early abort yields empty input arrays. Zero output yields empty output arrays. Unscored items omit score fields.
Input scoring uses the raw target distribution. It can restore a [matching checkpoint](prefix-cache.md#checkpoint-compatibility-and-resume) through the start offset and capture scores as the remaining prompt rows run.
[Serialization](../apps/server/native_generate.cpp#L91)<!-- prefix: "void append_" --> · [Frame](../apps/server/native_generate.cpp#L182)<!-- prefix: "if (result &" -->

Each scored row adds an FP32 log-softmax reduction outside graph replay. It reads BF16 target logits or FP32 scores adjusted by [output penalties](output-penalties.md) or logit bias. The argmax supplies the selected ID. The score readback transfers the selected record and requested alternatives to pinned host memory. Scratch and readback allocate on first scored admission.
[Scoring](../src/decoder_batch.cpp#L252)<!-- prefix: "if (lane.opt" --> · [Readback](../src/text/decoder_logprobs.cu#L192)<!-- prefix: "cudaError_t " --> · [Allocation](../src/decoder_batch.cpp#L181)<!-- prefix: "if (top \u003e= 0" -->

## Constraints and failure behavior

Send HTTP/1.1 with `Content-Length` and `Content-Type: application/json`. The media type accepts case variation and requires an exact value without parameters.
The shared reader enforces the [header and body limits](http-server.md#http-admission-and-shutdown).
Send one JSON object. The parser rejects duplicate keys, nesting beyond 32 containers, and floating-point range errors.

| Request field | Accepted values |
| --- | --- |
| `text` | A nonempty UTF-8 string or an array of such strings. Each string is at most 1,048,576 bytes. |
| `input_ids` | A nonempty flat array of IDs or an array of such arrays. Each ID is an integer in `[0, 248320)`. |
| `stream` | Boolean. Omission or null uses `false`. `true` enables SSE for a single prompt or batch. |
| `rid` | A string or one string per prompt for indexed requests. Omission or null generates IDs. See [abort and ID reuse](native-abort.md). |
| `sampling_params` | One object or one object per prompt for indexed requests. Omission or top-level null uses the defaults below. |
| `cache_salt` | One string or one string per prompt for indexed requests. Each is at most 256 UTF-8 bytes. Omission or null uses the empty salt. See [prefix reuse](prefix-cache.md#runtime-path). |

Supply exactly one non-null prompt field. A prompt array uses one prompt form. An indexed request has a prompt array, `n > 1`, or both.
Parameter arrays have one entry per original prompt. Missing sampling keys use the defaults.
The expanded item count is at most 32. The sum of per-prompt output budgets multiplied by `n` is at most 262,144 tokens. Each prompt plus its own budget must fit the model context. The route validates these limits before decoder admission.
[Batch form](../apps/server/text_request.cpp#L806)<!-- prefix: "input.batch " --> · [Parameter arrays](../apps/server/native_generate.cpp#L798)<!-- prefix: "const json::" --> · [Batch budget](../apps/server/text_request.cpp#L838)<!-- prefix: "int remainin" -->

| Sampling field | Accepted values and default |
| --- | --- |
| `max_new_tokens` | Nonnegative integer. Omission uses 128. Null is invalid. The prompt length plus this budget must fit 262,144 tokens. |
| `min_new_tokens` | [Minimum output](output-penalties.md#minimum-output), default zero. |
| `temperature`, `top_p`, `top_k`, `min_p` | Use the [shared sampling limits](text-serving.md#sampling). Defaults are `1`, `1`, `-1`, and `0`, respectively. |
| `sampling_seed` | Signed 64-bit integer, with no seed by default. |
| `frequency_penalty`, `presence_penalty` | [Output penalties](output-penalties.md). Omission or null uses zero. |
| `repetition_penalty` | [Output repetition penalty](output-penalties.md). Omission or null uses one. |
| `logit_bias` | Optional [per-token logit offsets](serving/openai-logprobs.md#logit-bias). Each item uses its own sampling object. |
| `n` | Integer `1`–`32`, default `1`. The expanded limits apply. |
| `ignore_eos` | Boolean, default `false`. Set `true` to disable both model and explicit stop IDs. |
| `stop_token_ids` | Array of at most 248,320 integer IDs in `[0, 248320)`. Without explicit IDs, the model stops on `248044` or `248046` unless `ignore_eos` is true. |
| `stop` | One UTF-8 string or an array of up to 32 UTF-8 strings. Omission, null, and `[]` disable text matching. The empty string `""` is a stop pattern. |
| `no_stop_trim` | Boolean, default `false`. Applies per item. See [text stops](#text-stops). |
| `skip_special_tokens` | Boolean, default `true`. Applies per item. See [text stops](#text-stops). |
| `regex` | Optional [whole-output regular expression](output-constraints.md#native-generate). |
| `json_schema` | Optional [JSON-encoded schema string](output-constraints.md#native-json-schema). |
| `ebnf` | Optional [finite-choice grammar string](output-constraints.md#native-ebnf). |

Null uses the sampling defaults except for `max_new_tokens`. Explicit stop IDs are always validated.
`stop` belongs inside `sampling_params`. A nested list or a non-string list element returns HTTP 400.
The empty stop matches at the first generated step if selected by the configured order. It retains the first generated ID and returns empty text. Zero output budget still yields a length finish.
[Stop parser](../apps/server/text_request.cpp#L78)<!-- prefix: "std::vector<" --> · [Batch options](../apps/server/text_request.cpp#L833)<!-- prefix: "const auto b" -->
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
Cache checkpoints the daemon already acknowledged can remain available.
The [cache reset exclusion](prefix-cache.md#native-http-controls) covers the whole claimed completion.
[Batch cancellation](../apps/server/batch_service.cpp#L463)<!-- prefix: "void BatchSe" --> · [HTTP error mapping](../apps/server/prefix_controls.cpp#L178)<!-- prefix: "void handle_" --> · [SSE error writer](../apps/server/native_generate.cpp#L243)<!-- prefix: "void write_s" -->

The route matches the decoded `/generate` path and ignores its query. `/generate/` follows the [shared redirect rules](http-server.md#request-targets-and-redirects). After HTTP parsing, methods other than `POST` and `PUT` return HTTP 405 with `Allow: POST, PUT` and `{"detail":"Method Not Allowed"}`. For `HEAD`, the server sends only the headers. [Native route matcher](../apps/server/http.cpp#L388)<!-- prefix: "GenerateRout" --> · [Method response](../apps/server/http.cpp#L562)<!-- prefix: "void write_m" -->

A disconnected client can cancel queued or active work through the [shared transport](http-server.md#http-admission-and-shutdown).
Send [`POST /abort_request`](native-abort.md) from a second connection to cancel by ID.

## Verification

Run the [single-prompt and batch examples](#runtime-path) on a reserved B200 with the pinned release checkpoint. Compare JSON and SSE for both methods. Check text, finish reason, token counts, cumulative IDs, one final event per item, and one `[DONE]` after all items finish. Run the batch cases at `--max-active 1` and `2`; check stable IDs, per-index counts, one-item array and SSE shapes, and zero IDs for the first nested-ID item.

Check zero budget and an explicit stop at the first greedy ID. The stop retains that ID and returns empty text. Require HTTP 400 for malformed JSON, conflicting prompts, a mismatched sampling array, or an invalid later item. SSE validation errors must precede headers. With PUT, check 411 for missing length and 415 for unsupported media type. Confirm HTTP 405 and `Allow: POST, PUT` with `curl -I http://127.0.0.1:17384/generate`.

For generated scores, compare greedy JSON and SSE at K0 and positive depth. Check top three with both text settings, top zero, full and suffix prompt scores, score prefixes and IDs, zero budget, stop IDs, and mixed lanes. Require HTTP 400 for invalid offsets, requested-ID scoring, or active sampling. In managed Pixi, run this independent full-vocabulary log-softmax comparison on the reserved B200.

```bash
checkpoint=/data0/shared/Qwen3.8-27B-FP8
xmake f -P . -o build -p linux -a x86_64 -m release --generated-logprobs-check=y
xmake build -P . generated-logprobs-check
./build/bin/generated-logprobs-check "$checkpoint"
```

Build the [CPU fixture](../tests/text/README.md#native-generate-on-cpu), then run both suites in the same environment from the repository root. Require zero exit status for all three checks.

```bash
out="$PWD/artifacts/native-generate-cpu"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
CUDA_VISIBLE_DEVICES='' /data1/shared/dongwu.chen/.pixi/envs/default/bin/python -B \
    tests/text/test_native_generated_logprobs_cpu.py \
    --binary "$out/server" --checkpoint "$checkpoint" --output "$out/logprob-evidence"
CUDA_VISIBLE_DEVICES='' python3 -B tests/text/test_native_text_stops_cpu.py \
    --binary "$out/server" --checkpoint "$checkpoint" --output "$out/text-stop-evidence"
```

The score fixture also checks token text, mixed opt-in, abort, and greedy `top_k: 1` at positive temperature. For text stops, cover single tokens, token boundaries, and UTF-8 pieces. Reverse competing matches in one step, and check hidden partials, incomplete UTF-8 at length, discarded speculative IDs and usage, indexed stops, broadcast with `ignore_eos: true`, and lane reuse. Reject nested stop arrays. Check [transport admission, keepalives, and cancellation](http-server.md#verification). Require the mapped JSON error for pre-admission failure. After an item event, require HTTP 200, one SSE error, and closure without `[DONE]`. Disconnect with an unsubmitted item and confirm cancellation, retirement, and reuse.

[Full verification appendix](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/docs/native-generate.md#verification)
