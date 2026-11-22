# Native generation

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP transport | Validate the request framing and route `POST /generate` through the shared completion queue. | [Framing](../apps/server/http.cpp#L183)<!-- prefix: "if (request." --> · [Route](../apps/server/prefix_controls.cpp#L337)<!-- prefix: "void Connect" --> · [Admission](http-server.md#http-admission-and-shutdown) |
| Request parser | Check the request shape for ID registration, then prepare every prompt and its generation options. | [Shape](../apps/server/text_request.cpp#L882)<!-- prefix: "GenerateInpu" --> · [Prompts](../apps/server/text_request.cpp#L919)<!-- prefix: "std::vector<" --> |
| Completion worker | Dispatch the prepared request to its response mode. | [Dispatch](../apps/server/native_generate.cpp#L370)<!-- prefix: "void complet" --> |
| Batch service | Submit items to available decoder lanes and collect results by input position. | [Batch entry](../apps/server/batch_service.cpp#L946)<!-- prefix: "BatchService" --> · [Result collection](../apps/server/batch_service.cpp#L640)<!-- prefix: "bool BatchSe" --> |
| Response writer | Encode final JSON or cumulative SSE. | [Result object](../apps/server/native_generate.cpp#L140)<!-- prefix: "std::string " --> · [Batch JSON](../apps/server/native_generate.cpp#L277)<!-- prefix: "void write_b" --> · [Batch SSE](../apps/server/native_generate.cpp#L309)<!-- prefix: "void stream_" --> |
| Text stops | Select stops before token publication and hold ambiguous text. | [Worker match](../include/sglang/server/text_stop.h#L156)<!-- prefix: "bool apply(D" --> · [Response frame](../apps/server/native_generate.cpp#L205)<!-- prefix: "std::string " --> |

## Runtime path

Use the [standard launch](../README.md#serve-token-ids-over-http) for raw text or prompt IDs and [chat completions](chat-serving.md) for conversation history.

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
[Lane selection](../apps/server/batch_service.cpp#L562)<!-- prefix: "void BatchSe" -->

### Parallel sampling

`sampling_params.n > 1` samples each prompt in any of the four input forms. Per-prompt option objects must agree on `n`. Each sample inherits options, cache salt and any explicit seed. With omitted or null seeds, each lane seeds independently.

The worker prepares each prompt once and submits samples together without warmup jobs. Each owns its decode state, output and cancellation.

Results are prompt-major. SSE uses `index = prompt_position * n + sample_position`, starting at zero. Every sample receives a fresh UUID in `meta_info.id`, including when a caller supplies `rid`. Use observed IDs for [named aborts](native-abort.md#runtime-path).

### JSON and SSE

HTTP 200 JSON returns one object for scalar `n: 1` and an ordered array for indexed requests, including one-item batches.
SSE opens [chunked HTTP 200](http-server.md#completion-streams) after admission or if all items were aborted while waiting. Indexed events can interleave and carry their zero-based result positions. Scalar `n: 1` omits `index`.
[Indexed result](../apps/server/native_generate.cpp#L205)<!-- prefix: "std::string " --> · [Batch admission and polling](../apps/server/batch_service.cpp#L695)<!-- prefix: "bool BatchSe" -->

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
Single-prompt streams can use [bounded decode output](http-server.md#bounded-decode-output) on an eligible cohort. Cumulative IDs, terminal metadata, and keepalives follow the same response contract.
[Batch result delivery](../apps/server/batch_service.cpp#L640)<!-- prefix: "bool BatchSe" -->

### Text stops

`sampling_params.stop` searches generated text including special token pieces, regardless of `skip_special_tokens`. A string is one stop, and a list gives the stops for one item. A single sampling object broadcasts that list to every batch item. Use a `sampling_params` array of objects to set different stops per item.

The worker checks each decoder step for a match. If several strings match within that step, the first matching string in configured order wins. The worker uses that string's first occurrence, including a match that began in an earlier step. A text match takes precedence over a token-ID stop or length finish in the same step. Text stops still apply when `ignore_eos` is true.
[Match order](../include/sglang/server/text_stop.h#L67)<!-- prefix: "std::optiona" --> · [Step finish](../include/sglang/server/text_stop.h#L173)<!-- prefix: "matched = ma" -->

For a text stop, the final object contains `{"type":"stop","matched":"…"}` with the exact configured string. IDs and usage include the token that completed it, even if the match starts inside that token. The worker drops any later MTP IDs before publishing and cancels the lane if it is still live. Other items keep running.
[Finish serialization](../apps/server/native_generate.cpp#L160)<!-- prefix: "if (type == " --> · [Clipping](../include/sglang/server/text_stop.h#L177)<!-- prefix: "const auto e" --> · [Lane retirement](../apps/server/batch_service.cpp#L459)<!-- prefix: "if (selected" -->

The response searches rendered `text` for the matched string. By default it trims from the first occurrence. Set `sampling_params.no_stop_trim` to `true` to keep text through the end of that occurrence. For example, stopping on `ll` in the token `hello` returns `hell`. If the matched string is absent after special token filtering, both settings return all rendered text through the matching token. IDs, usage, finish metadata, and optional token logprobs follow the same rules with either setting.

Set `sampling_params.skip_special_tokens` to `false` to show generated special tokens in `text`. A matched stop ID contributes visible text when `no_stop_trim` is true and its piece is visible. Prompt tokens stay outside the output and the stop matcher. Generated-token logprob text uses its independent piece decoding.

A split UTF-8 character enters the matcher after its last byte arrives. The decoder emits U+FFFD for invalid bytes or for an incomplete character when it flushes. Stop strings match the resulting UTF-8 text, including replacement characters. The writer holds trailing stop prefixes and text starting at a rendered occurrence until it knows which bytes to release. This keeps text cumulative even when hiding a special token joins visible pieces into a stop string. An event can add IDs while its text stays unchanged. When an item ends without a text match, the terminal event releases the held text.
[UTF-8 assembly](../src/utf8.cpp#L63)<!-- prefix: "std::string " --> · [UTF-8 flush](../src/utf8.cpp#L96)<!-- prefix: "std::string " --> · [Text release](../include/sglang/server/text_stop.h#L122)<!-- prefix: "std::string " -->

An unscored and unconstrained scalar prompt with `n: 1` and no text stops uses ID-based callbacks. Text stops, output constraints, and scores use step delivery with the same unindexed shape. An event can carry several MTP IDs, including on the terminal step. Eligible unindexed SSE can use bounded delivery. Indexed requests use ordinary scheduling.
[Dispatch](../apps/server/native_generate.cpp#L388)<!-- prefix: "} else if (!" --> · [Bounded delivery](http-server.md#bounded-decode-output)

### Token logprobs

Set top-level `return_logprob: true` and `sampling_params.temperature: 0`. The server requires MTP disabled. Every scored item uses unconstrained greedy decoding. A batch can mix scored and unscored items.
[Validation](../apps/server/text_request.cpp#L846)<!-- prefix: "void generat" --> · [MTP check](../apps/server/native_generate.cpp#L380)<!-- prefix: "if (scored &" -->

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
Native input scoring can restore a [matching checkpoint](prefix-cache.md#checkpoint-compatibility-and-resume) through the start offset. Prompt scores use the raw target distribution.
[Serialization](../apps/server/native_generate.cpp#L89)<!-- prefix: "void append_" --> · [Frame](../apps/server/native_generate.cpp#L178)<!-- prefix: "if (result &" -->

Each scored row adds an FP32 log-softmax reduction outside graph replay. It reads BF16 target logits or FP32 scores adjusted by [output penalties](output-penalties.md) or logit bias. The argmax supplies the selected ID. The score readback transfers the selected record and requested alternatives to pinned host memory. Scratch and readback allocate on first scored admission.
[Scoring](../src/decoder_batch.cpp#L603)<!-- prefix: "if (lane.opt" --> · [Readback](../src/text/decoder_logprobs.cu#L192)<!-- prefix: "cudaError_t " --> · [Allocation](../src/decoder_batch.cpp#L521)<!-- prefix: "if (top >= 0" -->

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
[Batch form](../apps/server/text_request.cpp#L902)<!-- prefix: "input.batch " --> · [Parameter arrays](../apps/server/text_request.cpp#L821)<!-- prefix: "const json::" --> · [Batch budget](../apps/server/text_request.cpp#L933)<!-- prefix: "int remainin" -->

| Sampling field | Accepted values and default |
| --- | --- |
| `max_new_tokens` | Nonnegative integer. Omission uses 128. Null is invalid. The prompt length plus this budget must fit 262,144 tokens. |
| `min_new_tokens` | [Minimum output](output-penalties.md#minimum-output), default zero. |
| `temperature`, `top_p`, `top_k`, `min_p` | Use the [shared sampling limits](text-serving.md#sampling). Defaults are `1`, `1`, `-1`, and `0`, respectively. |
| `sampling_seed` | Signed 64-bit integer, with no seed by default. |
| `frequency_penalty`, `presence_penalty` | [Output penalties](output-penalties.md). Omission or null uses zero. |
| `repetition_penalty` | [Output repetition penalty](output-penalties.md). Omission or null uses one. |
| `logit_bias` | Optional [per-token logit offsets](openai-logprobs.md#logit-bias). Each item uses its own sampling object. |
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
[Stop parser](../apps/server/text_request.cpp#L79)<!-- prefix: "std::vector<" --> · [Batch options](../apps/server/text_request.cpp#L929)<!-- prefix: "const auto b" -->
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
[Batch cancellation](../apps/server/batch_service.cpp#L541)<!-- prefix: "void BatchSe" --> · [HTTP error mapping](../apps/server/prefix_controls.cpp#L45)<!-- prefix: "void handle_" --> · [SSE error writer](../apps/server/native_generate.cpp#L231)<!-- prefix: "void write_s" -->
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

With MTP disabled, compare JSON and SSE using greedy scoring, top three, and both text settings. Check score prefixes and IDs. Repeat with top zero, zero budget, a stop ID, and mixed lanes. Verify full prompt scores and an interior suffix. Reject offsets past the prompt, requested-ID scoring, sampling, and MTP with HTTP 400.

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
