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

The scalar request shapes above also accept prompt arrays. For indexed batches, supply `text: ["A", "B"]` or `input_ids: [[220], [32, 220]]`. Broadcast one `sampling_params` object or supply one per prompt, and optionally supply per-prompt `cache_salt` and `rid`. [Copyable batch JSON and SSE examples](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/native-generate.md#runtime-path)

The worker validates and encodes every item before decoder admission. Multi-item batches submit in input order from lane zero, waiting at `--max-active`. The [shared scheduler](http-server.md#http-admission-and-shutdown) handles request IDs and lane selection. A lone decoding item can use [score-backed attention](singleton-gqa-decode.md). [Lane selection](../apps/server/batch_service.cpp#L486)<!-- prefix: "void BatchSe" -->

### Parallel sampling

`sampling_params.n > 1` samples each prompt in any of the four input forms. Per-prompt option objects must agree on `n`. Each sample inherits options, cache salt and any explicit seed. With omitted or null seeds, each lane seeds independently.

The worker prepares each prompt once and submits samples together without warmup jobs. Each owns its decode state, output and cancellation.

Results are prompt-major. SSE uses `index = prompt_position * n + sample_position`, starting at zero. Every sample receives a fresh UUID in `meta_info.id`, including when a caller supplies `rid`. Use observed IDs for [named aborts](native-abort.md#runtime-path).

### JSON and SSE

JSON 200 returns an object for scalar `n: 1` and an ordered array for every indexed request. [SSE admission](http-server.md#completion-streams) opens a chunked HTTP 200. Indexed events can interleave and carry zero-based result positions; scalar `n: 1` omits `index`. [Indexed result](../apps/server/native_generate.cpp#L232)<!-- prefix: "std::string " --> · [Batch admission and polling](../apps/server/batch_service.cpp#L628)<!-- prefix: "bool BatchSe" -->

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

Replace each item's text and IDs on every event. Text can stay fixed as IDs grow, and cumulative frames can use quadratic bytes. Each item has one terminal object. Budget finish is `{"type":"length","length":N}`, including zero. A token-ID stop returns `{"type":"stop","matched":ID}`, counts that ID, and hides its text by default. Use the shared [end marker, keepalive and TTFT rules](http-server.md#completion-streams). The writer runs outside the scheduler mutex. [Batch result delivery](../apps/server/batch_service.cpp#L573)<!-- prefix: "bool BatchSe" -->

### Text stops

Text stops inspect generated pieces including specials, regardless of `skip_special_tokens` or `ignore_eos`. A string or list applies per item, with one sampling object broadcasting. The matcher checks the current step's committable IDs in configured stop order, then chooses that stop's first occurrence, which can start in an earlier step. A match supersedes stop IDs and length in that step. Commit ends at the completing token, retaining its ID, usage, and scores. Other items continue. [Match order](../apps/server/detail/text_stop.cpp#L127)<!-- prefix: "int TokenTex" --> · [Checkpoint coverage](prefix-cache.md#checkpoint-boundaries) · [Commit cutoff](../apps/server/detail/text_stop.cpp#L136)<!-- prefix: "const auto e" -->

The final finish is `{"type":"stop","matched":"…"}`. Rendered text trims at its first occurrence by default, or through it with `no_stop_trim: true`. If hiding specials removed the match, keep rendered text through the matching token. `skip_special_tokens: false` exposes those pieces. `no_stop_trim: true` also retains a matched ID’s text if its piece is visible. Prompt IDs never enter matching. Token logprob text decodes independently. Partial UTF-8 waits for its last byte, and invalid or flushed incomplete pieces become U+FFFD. Hold partial stop prefixes until safe so cumulative text never retracts, then release them at an unmatched finish. [UTF-8 assembly](../src/utf8.cpp#L77)<!-- prefix: "std::string " --> · [UTF-8 flush](../src/utf8.cpp#L110)<!-- prefix: "std::string " --> · [Text release](../apps/server/detail/text_stop.cpp#L102)<!-- prefix: "std::string " -->

Unscored, unconstrained scalar `n: 1` without text stops uses per-token callbacks. Indexed requests or scalar text stops, scores, and constraints use steps, which can emit multiple MTP IDs. See [output buffering and K0 exceptions](http-server.md#bounded-decode-output). [Dispatch](../apps/server/native_generate.cpp#L408)<!-- prefix: "} else if (!" -->

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

The first three fields accept a broadcast scalar or an array for prompt arrays at `n: 1`. Output scores are `meta_info.output_token_logprobs` triples `[logprob, token_id, text]` with `output_token_logprobs_length`. Text is null unless requested, then decodes pieces independently with specials and UTF-8 replacement. `meta_info.input_token_logprobs` triples begin at the offset with a null score (empty at prompt length). Positive top counts add `input_top_logprobs` and `output_top_logprobs`, with null for the first input entry and ascending IDs on ties. SSE repeats input and accumulates output scores. Early abort and zero output leave the corresponding arrays empty. Unscored items omit score fields.

Input scores use raw target distributions and can restore a [matching checkpoint](prefix-cache.md#checkpoint-compatibility-and-resume) through the score offset. Output scoring runs FP32 log-softmax outside replay over BF16 row logits or FP32 scores adjusted by penalties or logit bias. Selected records and alternatives use pinned readback allocated at first scored admission. [Serialization](../apps/server/native_generate.cpp#L91)<!-- prefix: "void append_" --> · [Frame](../apps/server/native_generate.cpp#L182)<!-- prefix: "if (result &" --> · [Readback](../src/text/decoder_logprobs.cu#L192)<!-- prefix: "cudaError_t " -->

## Constraints and failure behavior

Use [HTTP/1.1 JSON framing and input limits](http-server.md#request-targets-and-redirects). Framing returns 411 for missing length or 415 for missing or unsupported media type. Send one object under the [shared JSON validation rules](text-serving.md#send-a-request).

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

Use the [shared status and SSE rules](http-server.md#completion-streams). A pre-header failure substitutes JSON with `error.message` for the entire result. A tokenizer load failure returns 500. A post-header failure attempts one stream-wide error with no index, `error.message: "Native decoder failed"`, `error.type: "server_error"`, `error.code: 500`, and `error.retryable: false`. Already sent events remain visible. Generation failure, disconnect, or write failure drops unsubmitted items and cancels owned jobs through retirement. Acknowledged cache checkpoints can survive. The [reset exclusion](prefix-cache.md#native-http-controls) covers the complete HTTP claim. [Batch cancellation](../apps/server/batch_service.cpp#L463)<!-- prefix: "void BatchSe" --> · [HTTP error mapping](../apps/server/prefix_controls.cpp#L178)<!-- prefix: "void handle_" --> · [SSE error writer](../apps/server/native_generate.cpp#L243)<!-- prefix: "void write_s" -->

The route matches the decoded `/generate` path and ignores its query. `/generate/` follows the [shared redirect rules](http-server.md#request-targets-and-redirects). After HTTP parsing, methods other than `POST` and `PUT` return HTTP 405 with `Allow: POST, PUT` and `{"detail":"Method Not Allowed"}`. For `HEAD`, the server sends only the headers. [Native route matcher](../apps/server/http.cpp#L388)<!-- prefix: "GenerateRout" --> · [Method response](../apps/server/http.cpp#L562)<!-- prefix: "void write_m" -->

Cancel queued or active work with a disconnect or [`POST /abort_request`](native-abort.md) from a second connection.

## Verification

On a reserved B200, verify POST and PUT with JSON/SSE at one and two lanes. Check indexed and singleton shape, stable IDs, cumulative fields, matching text, finishes and usage, one terminal per item, and one successful `[DONE]`. Include zero budget, first-ID stop, malformed JSON, both prompt fields, wrong option array length and invalid later items before headers. Check PUT 411/415 and HEAD 405 with `Allow: POST, PUT`. Apply the [stream end and failure checks](http-server.md#verification).

Compare scores at K0 and positive MTP for top zero/three, both text settings, prompt offsets, mixed lanes, empty output and stop IDs. Reject invalid offsets, requested-ID scoring, and active sampling. Run the full-vocabulary oracle in managed Pixi.

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

The CPU score fixture also covers token text, mixed opt-in, abort, and greedy `top_k: 1` at positive temperature. The stop fixture must cover token and UTF-8 boundaries, reversed matches, hidden partials and incomplete UTF-8 at length, speculative cutoff with exact usage, indexed and broadcast matching with `ignore_eos: true`, nested-list rejection and lane reuse. Check [transport admission and errors](http-server.md#verification). After an item, require HTTP 200, one native stream error and closure without `[DONE]`. Disconnect with an unsubmitted item and require cancellation, retirement and reuse.

[Full verification appendix](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/docs/native-generate.md#verification)
