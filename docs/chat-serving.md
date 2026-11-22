# Native chat completions

Send `POST /v1/chat/completions` with `model: "Qwen3.8-27B-FP8"` and a nonempty `messages` array. Follow the [server launch procedure](../README.md#serve-token-ids-over-http).

```bash
curl --fail-with-body -N http://127.0.0.1:17384/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen3.8-27B-FP8","messages":[{"role":"user","content":"What is the capital of Japan?"}],"stream":true,"stream_options":{"include_usage":true}}'
```

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP ingress | Select the chat route and validate the request. | [Chat route](../apps/server/prefix_controls.cpp#L164)<!-- prefix: "bool chat_ro" --> · [HTTP admission](http-server.md#http-admission-and-shutdown) · [Request parser](../apps/server/text_request.cpp#L961)<!-- prefix: "TextRequest " --> |
| Chat template | Verify checkpoint assets and render ordered messages and tools as a UTF-8 prompt. | [Asset checks](../apps/server/chat_template.cpp#L417)<!-- prefix: "ChatTemplate" --> · [Renderer](../apps/server/chat_template.cpp#L538)<!-- prefix: "std::string " --> |
| Media handler | Validate attachments, prepare patches, then encode features for the multimodal decoder. | [Attachment parser](../apps/server/chat_media.cpp#L341)<!-- prefix: "std::optiona" --> · [Vision completion](../apps/server/vision_completion.cpp#L61)<!-- prefix: "DecoderResul" --> |
| Chat output parsers | Apply the request options and parse reasoning and tool calls. | [Request options](../apps/server/text_request.cpp#L639)<!-- prefix: "void set_cha" --> · [Parser](../apps/server/chat_output.cpp#L799)<!-- prefix: "bool ChatOut" --> |
| Completion writer | Serialize committed output and own the UTF-8 state for one request. | [Token callback](../apps/server/text_completion.cpp#L524)<!-- prefix: "bool TextRes" --> · [Stream end](../apps/server/text_completion.cpp#L535)<!-- prefix: "void TextRes" --> |

## Runtime path

The server parses chat once as ordered JSON and returns HTTP 400 for floating-point range errors. It preserves oversized integer `top_k` values for validation and clamping. The native renderer uses the pinned checkpoint's template rules and literals. The [shared tokenizer](text-serving.md#tokenizer-assets) encodes the rendered prompt without adding special tokens.

Text chat submits the IDs to `BatchService::generate` with the connection's worker slot. Rendering, tokenization, and response writing run outside CUDA graphs. Eligible decode groups can mix text chat, [text completions](text-serving.md), and [benchmark ID requests](../benchmarks/streaming/README.md#wire-contract). [Batch dispatch](../apps/server/completions.cpp#L210)<!-- prefix: "void dispatc" -->

For media chat, the parser substitutes template placeholders in attachment order. The CPU prepares patches and expands the rendered IDs. The connection worker encodes vision features before it submits language generation to the [multimodal decoder](multimodal-decoder.md). Zero-budget or already cancelled requests skip vision encoding. Usage counts the expanded prompt IDs. Vision encoding stays outside the target step cap.

### Read JSON and SSE

The JSON response has one assistant choice at `choices[0].message`. Without output parsers, `content` holds the generated text, including reasoning and tool markup. Input tools and reasoning history still affect the rendered prompt.

JSON uses `object: "chat.completion"`. SSE uses `object: "chat.completion.chunk"`. Both carry `id`, `model`, and `created` in Unix seconds. All chunks in a stream share these values. The response ID starts with `chatcmpl-`.

On a successful stream, the first SSE data chunk carries `delta.role: "assistant"` and a null finish reason. [Headers and keepalive comments](http-server.md#completion-streams) can arrive earlier. Join later `delta.content` strings in order. The terminal choice has `delta: {}` and a finish reason. When requested, a usage event with `choices: []` follows the terminal choice. The stream ends with `[DONE]`.

JSON responses always include usage. `prompt_tokens` counts rendered prompt IDs. `completion_tokens` counts committed output IDs. `total_tokens` sums both counts. `prompt_tokens_details.cached_tokens` counts [restored prompt IDs](prefix-cache.md#runtime-path). Raw chat uses the [same stop and finish rules as text completions](text-serving.md#read-the-response). Parsed chat can also report `tool_calls` as the finish reason.

With `return_token_ids: true`, each committed ID appears once in `token_ids`, even when it produces empty text. A UTF-8 tail flush can carry text with zero IDs. In text chat, `prompt_token_ids` appears in the JSON choice or the first diagnostic SSE choice after the role. Media chat puts the expanded `prompt_token_ids` at the top level of the JSON response or the initial assistant SSE chunk. Media usage counts expanded prompt IDs. [Diagnostic choice](../apps/server/text_completion.cpp#L376)<!-- prefix: "std::string " --> · [Role delta](../apps/server/text_completion.cpp#L443)<!-- prefix: "bool TextRes" -->

### Qwen reasoning and tool calls

Start the [release server](../README.md#serve-token-ids-over-http) on a reserved B200 with both parser flags. Both flags default to off.

To parse supplied text, use [`POST /parse_function_call`](serving/parse-function-call.md).

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
  ./build/bin/sglang-server --checkpoint /data0/shared/Qwen3.8-27B-FP8 \
  --device 0 --port 17384 --max-active 1 --mtp-steps 0 --fp8-prefill exact \
  --prefix-cache-entries 0 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

Output parsing applies to text chat with free output or a [tool-call constraint](output-constraints.md#tool-call-constraints). Media chat and chat with `response_format`, `regex`, or `ebnf` keep generated text in `content`.

With the reasoning parser active, `separate_reasoning: true` places reasoning in `reasoning_content`. Set it to `false` to keep reasoning in `content`. `enable_thinking` controls the rendered prompt. The parser starts in the reasoning section when the prompt leaves `<think>` open. It also recognizes `<think>` at the start of generated text. `</think>` ends the section. So does `<tool_call>` when tool parsing is active and the request advertises functions. Set `stream_reasoning: false` to buffer reasoning until the section ends or the request completes. Both output options are booleans and default to `true`.

With the tool parser active, complete calls for advertised functions become `tool_calls`. In unconstrained output, ordinary text and invalid wrappers stay in `content`. The parser can recover at a later wrapper. Call-shaped text inside a complete string parameter stays part of the argument. [Tool frame scanner](../apps/server/chat_output.cpp#L682)<!-- prefix: "std::size_t " -->

In JSON, read `message.reasoning_content` and `message.tool_calls`. With calls and empty content, `message.content` is `null`. In SSE, join the `delta.reasoning_content` strings. Group the `delta.tool_calls` entries by zero-based `index`. The first delta gives the call's `id`, `type: "function"`, `function.name`, and empty arguments. Append the next delta's `function.arguments`. In both formats, parse the argument string as a JSON object. A stop with a complete call reports `tool_calls`. A token-limit cut without a stop reports `length` and can carry a call with incomplete arguments. Text stops match generated text before parsing.

Send each request twice to get a JSON response and an SSE stream. Expect `391` as the first answer, separate from the reasoning. Expect a `get_weather` call with arguments `{"city":"Paris"}` from the second request.

```bash
chat_url=http://127.0.0.1:17384/v1/chat/completions
for stream in false true; do
  curl --fail-with-body -NsS "$chat_url" -H 'Content-Type: application/json' \
    --data-binary @- <<JSON
{"model":"Qwen3.8-27B-FP8","temperature":0.0,"seed":0,"max_tokens":128,"stream":$stream,
 "messages":[{"role":"user","content":"Compute 17 \u00d7 23. Give just the final number."}]}
JSON
  printf '\n'
  curl --fail-with-body -NsS "$chat_url" -H 'Content-Type: application/json' \
    --data-binary @- <<JSON
{"model":"Qwen3.8-27B-FP8","temperature":0.0,"seed":0,"max_tokens":64,"stream":$stream,
 "messages":[{"role":"user","content":"Use the get_weather tool to look up the weather in Paris. Call the tool now."}],
 "enable_thinking":false,"tool_choice":"auto",
 "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city.",
   "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}
JSON
  printf '\n'
done
```

## Constraints and failure behavior

### Messages and tools

- Use `system`, `developer`, `root`, `user`, `assistant`, or `tool` roles. The renderer treats `developer` and `root` as `system`. A system message must be first. Include a user query distinct from tool responses.
- Use a string or text parts for text content. The renderer also accepts null or omitted content. Empty image and video placeholders can render template tokens outside the system message. Follow the attachment contract for actual media.
- Supply function schemas in `tools`. The renderer collects top-level tools first, then tools on `system`, `developer`, or `root` messages. Advertised tools need `type: "function"` and unique names of 1–64 UTF-8 bytes. Names exclude angle brackets, carriage return, and line feed. [Tool selection](../apps/server/chat_template.cpp#L496)<!-- prefix: "ChatTools ch" -->
- Supply earlier assistant `tool_calls` as an array. Put the `name` and `arguments` in a `function` object or directly in the call. If supplied, `type` must be `"function"`. Arguments can be objects or strings that encode JSON objects. A `tool` message supplies a tool response through its `content`.
- Use `reasoning_content`, `thinking`, or `reasoning` for earlier assistant reasoning. The renderer uses the first non-null field in that order.

### Tool choice

| `tool_choice` | Behavior |
| --- | --- |
| `"auto"`, null, or omitted | Advertise supplied functions. Allow prose and optional calls. An advertised `function.strict: true` activates the tool grammar when parsing is available. |
| `"none"` | Disable tool advertisement and extraction. Keep earlier assistant calls and tool responses in the prompt. Reasoning separation can still run. |
| `"required"` | Require a call frame for an advertised function. |
| `{"type":"function","function":{"name":"get_weather"}}` | Advertise only the selected function and require a single call frame for it. |

`parallel_tool_calls` is a boolean and defaults to `true`. With `false`, JSON and SSE expose at most one call. Auto with only non-strict tools leaves later wrappers in `content` after the first accepted call. Named choice always allows one call. Required and named choices need an eligible tool, text-only chat, and the `qwen3_coder` parser. They return HTTP 400 when a prerequisite is missing or `response_format`, `regex`, or `ebnf` is set. Active tool constraints require the model EOS stop set and reject nonempty text stops. [Tool admission](../apps/server/text_request.cpp#L610)<!-- prefix: "void check_t" --> · [Parser dispatch](../apps/server/chat_output.cpp#L799)<!-- prefix: "bool ChatOut" -->

The parser accepts an object schema for `function.parameters`, null, or omission. It reads `properties`, `required`, and `additionalProperties`. It removes up to one line feed from each end of a value. It converts scalars and JSON collections. A strict string parameter with the XML value `null` produces the JSON string `"null"`. A strict null parameter produces JSON null. In a loose tool, bare `null` in any letter case produces JSON null, and declared type alternatives are tried in order for other values. Loose tools allow extra keys by default. Validate loose arguments in the application. [Argument conversion](../apps/server/chat_output.cpp#L406)<!-- prefix: "void ToolFra" --> · [Strict schema limits](output-constraints.md#tool-call-constraints)

### Request options

| Field | Accepted values and effect |
| --- | --- |
| `max_tokens`, `max_completion_tokens` | Nonnegative integer or null. A non-null `max_completion_tokens` takes priority in either field order. |
| `min_tokens` | [Minimum output](output-penalties.md#minimum-output), default zero. |
| `n` | Omit or set to `1`. |
| `temperature`, `top_p`, `top_k`, `min_p`, `seed` | Use the [shared sampling controls](text-serving.md#sampling). A non-null value overrides its default. |
| `frequency_penalty`, `presence_penalty` | [Output penalties](output-penalties.md). |
| `repetition_penalty` | [Output repetition penalty](output-penalties.md). |
| `ebnf` | Use a [finite-choice constraint](text-serving.md#finite-choice-constraints) for the generated content. |
| `regex` | Use a [whole-output regular expression](output-constraints.md#regular-expressions) for the generated content. |
| `response_format` | Use a [bounded JSON Schema](output-constraints.md#json-schema) for the generated content. |
| `cache_salt` | Optional [prefix cache key](prefix-cache.md#runtime-path) component. |
| `stream`, `return_token_ids`, `ignore_eos`, `stop_token_ids`, `stop` | Use the [shared text options](text-serving.md#send-a-request). |
| `stream_options` | With streaming enabled, accepts one boolean member, `include_usage`. The default omits the usage event. |
| `logprobs`, `top_logprobs` | Optional [generated-token scores](serving/openai-logprobs.md#chat-completions). |
| `logit_bias` | Optional [per-token logit offsets](serving/openai-logprobs.md#logit-bias). |
| `add_generation_prompt`, `add_vision_id` | Booleans. HTTP defaults are `true` and `false`, respectively. |
| `enable_thinking`, `preserve_thinking` | Booleans that default to `true`. They control rendering of the generation prompt and assistant history. |
| `separate_reasoning`, `stream_reasoning` | Boolean [reasoning output controls](#qwen-reasoning-and-tool-calls) that default to `true`. |
| `tool_choice`, `parallel_tool_calls` | Use the [tool selection controls](#tool-choice). |
| `reasoning_effort` | With thinking enabled, use `"xhigh"` (default), `"medium"`, or `"low"`. |

The rendered prompt and its output budget share the 262,144-token context. The rendered prompt must contain 1 to 1,048,576 UTF-8 bytes. The [shared HTTP size limits](text-serving.md#send-a-request) also apply. Invalid options, messages, or limits return HTTP 400.

The first request that reaches rendering loads the pinned `tokenizer_config.json` and `chat_template.jinja` from the checkpoint directory. A missing, unreadable, or mismatched chat asset returns HTTP 500 before output starts. Later chat requests retry a failed load. Raw text can still load `tokenizer.json` independently. The benchmark ID route requires neither tokenizer nor chat assets.

Chat and completion routes share [resident memory admission](batch-graphs.md#resident-memory-admission) and [batch graph eligibility](batch-graphs.md). See the [server MTP policy](mtp.md#runtime-path) for text-only chat, including requests with constraints or token logprobs. Media chat uses K0.

A TCP reset or failed write cancels the request. The server discards pending UTF-8 bytes, buffered reasoning, and incomplete tool calls. It retires the lane before it reuses the slot. See [completion streams](http-server.md#completion-streams) for errors before and after headers. [Request lifecycle](../apps/server/text_completion.cpp#L636)<!-- prefix: "void complet" -->

### Sampling defaults and output budget

`--sampling-defaults model` is the server default. Chat reads sampling values from the selected checkpoint's `generation_config.json` on first use. For Qwen3.8-27B-FP8, the values are `temperature: 1`, `top_p: 0.95`, `top_k: 20`, and `min_p: 0`.

Use `--sampling-defaults openai` for `temperature: 1`, `top_p: 1`, `top_k: -1`, and `min_p: 0`. Model mode falls back to these values for missing assets or fields. Omission or null for these four controls inherits the selected default. The request supplies the seed. Raw text has separate defaults. [Server option](../apps/sglang-server.cpp#L29)<!-- prefix: "SamplingDefa" --> · [Defaults loader](../apps/server/text_request.cpp#L687)<!-- prefix: "const Decode" --> · [Request overrides](../apps/server/text_request.cpp#L533)<!-- prefix: "} else if ((" -->

If both budgets are absent or null, the output limit is 262,142 minus the final prompt length. The prompt must then contain fewer than 262,138 positions, including media expansion. Batch admission can lower this limit to fit the worker's total resident capacity. An explicit budget must fit in full. [Budget calculation](../apps/server/text_request.cpp#L378)<!-- prefix: "void finish_" --> · [Media expansion](../apps/server/text_request.cpp#L980)<!-- prefix: "if (media) {" --> · [Resident limit](../src/decoder_resources.cpp#L363)<!-- prefix: "int limit_re" -->

### Image and video inputs

Media chat accepts up to four attachments in `user` content and preserves their order among text parts. Each media lane uses private eager K0 state. Media lanes bypass prefix reuse even when the listener enables it. The route rejects `ebnf`, `regex`, `response_format`, and `cache_salt`. Media uses the same chat sampling and stop options. [Media depth selection](../apps/server/batch_service.cpp#L894)<!-- prefix: "job-\u003eoptions" -->

Omit `add_special_tokens` or set it to `false`. Set `enable_thinking` at the top level or as the only member of `chat_template_kwargs`.

Each attachment has two members. Set `type` to `"image_url"` or `"video_url"` and use that value as the key for its nested object. Use canonical base64 and an inline data URL. The server rejects network and file URLs.

| Format | Nested object |
| --- | --- |
| Decoded RGB8 image | `url` starts with `data:application/x-sglang-rgb8;base64,`. Supply `height` and `width`. |
| Decoded RGB8 video | Use the same URL prefix and dimensions. Also supply `source_frame_count`, `source_fps`, and `frame_indices`. |
| PNG | With codecs enabled, supply only `url` with `data:image/png;base64,`. |
| MP4 | With codecs enabled, supply only `url` with `data:video/mp4;base64,`. |

Each dimension is 16–1024 pixels. RGB8 bytes are tightly packed in frame, row, column, channel order. An image carries one frame. A decoded video carries 2–32 frames in the order of `frame_indices`. Supply nondecreasing zero-based indices within `source_frame_count`, which can be 2–1,000,000,000. Use finite `source_fps` from 0.001 through 1,000,000. [Decoded video metadata](../apps/server/chat_media.cpp#L55)<!-- prefix: "VideoMetadat" -->

PNG decoding preserves RGB and drops alpha. MP4 requires one unrotated MPEG-4 Part 2 `mp4v` video track. Use constant frame spacing, fixed dimensions, and YUV420P or YUVJ420P frames. MP4 limits include 2–256 source frames, 0.001–1,000 FPS, 4,096 packets, and 16 MiB of selected RGB8.

MP4 selects up to four evenly spaced source ordinals by default. The selection includes both endpoints and rounds fractional ordinals down. To change the count, set `media_io_kwargs` to an object with just `video`. In `video`, set `video_backend` and `backend` to `"opencv"` and `num_frames` to an integer from 2 through 32. The selection cannot exceed the source frame count. The only accepted `mm_processor_kwargs` is `{"do_sample_frames":false}`. FFmpeg handles decoding. [MP4 limits](../apps/server/media_mp4.cpp#L21)<!-- prefix: "constexpr in" --> · [Frame options](../apps/server/chat_media.cpp#L184)<!-- prefix: "int selected" -->

Requests must fit the 2 MiB HTTP body limit and the context and vision limits. The server bounds the patch grid at 4,096 patches per frame and 16,384 per request. Oversized HTTP bodies return 413. Malformed media, disabled codecs, or exceeded media limits return 400. [Grid bounds](../apps/server/chat_media.cpp#L88)<!-- prefix: "int bound_gr" -->

## Verification

Build the release server. Reserve a B200. Start the server with the pinned checkpoint and `--max-active 1 --mtp-steps 0`. Follow the [chat fixture runbook](../tests/text/README.md).

```bash
python3 -B tests/text/test_native_chat.py --port 17384 -v
```

Require exact rendered prompt IDs and usage against the frozen template fixtures. Check JSON and SSE parity, matched stop IDs without stop text, malformed requests, and replay after a reset. Compare raw requests with the equivalent rendered prompts. Run the [raw text checks](text-serving.md#verification) and the [benchmark streaming checks](development.md#streaming-verification) separately.

Check both sampling default modes with omitted, null, and explicit controls. Check budget alias priority in both field orders. Check the omitted-budget boundary after tokenization and media expansion. Fix the seed and score history. Compare output IDs and usage across JSON and SSE. Include a text stop that spans tokens.

Compare prompt IDs, committed IDs, finish reasons, and usage for unconstrained K4 and K0 requests. Use the same chat fixtures and limits at width one and with overlapping requests. For constrained chat, follow the [constrained MTP checks](mtp.md#constrained-verification). For batch graphs, overlap chat, raw text, and benchmark ID requests. Compare each response with a same-policy control. Correlate a multi-lane decode mask with an actual CUDA graph launch. Reset a chat stream while a peer is active. Check the peer and replacement request against their controls.

For serving performance, record template and tokenizer assets, request bytes, rendered prompt IDs, stops, and active decoder options. Use the [performance contract](performance.md) for frozen comparisons and timing definitions. Keep chat checks separate from W1–W6.

### Qwen output checks

Run the [Qwen chat output checks](../tests/text/README.md#qwen-chat-output) for parser boundaries, response fields, and finish reasons. Compare both public examples in JSON and SSE. Check their committed IDs against a control with both parsers disabled. Check isolated and overlapping requests.

Compare auto, none, required, and named choices in JSON and SSE. Include prose-only strict auto. Replay two calls with loose auto and `parallel_tool_calls: false`. Test a required strict string enum containing `"null"`. Check that serial responses expose at most one call. Verify the types of string and null arguments. Check that unsupported forced requests return 400 before SSE starts. Run the [tool constraint checks](output-constraints.md#tool-constraint-checks) for masked generation and length cuts.

### Media checks

Reserve a B200 and set `checkpoint` to the pinned checkpoint path. Build the release server. Start it at capacity one with prefix reuse off and exact K0.

```bash
xmake f -P . -m release -o build/media-server
xmake build -P . sglang-server
build/media-server/bin/sglang-server --checkpoint "$checkpoint" \
  --port 17384 --max-active 1 --prefix-cache-entries 0 --mtp-steps 0 --fp8-prefill exact
```

Verify the [pinned W1 media corpus](../benchmarks/media/README.md) from the repository root.

```bash
python3 -B -m benchmarks.media.collect verify
```

Decode the frozen PNG to RGB8. Decode MP4 frames `[0,2,4,7]` to RGB8. Check the decoded byte hashes against the case files. Replace the attachments with the decoded format above. Preserve the text, video FPS, frame indices, and generation options. Remove `media_io_kwargs` and `mm_processor_kwargs` from the decoded request.

Match all 128 expanded prompt IDs to the case file. Require 128 generated IDs, `length`, usage of `128/128/256`, and zero cached prompt tokens. Compare all generated IDs between native JSON and SSE. Repeat with `--eager` and compare the outputs. Require `[DONE]` on SSE.

Verify HTTP rejection before SSE for malformed media. Cancel a live stream and check recovery with a valid request. Run [admission checks](../tests/batch/README.md) for media alongside text. With prefix reuse enabled, check that a valid media completion reports zero cached IDs. Compare the next text request with its warm text control.

### Codec checks

For compressed media, configure an absolute `codec_prefix` containing libpng and FFmpeg headers and libraries. The server links `png16`, `avformat`, `avcodec`, `swscale`, and `avutil`. The default build accepts decoded RGB8. [Build option](../xmake.lua#L138)<!-- prefix: "option(\"medi" --> · [Codec setup](../xmake.lua#L144)<!-- prefix: "local functi" -->

```bash
xmake f -P . -m release --media-codec-prefix="$codec_prefix" -o build/media-server
xmake build -P . sglang-server
```

Start the codec-enabled server on the reserved B200 with prefix reuse off. Verify the [pinned media corpus](../benchmarks/media/README.md), then send its image and video bodies unchanged.

```bash
python3 -B -m benchmarks.media.collect verify
for kind in image video; do
  curl --fail-with-body -N http://127.0.0.1:17384/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary "@benchmarks/media/W1-${kind}.http-body"
done
```

Require the same expanded prompt IDs, output IDs, usage, and finish in all four native configurations. Compare decoded RGB8 on the HTTP-only and default codec builds. Compare the two compressed inputs on the codec-enabled build with default and explicit eager options. Check JSON and SSE in each configuration. Test disabled codecs and malformed or over-limit inputs before SSE.

Use the archived vLLM output IDs for cross-engine diagnostics. Functional replay checks the native model and response contract. Accepted cross-engine performance requires a separate run under the [frozen media timing protocol](../benchmarks/media/README.md#reproduce-the-measurement).
