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
| HTTP ingress | Select the chat route and validate the request. | [Chat route](../apps/server/prefix_controls.cpp#L163)<!-- prefix: "bool chat_ro" --> · [HTTP admission](http-server.md#http-admission-and-shutdown) · [Request parser](../apps/server/text_request.cpp#L866)<!-- prefix: "TextRequest " --> |
| Qwen chat template | Verify checkpoint assets and render ordered messages and tools as a UTF-8 prompt. | [Asset checks](../apps/server/qwen/qwen_chat_template.cpp#L433)<!-- prefix: "QwenChatTemp" --> · [Renderer](../apps/server/qwen/qwen_chat_template.cpp#L560)<!-- prefix: "std::string " --> |
| Media handler | Validate attachments, prepare patches, then encode features for the multimodal decoder. | [Attachment parser](../apps/server/chat_media.cpp#L341)<!-- prefix: "std::optiona" --> · [Vision completion](../apps/server/vision_completion.cpp#L67)<!-- prefix: "DecoderResul" --> |
| Serving profile | Select the model's template and parser policy. Resolve request options and parse reasoning and tool calls. | [Qwen profile](../apps/server/qwen/qwen_serving_profile.cpp#L134)<!-- prefix: "const Servin" --> · [Request options](../apps/server/qwen/qwen_serving_profile.cpp#L99)<!-- prefix: "void set_cha" --> · [Parser](../apps/server/qwen/qwen_chat_output.cpp#L824)<!-- prefix: "bool ChatOut" --> |
| Completion writer | Serialize committed output and own the UTF-8 state for one request. | [Token callback](../apps/server/text_completion.cpp#L581)<!-- prefix: "bool TextRes" --> · [Stream end](../apps/server/text_completion.cpp#L593)<!-- prefix: "void TextRes" --> |

## Runtime path

Chat parses ordered JSON, rejecting floating-point range errors and validating oversized integer `top_k` before clamping. The pinned template renders UTF-8, then the [shared tokenizer](text-serving.md#tokenizer-assets) encodes it without extra special tokens. Text chat uses the connection worker slot and can share decode groups with raw text and benchmark IDs. Rendering, tokenization, and writing stay outside CUDA graphs.

Media placeholders preserve attachment order. CPU patch preparation expands prompt IDs. The worker encodes vision features before submitting to the [multimodal decoder](multimodal-decoder.md). Zero-budget or already cancelled requests skip vision encoding. Vision encoding is outside the target step cap.

### Read JSON and SSE

JSON uses `object: "chat.completion"` and puts assistant messages in `choices`. SSE uses `object: "chat.completion.chunk"`. Both include `id`, `model`, and `created` in Unix seconds. Each choice keeps its own `chatcmpl-` ID. Without parsers, generated reasoning and tool markup stay in `content`, while supplied tools and history still affect the prompt.

Each SSE index begins with `delta.role: "assistant"` and a null finish reason. [Headers and keepalive comments](http-server.md#completion-streams) can precede it. Join `delta.content` by index. The terminal choice has `delta: {}` and a finish reason. Optional usage with `choices: []` follows all terminal choices, then `[DONE]`. JSON always includes usage. The [shared stop and usage rules](text-serving.md#read-the-response) apply, with an additional `tool_calls` finish for parsed calls. Media usage counts expanded prompt IDs.

With `return_token_ids: true`, committed IDs appear once in `token_ids`, including IDs with empty text. A UTF-8 flush can carry text with zero IDs. Text chat puts `prompt_token_ids` in the JSON choice or first diagnostic SSE choice after the role. Media chat puts expanded IDs at the top level of JSON or the initial assistant SSE chunk.

### Qwen reasoning and tool calls

Enable output parsers with `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`. Both default to off. To parse supplied text, use [`POST /parse_function_call`](serving/parse-function-call.md).

Output parsing applies to text chat with free output or a [tool-call constraint](output-constraints.md#tool-call-constraints). Media chat and chat with `response_format`, `regex`, or `ebnf` keep generated text in `content`.

With the reasoning parser active, `separate_reasoning: true` places reasoning in `reasoning_content`. Set it to `false` to keep reasoning in `content`. `enable_thinking` controls the rendered prompt. The parser starts in the reasoning section when the prompt leaves `<think>` open. It also recognizes `<think>` at the start of generated text. `</think>` ends the section. So does `<tool_call>` when tool parsing is active and the request advertises functions. Set `stream_reasoning: false` to buffer reasoning until the section ends or the request completes. Both output options are booleans and default to `true`.

With the tool parser active, complete calls for advertised functions become `tool_calls`. In unconstrained output, ordinary text and invalid wrappers stay in `content`. The parser can recover at a later wrapper. Call-shaped text inside a complete string parameter stays part of the argument. [Tool frame scanner](../apps/server/qwen/qwen_chat_output.cpp#L688)<!-- prefix: "std::size_t " -->

In JSON, read `message.reasoning_content` and `message.tool_calls`. With calls and empty content, `message.content` is `null`. In SSE, join the `delta.reasoning_content` strings. Group the `delta.tool_calls` entries by zero-based `index`. The first delta gives the call's `id`, `type: "function"`, `function.name`, and empty arguments. Append the next delta's `function.arguments`. In both formats, parse the argument string as a JSON object. A stop with a complete call reports `tool_calls`. A token-limit cut without a stop reports `length` and can carry a call with incomplete arguments. Text stops match generated text before parsing.

The [reasoning and weather requests](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/chat-serving.md#qwen-reasoning-and-tool-calls) reproduce `391` separate from reasoning and a `get_weather` call with `{"city":"Paris"}` in JSON and SSE.

## Constraints and failure behavior

### Messages and tools

- Use `system`, `developer`, `root`, `user`, `assistant`, or `tool` roles. The renderer treats `developer` and `root` as `system`. A system message must be first. Include a user query distinct from tool responses.
- Use a string or text parts for text content. The renderer also accepts null or omitted content. Empty image and video placeholders can render template tokens outside the system message. Follow the attachment contract for actual media.
- Supply function schemas in `tools`. The renderer collects top-level tools first, then tools on `system`, `developer`, or `root` messages. Advertised tools need `type: "function"` and unique names of 1–64 UTF-8 bytes. Names exclude angle brackets, carriage return, and line feed. [Tool selection](../apps/server/qwen/qwen_chat_template.cpp#L518)<!-- prefix: "ChatTools se" -->
- Supply earlier assistant `tool_calls` as an array. Put the `name` and `arguments` in a `function` object or directly in the call. If supplied, `type` must be `"function"`. Arguments can be objects or strings that encode JSON objects. A `tool` message supplies a tool response through its `content`.
- Use `reasoning_content`, `thinking`, or `reasoning` for earlier assistant reasoning. The renderer uses the first non-null field in that order.

### Tool choice

| `tool_choice` | Behavior |
| --- | --- |
| `"auto"`, null, or omitted | Advertise supplied functions. Allow prose and optional calls. An advertised `function.strict: true` activates the tool grammar when parsing is available. |
| `"none"` | Disable tool advertisement and extraction. Keep earlier assistant calls and tool responses in the prompt. Reasoning separation can still run. |
| `"required"` | Require a call frame for an advertised function. |
| `{"type":"function","function":{"name":"get_weather"}}` | Advertise only the selected function and require a single call frame for it. |

`parallel_tool_calls` is a boolean and defaults to `true`. With `false`, JSON and SSE expose at most one call. Auto with only non-strict tools leaves later wrappers in `content` after the first accepted call. Named choice always allows one call. Required and named choices need an eligible tool, text-only chat, and the `qwen3_coder` parser. They return HTTP 400 when a prerequisite is missing or `response_format`, `regex`, or `ebnf` is set. Active tool constraints require the model EOS stop set and reject nonempty text stops. [Tool admission](../apps/server/qwen/qwen_serving_profile.cpp#L70)<!-- prefix: "void check_t" --> · [Parser dispatch](../apps/server/qwen/qwen_chat_output.cpp#L824)<!-- prefix: "bool ChatOut" -->

The parser accepts an object schema for `function.parameters`, null, or omission. It reads `properties`, `required`, and `additionalProperties`. It removes up to one line feed from each end of a value. It converts scalars and JSON collections. A strict string parameter with the XML value `null` produces the JSON string `"null"`. A strict null parameter produces JSON null. In a loose tool, bare `null` in any letter case produces JSON null, and declared type alternatives are tried in order for other values. Loose tools allow extra keys by default. Validate loose arguments in the application. [Argument conversion](../apps/server/qwen/qwen_chat_output.cpp#L412)<!-- prefix: "void ToolFra" --> · [Strict schema limits](output-constraints.md#tool-call-constraints)

### Request options

| Field | Accepted values and effect |
| --- | --- |
| `max_tokens`, `max_completion_tokens` | Nonnegative integer or null. A non-null `max_completion_tokens` takes priority in either field order. |
| `min_tokens` | [Minimum output](output-penalties.md#minimum-output), default zero. |
| `n` | [Multiple choices](serving/openai-choices.md), integer `1`–`4` for text. |
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

Chat and text share [resident memory admission](batch-graphs.md#resident-memory-admission) and [batch graph eligibility](batch-graphs.md). Text and media use the [configured MTP policy](mtp.md#runtime-path).

A TCP reset or failed write cancels the request. The server discards pending UTF-8 bytes, buffered reasoning, and incomplete tool calls. It retires the lane before it reuses the slot. See [completion streams](http-server.md#completion-streams) for errors before and after headers. [Request lifecycle](../apps/server/text_completion.cpp#L727)<!-- prefix: "void complet" -->

### Sampling defaults and output budget

`--sampling-defaults model` is the server default. Chat reads sampling values from the selected checkpoint's `generation_config.json` on first use. For Qwen3.8-27B-FP8, the values are `temperature: 1`, `top_p: 0.95`, `top_k: 20`, and `min_p: 0`.

Use `--sampling-defaults openai` for `temperature: 1`, `top_p: 1`, `top_k: -1`, and `min_p: 0`. Model mode falls back to these values for missing assets or fields. Omission or null for these four controls inherits the selected default. The request supplies the seed. Raw text has separate defaults. [Server option](../apps/sglang-server.cpp#L44)<!-- prefix: "SamplingDefa" --> · [Defaults loader](../apps/server/text_request.cpp#L586)<!-- prefix: "const Decode" --> · [Request overrides](../apps/server/text_request.cpp#L529)<!-- prefix: "} else if ((" -->

If both budgets are absent or null, the output limit is 262,142 minus the final prompt length. The prompt must then contain fewer than 262,138 positions, including media expansion. Batch admission can lower this limit to fit the worker's total resident capacity. An explicit budget must fit in full. [Budget calculation](../apps/server/text_request.cpp#L381)<!-- prefix: "void finish_" --> · [Media expansion](../apps/server/text_request.cpp#L882)<!-- prefix: "if (media) {" --> · [Resident limit](../src/runtime/model/qwen_request.cpp#L87)<!-- prefix: "int limit_re" -->

### Image and video inputs

Media chat accepts up to four attachments in `user` content and preserves their order among text parts. It uses the same sampling, stop, and configured MTP settings as text. A matching [media checkpoint](prefix-cache.md#media-identity-and-prefill) can restore prompt rows. `cache_salt` partitions reuse. [Generated logprobs](serving/openai-logprobs.md#chat-completions) work for unconstrained greedy media chat.

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

On a reserved B200, start the Release server with the pinned checkpoint and `--max-active 1 --mtp-steps 0`. Run `python3 -B tests/text/test_native_chat.py --port 17384 -v` using the [chat fixtures](../tests/text/README.md#chat-layer-1). Require exact rendered IDs and usage, JSON/SSE parity, stop IDs without stop text, HTTP rejection and reset recovery. Match equivalent raw text, including a cross-token stop with fixed seed and score history. In both sampling modes cover omitted, null, and explicit controls, both alias orders, and the omitted-budget boundary after rendering and media expansion.

Compare K4 and K0 on matched unconstrained fixtures at width one and with overlaps, including prompt/output IDs, finish and usage. Use the [constrained MTP gates](mtp.md#constrained-verification). In a mixed chat/text/benchmark batch, require same-policy response parity and a real multi-lane CUDA graph launch. After resetting chat, match peer and replacement controls. Run [text](text-serving.md#verification) and [benchmark](development.md#streaming-verification) independently. The [performance contract](performance.md) defines reporting for matched assets, request bytes, rendered IDs, stops, and decoder settings.

### Qwen output checks

Run the [Qwen chat output checks](../tests/text/README.md#qwen-chat-output) on isolated and overlapping requests. Require parser-off ID parity, JSON/SSE field and finish parity, typed string and null arguments, and pre-SSE 400 for invalid forced choices. Cover auto, none, required, named, prose-only strict auto, loose serial auto with two wrappers, and a strict string enum containing `"null"`. Serial choice exposes at most one call. Check [masked generation and length cuts](output-constraints.md#tool-constraint-checks).

### Media checks

Use K0, capacity one and prefix reuse off on a reserved B200. Run `python3 -B -m benchmarks.media.collect verify`. Decode the [frozen](../benchmarks/media/README.md) PNG and MP4 frames `[0,2,4,7]` to RGB8 and verify their byte hashes. Preserve text, FPS, indices and generation options, dropping codec-only preprocessing kwargs. Require all 128 frozen expanded IDs, 128 output IDs, `length`, usage `128/128/256`, zero cached IDs and SSE `[DONE]`. Match generated IDs across JSON, SSE and eager execution. Require pre-SSE malformed-media rejection, cancellation recovery, and the [mixed media/text gates](../tests/media/http-contract.md) for positive MTP, cold/warm usage, and misses on changed media. [Build and replay procedure](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/chat-serving.md#media-checks).

### Codec checks

Configure `--media-codec-prefix` with an absolute directory containing libpng and FFmpeg headers and libraries. The server links `png16`, `avformat`, `avcodec`, `swscale` and `avutil`. The default build accepts decoded RGB8. Send the frozen image/video bodies unchanged to the codec-enabled server. Require matching expanded IDs, output IDs, usage and finishes in JSON and SSE for decoded RGB8 on the HTTP-only and codec builds, and compressed media on the codec build in default and eager modes. Check disabled codecs and malformed or over-limit input before SSE. [Codec build and replay](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/chat-serving.md#codec-checks) · [Cross-engine timing](../benchmarks/media/README.md#reproduce-the-measurement).
