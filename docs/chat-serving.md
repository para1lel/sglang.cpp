# Native chat completions

Send `POST /v1/chat/completions` with `model: "Qwen3.8-27B-FP8"` and a nonempty `messages` array. Follow the [server launch procedure](../README.md#serve-token-ids-over-http).

```bash
curl --fail-with-body -N http://127.0.0.1:17384/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen3.8-27B-FP8","messages":[{"role":"user","content":"What is the capital of Japan?"}],"stream":true,"stream_options":{"include_usage":true}}'
```

## Responsibilities and sources of truth

The [request parser](../apps/server/text_request.cpp#L866)<!-- prefix: "TextRequest " --> validates chat input. The [Qwen template](../apps/server/qwen/qwen_chat_template.cpp#L560)<!-- prefix: "std::string " --> renders messages and tools. The [serving profile](../apps/server/qwen/qwen_serving_profile.cpp#L134)<!-- prefix: "const Servin" --> selects parsers, and the shared completion writer owns UTF-8 state. [Implementation map](https://github.com/para1lel/sglang.cpp/blob/49a48f98e44223bd619e7b6f43569b45175504c0/docs/chat-serving.md#responsibilities-and-sources-of-truth)

## Runtime path

Chat shares [text JSON validation and tokenization](text-serving.md#send-a-request). The pinned template renders ordered messages and tools without adding tokenizer special tokens. Chat can share decoder groups with text and benchmark IDs. Rendering, tokenization, and writing stay outside CUDA graphs.

CPU patch preparation expands prompt IDs before submission to the [multimodal decoder](multimodal-decoder.md). A cancelled or zero-budget request skips vision encoding, which stays outside the step cap.

### Read JSON and SSE

JSON uses `object: "chat.completion"` and puts assistant messages in `choices`. One top-level `chatcmpl-` ID identifies the response. Each choice is identified by its `index` alone. SSE uses `object: "chat.completion.chunk"` and a distinct top-level `chatcmpl-` ID for each index. Both formats include `model` and `created` in Unix seconds. Without parsers, generated reasoning and tool markup stay in `content`, while supplied tools and history still affect the prompt.

Each SSE index starts with `delta.role: "assistant"` and a null finish reason. Join `delta.content` by index. The terminal choice has `delta: {}`. Follow the [shared stop, usage, and end marker rules](text-serving.md#read-the-response), with a separate `tool_calls` finish. JSON always includes usage. Media usage counts expanded prompt IDs.

With `return_token_ids: true`, committed IDs appear once in `token_ids`, including IDs with empty text. A UTF-8 flush can carry text with zero IDs. Text chat puts `prompt_token_ids` in the JSON choice or first diagnostic SSE choice after the role. Media chat puts expanded IDs at the top level of JSON or the initial assistant SSE chunk.

### Qwen reasoning and tool calls

Use `--reasoning-parser qwen3` for reasoning or `--tool-call-parser qwen3_coder` for tool calls. Each defaults to off and applies to text chat with free output or a [tool-call constraint](output-constraints.md#tool-call-constraints). Media chat and requests with `response_format`, `regex` or `ebnf` keep generated text in `content`. To parse supplied text, use [`POST /parse_function_call`](serving/parse-function-call.md).

With the reasoning parser active, `separate_reasoning: true` places reasoning in `reasoning_content`. Set it to `false` to keep reasoning in `content`. `enable_thinking` controls the rendered prompt. The parser starts in the reasoning section when the prompt leaves `<think>` open. It also recognizes `<think>` at the start of generated text. `</think>` ends the section. So does `<tool_call>` when tool parsing is active and the request advertises functions. Set `stream_reasoning: false` to buffer reasoning until the section ends or the request completes. Both output options are booleans and default to `true`.

With the tool parser active, complete calls for advertised functions become `tool_calls`. In unconstrained output, ordinary text and invalid wrappers stay in `content`. The parser can recover at a later wrapper. Call-shaped text inside a complete string parameter stays part of the argument. [Tool frame scanner](../apps/server/qwen/qwen_chat_output.cpp#L633)<!-- prefix: "std::size_t " -->

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

`parallel_tool_calls` is a boolean and defaults to `true`. With `false`, JSON and SSE expose at most one call. Auto with only non-strict tools leaves later wrappers in `content` after the first accepted call. Named choice always allows one call. Required and named choices need an eligible tool, text-only chat, and the `qwen3_coder` parser. They return HTTP 400 when a prerequisite is missing or `response_format`, `regex`, or `ebnf` is set. Active tool constraints require the model EOS stop set and reject nonempty text stops. [Tool admission](../apps/server/qwen/qwen_serving_profile.cpp#L70)<!-- prefix: "void check_t" --> · [Parser dispatch](../apps/server/qwen/qwen_chat_output.cpp#L769)<!-- prefix: "bool ChatOut" -->

The parser accepts an object schema for `function.parameters`, null, or omission. It reads `properties`, `required`, and `additionalProperties`. It removes up to one line feed from each end of a value. It converts scalars and JSON collections. A strict string parameter with the XML value `null` produces the JSON string `"null"`. A strict null parameter produces JSON null. In a loose tool, bare `null` in any letter case produces JSON null, and declared type alternatives are tried in order for other values. Loose tools allow extra keys by default. Validate loose arguments in the application. [Argument conversion](../apps/server/qwen/qwen_chat_output.cpp#L362)<!-- prefix: "void ToolFra" --> · [Strict schema limits](output-constraints.md#tool-call-constraints)

### Request options

| Field | Accepted values and effect |
| --- | --- |
| `max_tokens`, `max_completion_tokens` | Nonnegative integer or null. A non-null `max_completion_tokens` takes priority in either field order. |
| `min_tokens` | [Minimum output](output-penalties.md#minimum-output), default zero. |
| `n` | [Multiple choices](serving/openai-choices.md), integer `1`–`4` for text. |
| `temperature`, `top_p`, `top_k`, `min_p`, `seed` | Use the [shared sampling controls](text-serving.md#sampling). A non-null value overrides its default. |
| `frequency_penalty`, `presence_penalty`, `repetition_penalty` | [Output penalties](output-penalties.md). |
| `ebnf`, `regex`, `response_format` | [Whole-output constraints](output-constraints.md). |
| `cache_salt` | Optional [prefix key](prefix-cache.md#runtime-path) component. |
| `stream`, `return_token_ids`, `ignore_eos`, `stop_token_ids`, `stop` | [Shared text options](text-serving.md#send-a-request). |
| `stream_options` | With streaming enabled, accepts one boolean member, `include_usage`. The default omits the usage event. |
| `logprobs`, `top_logprobs` | Optional [generated-token scores](serving/openai-logprobs.md#chat-completions). |
| `logit_bias` | Optional [per-token logit offsets](serving/openai-logprobs.md#logit-bias). |
| `add_generation_prompt`, `add_vision_id` | Booleans. HTTP defaults are `true` and `false`, respectively. |
| `enable_thinking`, `preserve_thinking` | Booleans that default to `true`. They control rendering of the generation prompt and assistant history. |
| `separate_reasoning`, `stream_reasoning` | Boolean [reasoning output controls](#qwen-reasoning-and-tool-calls) that default to `true`. |
| `tool_choice`, `parallel_tool_calls` | Use the [tool selection controls](#tool-choice). |
| `reasoning_effort` | With thinking enabled, use `"xhigh"` (default), `"medium"`, or `"low"`. |

Rendered prompts require 1–1,048,576 UTF-8 bytes. The prompt and budget share 262,144 tokens. Apply [HTTP size limits](http-server.md#http-admission-and-shutdown). Invalid options, messages, or limits return 400.

First render loads the pinned `tokenizer_config.json` and `chat_template.jinja` from the checkpoint. Unreadable, missing, or mismatched assets return 500 before output. Later chat requests retry. [Tokenizer loading](text-serving.md#send-a-request) stays independent.

Use shared [resident admission](batch-graphs.md#resident-memory-admission), [graph eligibility](batch-graphs.md#runtime-path), [MTP policy](mtp.md#runtime-path), and [stream failure handling](http-server.md#completion-streams). Failed writes and resets also discard pending UTF-8, buffered reasoning, and incomplete tool calls. [Request lifecycle](../apps/server/text_completion.cpp#L725)<!-- prefix: "void complet" -->

### Sampling defaults and output budget

The server defaults to `--sampling-defaults model`. On first use chat reads `generation_config.json` from the selected checkpoint. For Qwen3.8-27B-FP8 this gives `temperature: 1`, `top_p: 0.95`, `top_k: 20`, and `min_p: 0`. Use `--sampling-defaults openai` for the [raw-text values](text-serving.md#sampling). Model mode also uses those values for missing assets or fields. Omission or null inherits the selected values. The request supplies the seed.

If both budgets are absent or null, the limit is 262,142 minus the final prompt length. This requires fewer than 262,138 positions, including media expansion. [Resident admission](batch-graphs.md#resident-memory-admission) can lower the implicit budget. An explicit budget must fit in full. [Server option](../apps/sglang-server.cpp#L44)<!-- prefix: "SamplingDefa" --> · [Defaults loader](../apps/server/text_request.cpp#L586)<!-- prefix: "const Decode" --> · [Request overrides](../apps/server/text_request.cpp#L529)<!-- prefix: "} else if ((" --> · [Budget calculation](../apps/server/text_request.cpp#L381)<!-- prefix: "void finish_" --> · [Media expansion](../apps/server/text_request.cpp#L882)<!-- prefix: "if (media) {" --> · [Resident limit](../src/runtime/model/qwen_request.cpp#L112)<!-- prefix: "int limit_re" -->

### Image and video inputs

Media chat accepts up to four ordered attachments in `user` content and shares text sampling and stops. Use [media-aware prefix reuse](prefix-cache.md#media-identity-and-prefill) for matching checkpoints. [Generated logprobs](serving/openai-logprobs.md#chat-completions) are available for unconstrained greedy requests.

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

The server caps patch grids at 4,096 patches per frame and 16,384 per request. Malformed media, disabled codecs, and media-limit violations return 400. The [body limit](http-server.md#http-admission-and-shutdown) still applies. [Grid bounds](../apps/server/chat_media.cpp#L88)<!-- prefix: "int bound_gr" -->

## Verification

On a reserved B200, use a release server with the pinned checkpoint and `--max-active 1 --mtp-steps 0`. Run `python3 -B tests/text/test_native_chat.py --port 17384 -v` with the [chat fixtures](../tests/text/README.md#chat-layer-1). Require exact rendered IDs and usage, JSON/SSE parity, stop IDs without stop text, HTTP rejection and reset recovery. Compare equivalent raw text with a cross-token stop, fixed seed and scores. Cover omitted, null, and explicit sampling in both modes, both alias orders, and default-budget boundaries after rendering and expansion.

Apply [MTP parity checks](mtp.md#verification) at singleton and overlapping widths and the [constraint checks](mtp.md#constrained-verification). A mixed chat/text/benchmark cohort must match same-policy controls and launch a real multi-lane graph. Match peers and replacements after a chat reset. Run [text](text-serving.md#verification) and [benchmark](development.md#streaming-verification) separately. Record the [performance controls](performance.md).

### Qwen output checks

Run the [Qwen chat output checks](../tests/text/README.md#qwen-chat-output) on isolated and overlapping requests. Require parser-off ID parity, JSON/SSE field and finish parity, typed string and null arguments, and pre-SSE 400 for invalid forced choices. Cover auto, none, required, named, prose-only strict auto, loose serial auto with two wrappers, and a strict string enum containing `"null"`. Serial choice exposes at most one call. Check [masked generation and length cuts](output-constraints.md#tool-constraint-checks).

### Media checks

Run `python3 -B -m benchmarks.media.collect verify` on a reserved B200 with K0, capacity one, and reuse off. Decode the [frozen](../benchmarks/media/README.md) PNG and MP4 frames `[0,2,4,7]` to RGB8 and check byte hashes. Preserve text, FPS, indices and options except codec-only kwargs. Require all 128 frozen expanded IDs, 128 output IDs, `length`, usage `128/128/256`, zero cached IDs and SSE `[DONE]`. Match IDs across JSON, SSE and eager. Check pre-SSE rejection and cancellation. Use the [mixed media/text gates](../tests/media/http-contract.md) for positive depth, cold/warm usage, and changed-media misses. [Build and replay procedure](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/chat-serving.md#media-checks)

### Codec checks

Set an absolute `--media-codec-prefix` with libpng and FFmpeg headers and libraries (`png16`, `avformat`, `avcodec`, `swscale`, `avutil`). Match expanded IDs, outputs, usage and finishes across JSON/SSE for RGB8 on both builds and unchanged frozen compressed requests on the codec build, in default and eager modes. Reject disabled codecs and malformed or over-limit input before SSE. [Codec build and replay](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/chat-serving.md#codec-checks) · [Cross-engine timing](../benchmarks/media/README.md#reproduce-the-measurement)
