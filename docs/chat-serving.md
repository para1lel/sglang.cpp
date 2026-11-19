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
| HTTP ingress | Select the chat route and pass validated prompt IDs to the decoder. | [Route](../apps/sglang-server.cpp#L126)<!-- prefix: "} else if (r" --> · [Cache-enabled route](../apps/server/prefix_controls.cpp#L24)<!-- prefix: "bool chat_ro" --> · [Request parser](../apps/server/text_request.cpp#L265)<!-- prefix: "TextRequest " --> |
| Chat template | Verify checkpoint assets and render ordered messages and tools as a UTF-8 prompt. | [Asset checks](../apps/server/chat_template.cpp#L410)<!-- prefix: "ChatTemplate" --> · [Renderer](../apps/server/chat_template.cpp#L421)<!-- prefix: "std::string " --> |
| Completion writer | Serialize committed output and own the UTF-8 state for one request. | [Token callback](../apps/server/text_completion.cpp#L150)<!-- prefix: "bool TextRes" --> · [Stream end](../apps/server/text_completion.cpp#L165)<!-- prefix: "void TextRes" --> |

## Runtime path

The server parses chat once as ordered JSON and returns HTTP 400 for floating-point range errors. It preserves oversized integer `top_k` values for validation and clamping. The native renderer uses the pinned checkpoint's template rules and literals. The [shared tokenizer](text-serving.md#tokenizer-assets) encodes the rendered prompt without adding special tokens.

The serial route calls `Decoder::generate`. The batch route submits the IDs to `BatchService::generate` with the connection's worker slot. Rendering, tokenization, and response writing run outside CUDA graphs. Eligible decode cohorts can mix chat, [raw strings](text-serving.md), and [benchmark ID requests](../benchmarks/streaming/README.md#wire-contract). [Serial dispatch](../apps/server/completions.cpp#L237)<!-- prefix: "void complet" --> · [Batch dispatch](../apps/server/completions.cpp#L250)<!-- prefix: "void complet" -->

### Read JSON and SSE

The JSON response has one assistant choice at `choices[0].message`. Its `content` holds the generated text. Input tools and reasoning history affect the rendered prompt. Generated reasoning and tool-call markup stay in `content`. Structured output parsing is a separate feature.

JSON uses `object: "chat.completion"`. SSE uses `object: "chat.completion.chunk"`. Both carry `id`, `model`, and `created` in Unix seconds. All chunks in a stream share these values. The response ID starts with `chatcmpl-`.

The first SSE chunk carries `delta.role: "assistant"` and a null finish reason. Join later `delta.content` strings in order. The terminal choice has `delta: {}` and `finish_reason: "stop"` or `"length"`. When requested, a usage event with `choices: []` follows the terminal choice. The stream ends with `[DONE]`.

JSON responses always include usage. `prompt_tokens` counts rendered prompt IDs. `completion_tokens` counts committed output IDs. `total_tokens` sums both counts. `prompt_tokens_details.cached_tokens` counts [restored prompt IDs](prefix-cache.md#runtime-path). A matched stop ID counts as output, even at the budget. The writer omits its text and reports `stop`. For unconstrained requests, exhaustion without a stop reports `length`, including a zero budget. See [constrained finishes](text-serving.md#finite-choice-constraints).

With `return_token_ids: true`, JSON includes `prompt_token_ids` and `token_ids` in its choice. In SSE, the first diagnostic choice after the role carries `prompt_token_ids`. Each committed ID appears once in `token_ids`, even when it produces empty text. A UTF-8 tail flush can carry text with zero IDs. [Diagnostic choice](../apps/server/text_completion.cpp#L114)<!-- prefix: "std::string " --> · [Role delta](../apps/server/text_completion.cpp#L136)<!-- prefix: "bool TextRes" -->

## Constraints and failure behavior

### Messages and tools

- Use `system`, `user`, `assistant`, or `tool` roles. A system message must be first. Include a user query distinct from tool responses.
- Use a string or text parts for `content`. The renderer also accepts null or omitted content. Empty image and video placeholders can render template tokens outside the system message. Media payloads return HTTP 400.
- Supply tool schemas as an array of objects. Omit `tool_choice` or set it to `"auto"`. The renderer inserts schemas into the prompt. Tool schemas guide generation through the prompt alone.
- Supply earlier assistant `tool_calls` as an array. Put the `name` and `arguments` in a `function` object or directly in the call. If supplied, `type` must be `"function"`. Arguments can be objects or strings that encode JSON objects. A `tool` message supplies a tool response through its `content`.
- Use `reasoning_content`, `thinking`, or `reasoning` for earlier assistant reasoning. The renderer uses the first non-null field in that order.

### Request options

| Field | Accepted values and effect |
| --- | --- |
| `max_tokens`, `max_completion_tokens` | Use at most one. Each sets a nonnegative integer budget. The default is 16. |
| `n` | Omit or set to `1`. |
| `temperature`, `top_p`, `top_k`, `seed` | Use the [shared sampling controls](text-serving.md#sampling). |
| `ebnf` | Use a [finite-choice constraint](text-serving.md#finite-choice-constraints) for the generated content. |
| `cache_salt` | Optional [prefix cache key](prefix-cache.md#runtime-path) component. |
| `stream`, `return_token_ids`, `ignore_eos`, `stop_token_ids` | Use the [shared text options](text-serving.md#send-a-request). |
| `stream_options` | With streaming enabled, accepts one boolean member, `include_usage`. The default omits the usage event. |
| `add_generation_prompt`, `add_vision_id` | Booleans. HTTP defaults are `true` and `false`, respectively. |
| `enable_thinking`, `preserve_thinking` | Booleans that default to `true`. They control rendering of the generation prompt and assistant history. |
| `reasoning_effort` | With thinking enabled, use `"xhigh"` (default), `"medium"`, or `"low"`. |

The rendered prompt and its output budget share the 262,144-token context. The rendered prompt must contain 1 to 1,048,576 UTF-8 bytes. The [shared HTTP size limits](text-serving.md#send-a-request) also apply. Invalid options, messages, or limits return HTTP 400.

The first request that reaches rendering loads the pinned `tokenizer_config.json` and `chat_template.jinja` from the checkpoint directory. A missing, unreadable, or mismatched chat asset returns HTTP 500 before output starts. Later chat requests retry a failed load. Raw text can still load `tokenizer.json` independently. The benchmark ID route requires neither tokenizer nor chat assets.

Chat and completion routes share the [serving limits](development.md#native-streaming-server) and [batch graph eligibility](batch-graphs.md). Unconstrained greedy chat can use the configured [MTP depth](mtp.md). Sampled and constrained chat use K0.

A reset or failed write cancels the request and discards its pending UTF-8 bytes. In batch mode, the server retires the lane before it reuses the slot. If generation fails after SSE starts, the server attempts an error event and closes the stream. [Request lifecycle](../apps/server/text_completion.cpp#L201)<!-- prefix: "void complet" -->

## Verification

Build the release server. Reserve a B200. Start the server with the pinned checkpoint, `--max-active 1`, and MTP off. Follow the [chat fixture runbook](../tests/text/README.md).

```bash
python3 -B tests/text/test_native_chat.py --port 17384 -v
```

Require exact rendered prompt IDs and usage against the frozen template fixtures. Check JSON and SSE parity, matched stop IDs without stop text, malformed requests, and replay after a reset. Compare raw requests with the equivalent rendered prompts. Run the [raw text checks](text-serving.md#verification) and the [benchmark streaming checks](development.md#streaming-verification) separately.

For a K4 check, match prompt IDs, committed IDs, finish reasons, and usage against MTP off. Use the same chat fixtures and limits at width one and with overlapping requests. For batch graphs, overlap chat, raw text, and benchmark ID requests. Compare each response with a same-policy control. Correlate a multi-lane decode mask with an actual CUDA graph launch. Reset a chat stream while a peer is active. Check the peer and replacement request against their controls.

For serving performance, record template and tokenizer assets, request bytes, rendered prompt IDs, stops, and active decoder options. Use the [performance contract](performance.md) for frozen comparisons and timing definitions. Keep chat checks separate from W1–W6.
