# Anthropic Messages

Send `POST /v1/messages` with `Content-Type: application/json` and the served model name
`Qwen3.8-27B-FP8`. Include `messages` and a positive `max_tokens`. The adapter renders the
checkpoint's chat template and submits the result through the native chat generation queue.
Use [`/v1/messages/count_tokens`](../native-tokenization.md#count-anthropic-prompt-tokens) for
prompt token counts. [Dispatch](../../apps/server/prefix_controls.cpp#L642)<!-- prefix: "void Connect" --> ·
[Request adapter](../../apps/server/anthropic_count.cpp#L777)<!-- prefix: "OrderedObjec" --> ·
[Chat generation](../../apps/server/text_completion.cpp#L867)<!-- prefix: "void complet" -->

A configured Ollama route takes precedence for the same method and path. For example,
`SGLANG_OLLAMA_CHAT_ROUTE=/v1/messages` serves Ollama chat on `POST /v1/messages`.
An override applies only to its configured path.

```json
{
  "model": "Qwen3.8-27B-FP8",
  "system": "Answer briefly.",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "max_tokens": 32,
  "temperature": 0,
  "thinking": {"type": "disabled"}
}
```

## Request scope

- `system` accepts a string or text blocks. Inline system turns contribute to the same system
  prompt. The checkpoint template handles ordered user and assistant text and tool history.
  Textual `search_result` history contributes text.
- `temperature`, `top_p`, `top_k`, `max_tokens`, `stop_sequences`, and `cache_salt` use the
  native chat controls and validation. `output_config.format` selects the existing native
  JSON schema constraint. `metadata` and `betas` carry no generation behavior.
- Start with `--reasoning-parser qwen3` for typed thinking and assistant thinking history.
  Omitted thinking follows the checkpoint template default. Set `thinking.type` to `disabled`
  to disable thinking. The checkpoint applies its supported effort levels. The pin maps
  Anthropic `xhigh` to `max`, which this checkpoint's template rejects with thinking enabled.
  `thinking.budget_tokens` and `output_config.task_budget` are compatibility fields and do
  not impose an inference limit. Thinking output is unsigned.
- Start with `--tool-call-parser qwen3_coder` for client function tools with a JSON
  `input_schema`. Tool choice supports `auto`, `none`, `any` and a named `tool`.
  `disable_parallel_tool_use` limits publication to one call. Forced choices inherit
  the [native constraints](../chat-serving.md#tool-choice), including text-only input and no
  text stop sequences. Read `tool_use` blocks, run the function in the client, and send
  `tool_result` history on the next request. Server tool definitions are skipped.
- User image blocks accept inline base64 PNG in a codec-enabled build and use the native
  [media limits](../chat-serving.md#image-and-video-inputs). Media generation runs at K0. Remote image
  and document sources, redacted thinking, deferred tools and disaggregated routing are
  outside the native scope.

## Response and cancellation

The default response is a single assistant `message` with typed `thinking`, `text`, and
`tool_use` blocks, usage, and `stop_reason`. `stream: true` sends named SSE events.
`message_start` carries empty content and initial usage. Indexed blocks use
`content_block_start`, typed `content_block_delta`, and `content_block_stop`. `message_delta`
carries the final stop reason and output token count. `message_stop` ends the response.
The adapter uses `end_turn` for EOS or a text stop, `max_tokens` for a length cut, and `tool_use`
for a completed tool turn. The pin omits `stop_sequence`. Native prefix hits split input
usage into uncached `input_tokens` and `cache_read_input_tokens`.
[Event writer](../../apps/server/anthropic_messages.cpp#L96)<!-- prefix: "bool Anthrop" --> ·
[Finish](../../apps/server/anthropic_messages.cpp#L112)<!-- prefix: "void Anthrop" -->

Request validation errors return an Anthropic JSON envelope. Native transport limits and
unsupported model features can produce different validation details from Python SGLang.
A decoder failure while streaming closes any open block, sends `error`, and sends
`message_stop` if the socket remains writable. Closing the connection cancels the native
job. [Failure events](../../apps/server/anthropic_messages.cpp#L140)<!-- prefix: "void Anthrop" --> ·
[Socket lifecycle](../http-server.md)

The [pinned adapter](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/anthropic/serving.py#L797)<!-- prefix: "async def _h" -->
defines the generation and event contract. The [native wire test](../../tests/constraints/test_messages.py#L1)<!-- prefix: "\"\"\"Exercise " -->
checks event order, reconstructed content, usage and error envelopes against native chat.
