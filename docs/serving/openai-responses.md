# OpenAI Responses

Send `POST /v1/responses` with `Content-Type: application/json` and the served model name.
The server uses the checkpoint's chat template and native text generation queue.

```json
{
  "model": "Qwen3.8-27B-FP8",
  "input": "What is 6 times 7? Answer with just the number.",
  "reasoning": {"effort": "none"},
  "max_output_tokens": 64,
  "temperature": 0
}
```

`input` also accepts an array of user, assistant, system, or developer messages with string
content or text parts. Supply top-level `instructions` to add a system instruction for the
current request. `stream:true` emits SSE lifecycle, item, part, and delta events with
increasing `sequence_number` values and a final response snapshot. A length cutoff returns
`incomplete` with `incomplete_details.reason` set to `max_output_tokens`. An explicit output
budget reserves two tokens, following the pinned SGLang handler.

For plain text output, start with `--reasoning-parser qwen3` to receive reasoning as a
distinct output item. A reasoning summary request projects that text through summary content
and events. Without the parser, plain text output carries generated reasoning in the message
text. Select `reasoning.effort:"none"` to disable thinking, subject to explicit chat template
overrides.
`reasoning.reasoning_effort:"none"` also disables thinking when `effort` is omitted or null.

The `text` config accepts an empty object, a plain text or JSON Schema format, and `low`,
`medium` or `high` verbosity metadata. `service_tier` accepts `auto`, `default`, `flex`,
`scale` or `priority` with the same native scheduling for all tiers. Unrecognized top-level
JSON keys are ignored.
`request_id` can supply the response ID. Use `resp_` followed by letters, digits, underscores
or hyphens, up to 128 characters total. Otherwise the server supplies an ID.

## Inline images

User messages can include an `input_image` part with an `image_url` such as
`data:image/png;base64,...` alongside `input_text`. Omit `detail` or use `null` or `auto`. A
codec-enabled build uses the [native chat media limits](../chat-serving.md) and reports
expanded prompt tokens in `usage.input_tokens`. Blocking, SSE, and stored follow-ups use
the same media path. External URLs, file IDs, video and image detail or resizing overrides
return HTTP 400. [Image translation](../../apps/server/responses.cpp#L109)<!-- prefix: "Value messag" -->

## Output logprobs

For buffered greedy output from a text or inline PNG prompt, include
`["message.output_text.logprobs"]` and set `temperature:0`. `top_logprobs` accepts 0
through 20 and defaults to 0. The output text part then carries selected token
scores with UTF-8 bytes and alternatives.
The score rows cover all generated IDs, including reasoning IDs when the reasoning
parser emits a separate item. Without the include, the `logprobs` field remains null.
The effective generation budget, after the two-token reservation for explicit budgets,
times `1 + top_logprobs` may be at most 8,192.
Inline PNG requests use target-only decoding. Text requests use the
[existing logprob fallback policy](openai-logprobs.md). Requests for output
logprobs with streaming, sampling or a response constraint return HTTP 400.
Stored scored responses use the same retention limit as other responses.

## JSON Schema output

For text requests, set `text.format` to a flat JSON Schema descriptor.

```json
{
  "model": "Qwen3.8-27B-FP8",
  "input": "What is 6 times 7? Answer only with the number.",
  "reasoning": {"effort": "none"},
  "temperature": 0,
  "max_output_tokens": 48,
  "store": false,
  "text": {
    "format": {
      "type": "json_schema",
      "name": "answer",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": false
      }
    }
  }
}
```

The descriptor requires a `schema` object and a `name` of 1 to 64 ASCII letters, digits,
underscores or hyphens. Supply `strict:true` or omit it. An optional `description` can contain
up to 256 UTF-8 bytes. The server validates the descriptor and compiles the
[native JSON Schema subset](../output-constraints.md#json-schema) with the checkpoint
tokenizer. Other formats such as `json_object`, unsupported schema keywords, text stops,
and output budgets below the grammar minimum return HTTP 400 before SSE opens. The explicit
budget still reserves two tokens. [Descriptor validation](../../apps/server/responses.cpp#L393)<!-- prefix: "OrderedObjec" -->

The response echoes the descriptor in `text.format` and returns the generated JSON as an
`output_text` string in the assistant message. The schema constrains output from the first
generated token. Reasoning items, summaries and reasoning SSE events are unavailable with this
format, including when `reasoning.effort` is omitted or set to `medium`.
[Reasoning parser setup](../../apps/server/qwen/qwen_serving_profile.cpp#L106)<!-- prefix: "if (request." -->

For streaming, join the `response.output_text.delta` strings and parse the JSON after
`response.completed`.
The [checkpoint wire check](../../tests/constraints/responses_schema_http.py#L87)<!-- prefix: "def check_ou" -->
compares blocking text, streamed deltas and the terminal snapshot.

## Named function tools

Start the server with `--tool-call-parser qwen3_coder`. Declare one flat function in
`tools` and select it with `tool_choice`. For example, add these fields to a text request
asking for weather in Paris.

```json
{
  "tools": [{
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"],
      "additionalProperties": false
    }
  }],
  "tool_choice": {"type": "function", "name": "get_weather"}
}
```

Use a name of 1 to 64 ASCII letters, digits, underscores or hyphens. Omit `strict` or use
`true`. The [native chat tool grammar](../output-constraints.md#tool-call-constraints) checks
the parameter schema. The server emits a single call in `output` with `type:"function_call"`,
a `call_id`, a function name, and an `arguments` string containing a typed JSON object.
The client executes the function.

With `stream:true`, the function item is added with empty arguments. The server sends
`response.function_call_arguments.delta` when the parser has the arguments, then sends
`response.function_call_arguments.done` and the completed output item. The final response
snapshot includes the same item and usage. [Checkpoint wire check](../../tests/constraints/responses_named_tool_http.py#L141)<!-- prefix: "def run(args" -->

To return a result, send an `input` item with `type:"function_call_output"`, the matching
`call_id`, and a string `output`. Supply `previous_response_id` for a stored call. For
stateless replay, include the original user message and `function_call` item before the
result. Use a fresh `tools` declaration and named choice if the new response should call
a function again.

With a declared tool, automatic, required and disabled selection, multiple functions,
and hosted tools return HTTP 400. Named calls also return HTTP 400 with inline images,
JSON Schema output, text stops or output logprobs. Requests with an empty tool list accept
`tool_choice:"auto"` and `tool_choice:"none"`.

## State and retention

`--enable-response-store` enables `GET /v1/responses/{id}`,
`POST /v1/responses/{id}/cancel`, `previous_response_id`, and background requests.
Omitting `store` defaults to `true`. Both `store:false` and `store:null` opt out of retention
and echo their supplied values. Background requests require `store:true`. A follow-up inherits
the stored messages and output. Supply new top-level `instructions` for each request that
needs them. `DELETE` returns HTTP 405. A duplicate ID for a new stored record fails with HTTP
409, or with `response_id_conflict` if SSE has started, and preserves the original record.

With `background:true` and `stream:false`, a worker sends a queued acknowledgment and
continues if the client disconnects. Poll the ID for the terminal result or cancel it through
the cancel endpoint. Streaming stays attached to the HTTP connection, including requests
with `background:true` and `stream:true`. The store uses process memory and expires on restart.

The store retains up to 128 records, with at most 256 KiB of serialized response and replay
history per record. Capacity pressure evicts the oldest terminal record by insertion order.
Queued and in-progress records stay available. If all records are active, the server returns
HTTP 503 for a new record. Oversized stored requests get HTTP 413 before streaming or
acknowledgment. If a result crosses the storage limit during generation, a buffered request
gets HTTP 413, an open stream ends with `response.failed` and `response_store_limit`, and an
acknowledged background request remains retrievable with `status:"failed"` and that error
code. A queued acknowledgment confirms admission. Poll for the result of generation and
retention. Failure snapshots have an empty `output` and null `usage`, even when a client has
already received item events. Internal generation failures use a generic `server_error`.

Responses create, retrieve and cancel take precedence over configured Ollama aliases at the
same method and path. Other `include` expansions and automatic truncation return HTTP 400.
