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

## JSON Schema output

Set `text.format` to a flat JSON Schema descriptor.

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
budget still reserves two tokens. [Descriptor validation](../../apps/server/responses.cpp#L358)<!-- prefix: "OrderedObjec" -->

The response echoes the descriptor in `text.format` and returns the generated JSON as an
`output_text` string in the assistant message. The schema constrains output from the first
generated token. Reasoning items, summaries and reasoning SSE events are unavailable with this
format, including when `reasoning.effort` is omitted or set to `medium`.
[Reasoning parser setup](../../apps/server/text_request.cpp#L647)<!-- prefix: "if (request." -->

For streaming, join the `response.output_text.delta` strings and parse the JSON after
`response.completed`.
The [checkpoint wire check](../../tests/constraints/responses_schema_http.py#L87)<!-- prefix: "def check_ou" -->
compares blocking text, streamed deltas and the terminal snapshot.

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
same method and path. Hosted tools, tool calls, media input, output logprobs, requested
`include` expansions and automatic truncation are outside the text adapter.
