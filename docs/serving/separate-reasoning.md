# Separate reasoning from text

Send `POST /separate_reasoning` with text that already contains model output. The server returns the reasoning and ordinary text in one JSON response.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP reader | Match the path, check the media type, and dispatch the request. | [Path match](../../apps/server/prefix_controls.cpp#L660)<!-- prefix: "if (const au" --> · [Media types](../../apps/server/http.cpp#L117)<!-- prefix: "bool json_me" --> · [Dispatch](../../apps/server/prefix_controls.cpp#L662)<!-- prefix: "if (request." --> |
| Request and response | Validate the input, select the parser, and serialize the result. | [Request validation](../../apps/server/prefix_controls.cpp#L96)<!-- prefix: "void separat" --> · [Response writer](../../apps/server/prefix_controls.cpp#L70)<!-- prefix: "void write_s" --> |
| Qwen3 parser | Split the supplied text at literal reasoning and tool markers. | [Text split](../../apps/server/qwen/qwen_chat_output.cpp#L707)<!-- prefix: "std::optiona" --> · [Pinned SGLang Qwen3 detector](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/parser/reasoning_parser.py#L447)<!-- prefix: "class Qwen3D" --> |

## Runtime path

The HTTP reader splits the raw target at the first literal `?`, then percent-decodes the path once. For example, `POST /separate%5Freasoning?x=1` reaches `/separate_reasoning` and ignores the query. A trailing slash follows the [shared redirect rules](../http-server.md#request-targets-and-redirects).

The reader reads the supplied body under the ordinary [HTTP limits](../http-server.md#http-admission-and-shutdown). Parsing runs on the reader without a completion slot, tokenizer call, or model inference. The request selects its parser independently of the [chat output parser](../chat-serving.md#qwen-reasoning-and-tool-calls).

`qwen3`, `dots`, `interns1`, and `nanbeige` return the whole input as ordinary text if it has no literal `<think>`. `qwen3-thinking` and `minimax` start in reasoning even without that marker. Parser names are case-insensitive for ASCII and must match without leading or trailing spaces.

When parsing reasoning, the parser removes consecutive `<think>` markers at the start. An opener elsewhere stays in the output. The first complete `</think>` ends reasoning and is removed. If `<tool_call>` comes first, it ends reasoning and stays in ordinary text with the entire suffix. The parser preserves an incomplete marker at the end of the input and preserves whitespace. Tool text stays in the ordinary text string, regardless of whether the tool call is complete.

HTTP 200 always carries `reasoning_text` and `text` as strings. With `return_blocks` enabled, `reasoning_blocks` has one entry for nonempty reasoning and zero entries otherwise. `text_blocks` always holds one string, even when it is empty. `blocks` holds the same segments as objects with `type` and `text`. The optional `reasoning` object precedes the required `text` object.

## Constraints and failure behavior

Send a UTF-8 JSON object. A nonempty body requires `application/json` or an `application` subtype ending in `+json`. Media type matching ignores ASCII case and parameters.

| Field | Accepted values |
| --- | --- |
| `text`, `reasoning_parser` | Required strings. |
| `return_blocks` | Optional, default `false`. Accepts booleans, numeric `0` or `1`, and the strings `0`, `1`, `false`, `true`, `f`, `t`, `no`, `yes`, `n`, `y`, `off`, and `on` without regard to ASCII case. |
| `rid`, `http_worker_ipc` | Optional strings or null. The route validates these fields and leaves them unused. |

Unknown fields are ignored. If a key appears more than once, its last value applies. The route returns JSON even when the client requests SSE.

| Condition | Response |
| --- | --- |
| Trailing slash, including with an unsupported method | HTTP 307 with an empty body. `Location` uses the canonical path and preserves the original nonempty query with URL quoting. |
| Method other than POST on the canonical path | HTTP 405 with `Allow: POST` and `{"detail":"Method Not Allowed"}`. HEAD omits the body. |
| Invalid JSON, request shape, field, or media type | HTTP 400 with JSON fields `object: "error"`, `message`, `type: "Bad Request"`, `param: null`, and `code: 400`. |
| `deepseek-v4`, `deepseekv4`, `deepseek-v41`, or `deepseekv41` in exact lowercase | HTTP 400 with the same fields and `type: "400"`. These names require a model adapter. |
| Any other parser name outside the supported set | HTTP 500 with plain text `Internal Server Error`. |

The native contract follows the [pinned SGLang route](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1779)<!-- prefix: "@app.post(\"/" --> and [request definition](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/io_struct.py#L2742)<!-- prefix: "class Separa" --> for the supported Qwen3 detector names. Validation message text can differ.

## Verification

Start the native server with the [launch procedure](../../README.md#serve-token-ids-over-http). The server's chat parser flags are optional for this check.

```bash
curl --fail-with-body http://127.0.0.1:17384/separate_reasoning \
  -H 'Content-Type: application/json' \
  --data-binary '{"text":"<think>plan</think>answer","reasoning_parser":"qwen3","return_blocks":true}'
```

Require HTTP 200 and the following JSON values.

```json
{
  "reasoning_text": "plan",
  "text": "answer",
  "reasoning_blocks": ["plan"],
  "text_blocks": ["answer"],
  "blocks": [
    {"type": "reasoning", "text": "plan"},
    {"type": "text", "text": "answer"}
  ]
}
```

Repeat with omitted `return_blocks` and require only the two strings. With `qwen3-thinking`, check an input that begins with reasoning and has only a closing marker. Check an incomplete marker and a tool suffix for exact text preservation. For an invalid field or method, require the status and response shape in the failure table.

Repeat the POST at `/separate%5Freasoning?tag=a%2Fb` and require the same JSON. Send GET to that target and require HTTP 405 with `Allow: POST`. Send GET to `/separate%5Freasoning/?tag=a%2Fb` and require HTTP 307 with `Location: http://127.0.0.1:17384/separate_reasoning?tag=a%2Fb`.
