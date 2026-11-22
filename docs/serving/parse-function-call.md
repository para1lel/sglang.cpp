# Parse function calls from text

Send `POST /parse_function_call` to extract function calls from supplied Qwen-style text. The server returns one JSON response.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP reader | Check the media type and dispatch the request. | [Media types](../../apps/server/http.cpp#L117)<!-- prefix: "bool json_me" --> · [Dispatch](../../apps/server/prefix_controls.cpp#L619)<!-- prefix: "if (request." --> |
| Native parser | Validate the request, select the detector, and return the parsed text and calls. | [Request handler](../../apps/server/parse_function_call.cpp#L543)<!-- prefix: "void parse_f" --> · [Text parser](../../apps/server/parse_function_call.cpp#L506)<!-- prefix: "std::string " --> |
| SGLang reference | Define the upstream route and detector aliases. | [Pinned route](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1739)<!-- prefix: "@app.post(\"/" --> · [Pinned detector registry](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/function_call/function_call_parser.py#L110)<!-- prefix: "\"qwen3_coder" --> |

## Runtime path

The HTTP reader parses the supplied text without a completion slot, tokenizer call, or model inference. The request selects the detector independently of the [chat parser flags](../chat-serving.md#qwen-reasoning-and-tool-calls).

With a nonempty `tools` array and a literal `<tool_call>` marker, `normal_text` contains the exact text before the first marker. Otherwise, `normal_text` contains the full input and `calls` is empty.

The parser reads `<function=name>` and `<parameter=key>` tags. It uses the matching function schema to convert arguments. It can return an unlisted function and can recover calls from incomplete markup.

Each entry in `calls` has a zero-based `tool_index` in output order, a `name`, and a serialized argument string in `parameters`.

## Constraints and failure behavior

Send a JSON object within the [shared HTTP limits](../http-server.md#http-admission-and-shutdown). A nonempty body requires `application/json` or an `application` subtype ending in `+json`. Media type matching ignores ASCII case and parameters.

| Field | Accepted values |
| --- | --- |
| `text` | Required string. |
| `tool_call_parser` | Select `qwen3_coder`, `step3p5`, or `nanbeige`. All three names select the same detector and are case-sensitive. |
| `tools` | Optional array. Each entry has a `function` object. Use `function.name` and `function.parameters.properties` to declare argument types. The optional `type` defaults to `function`. |

| Condition | Response |
| --- | --- |
| Invalid JSON, request field, or media type | HTTP 400 with a JSON error. |
| Missing, null, or unsupported parser name | HTTP 400 with a JSON error. |
| Method other than POST on the canonical path | HTTP 405 with `Allow: POST`. |
| Trailing slash | HTTP 307 under the [shared redirect rules](../http-server.md#request-targets-and-redirects). |

### Compatibility limits

For object and array arguments, the parser tries JSON first, then a subset of Python literals. The subset includes dictionaries with string keys, lists, tuples, quoted strings, decimal numbers, `True`, `False`, and `None`. For example, `{'days': (1, 2), 'active': True}` becomes `{"days": [1, 2], "active": true}`. Inputs with non-string dictionary keys, hexadecimal numbers, or byte strings stay strings. [Native structured conversion](../../apps/server/parse_function_call.cpp#L379)<!-- prefix: "bool append_" --> · [Native literal parser](../../apps/server/tool_literal.cpp#L236)<!-- prefix: "std::optiona" --> · [SGLang literal fallback](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/function_call/qwen3_coder_detector.py#L181)<!-- prefix: "param_value " -->

For a string argument with `anyOf` or `oneOf`, the parser checks JSON objects and arrays against the schema. It converts a value only when it can confirm a match. For `oneOf`, exactly one branch must match. For a tag such as `"kind": "city"`, add `kind` to the branch's `required` list. Set a `const` or scalar `enum` constraint on the `kind` property. The parser treats `discriminator` as an annotation.

Union conversion accepts JSON input and uses a subset of JSON Schema Draft 2020-12. For an array branch with `minItems: 2`, `[8, 9]` meets the length constraint. With only string and array branches, `[8]` stays a string. [Native union matcher](../../apps/server/tool_union.cpp#L305)<!-- prefix: "bool matches" --> · [Array length check](../../apps/server/tool_union.cpp#L247)<!-- prefix: "if (key == \"" -->

The matcher resolves `$ref` from `function.parameters` using `#` or an object-key JSON Pointer such as `#/$defs/Lookup`. It supports `~0` and `~1` escapes. External references, named anchors, percent-encoded paths, array indexes, and nested `$id` scopes are outside this subset. A value stays a string if the parser cannot confirm a match within its supported rules, depth limit, and work budget. [Reference resolution](../../apps/server/tool_union.cpp#L163)<!-- prefix: "const json::" --> · [Depth and work limits](../../apps/server/tool_union.cpp#L155)<!-- prefix: "Match UnionM" --> · [SGLang union conversion](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/function_call/utils.py#L467)<!-- prefix: "def convert_" -->

The request body rejects bare `NaN` and `Infinity`. Numeric argument text can produce these tokens inside `parameters`, where strict JSON decoders reject them. A floating-point value outside the JSON decoder's range causes HTTP 400 in the request body. If a structured argument contains such a value, the parser keeps the original argument text as a string. [JSON number parser](../../src/json.cpp#L192)<!-- prefix: "Value Parser" --> · [Argument number writer](../../apps/server/parse_function_call.cpp#L257)<!-- prefix: "void append_" -->

Validation messages can differ from SGLang.

## Verification

Start the native server with the [launch procedure](../../README.md#serve-token-ids-over-http). The chat parser flags are optional.

```bash
curl --fail-with-body http://127.0.0.1:17384/parse_function_call \
  -H 'Content-Type: application/json' --data-binary @- <<'JSON'
{
  "text": "I will check.\n<tool_call>\n<function=get_forecast>\n<parameter=city>Paris</parameter>\n<parameter=days>2</parameter>\n</function>\n</tool_call>",
  "tool_call_parser": "qwen3_coder",
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_forecast",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}
      }
    }
  }]
}
JSON
```

Expect HTTP 200 with these JSON values.

```json
{
  "normal_text": "I will check.\n",
  "calls": [{
    "tool_index": 0,
    "name": "get_forecast",
    "parameters": "{\"city\": \"Paris\", \"days\": 2}"
  }]
}
```

Repeat with `step3p5` and `nanbeige` to check that both aliases give the same response.
