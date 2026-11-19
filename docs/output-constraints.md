# Native output constraints

Native text and text-only chat accept [finite-choice EBNF](text-serving.md#finite-choice-constraints), a regular expression, or the JSON Schema subset below. Pick one constraint per request. Both JSON and SSE use the same constraint.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP request | Validate the constraint fields and compile with the checkpoint tokenizer. | [Request validation](../apps/server/text_request.cpp#L179)<!-- prefix: "void set_con" --> |
| Regular expression compiler | Build the token transition table and compute the minimum budget. | [Tokenizer compiler](../src/tokenizer/regex_constraint.cpp#L67)<!-- prefix: "std::shared_" --> |
| Schema compiler | Convert the supported schema to a token constraint. | [Schema compiler](../src/tokenizer/json_schema.cpp#L267)<!-- prefix: "std::shared_" --> |
| Decoder | Apply the request cursor to target scores and commit the selected ID. | [Selection](../src/decoder_constraint.cpp#L28)<!-- prefix: "void select_" --> · [Cursor update](../include/sglang/decoder_constraint.h#L95)<!-- prefix: "bool commit(" --> |

## Runtime path

The server compiles the constraint before it starts generation. The decoder scores the request, filters candidates by its cursor, and updates the cursor after selection.

## Constraints and failure behavior

### Regular expressions

Send a `regex` string with `POST /v1/completions`.

```json
{
  "model": "Qwen3.8-27B-FP8",
  "prompt": "Choose a two-digit order number.",
  "regex": "^ORD-[0-9]{2}$",
  "max_tokens": 12
}
```

For `POST /v1/chat/completions`, send the same constraint with a `messages` array.

```json
{
  "model": "Qwen3.8-27B-FP8",
  "messages": [{"role": "user", "content": "Choose a two-digit order number."}],
  "regex": "^ORD-[0-9]{2}$",
  "max_tokens": 12
}
```

The regex matches the entire generated text. The leading `^` and trailing `$` are optional. Grouping with `(...)` or `(?:...)` has the same effect.

- Use literals, concatenation, `|`, `.`, bracket classes, ranges, or negated classes. The dot matches one Unicode scalar other than line feed.
- Use `?`, `*`, `+`, `{n}`, `{n,}`, or `{m,n}`. Each explicit bound is at most 64, and groups can nest up to 32 levels.
- Use ASCII classes `\d`, `\w`, and `\s` or their uppercase complements. Use `\n`, `\r`, `\t`, `\f`, `\v`, escaped punctuation, or scalar escapes `\xHH`, `\uHHHH`, and `\UHHHHHHHH`.

The pattern must contain 1 to 1,024 UTF-8 bytes after JSON decoding. Escape a regex backslash as `\\` in the JSON string. Hex escapes denote Unicode scalars. Lone surrogates and malformed UTF-8 are invalid. Lookaround, backreferences, inline flags, lazy quantifiers, and Unicode property escapes are unsupported. [Syntax parser](../src/tokenizer/regex_parse.cpp#L276)<!-- prefix: "Expression p" -->

Compilation allows up to 4,096 nondeterministic states, 256 deterministic states, and 4,194,304 token transitions. It builds transitions from checkpoint token bytes, including pieces that split UTF-8 scalars. A token can span several pattern positions. Only complete UTF-8 matches can accept.

### JSON Schema

Send `response_format` with `POST /v1/chat/completions`. For `POST /v1/completions`, replace `messages` with `prompt`. Chat returns the generated JSON as a string in `message.content`. Text returns it in `choices[0].text`. For SSE, join the text deltas before parsing the JSON.

```json
{
  "model": "Qwen3.8-27B-FP8",
  "messages": [{"role": "user", "content": "Is the order ready?"}],
  "max_tokens": 32,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "order_status",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"status": {"enum": ["ready", "pending"]}},
        "required": ["status"],
        "additionalProperties": false
      }
    }
  }
}
```

`response_format` contains exactly `type` and `json_schema`. The descriptor requires `schema`. It accepts optional `strict: true`, a nonempty `name` of at most 64 bytes, and `description` of at most 256 bytes. A supplied `strict: false` is invalid. HTTP rejects `json_object`, `text`, and other format types. [Envelope validation](../apps/server/text_request.cpp#L145)<!-- prefix: "const json::" -->

| Schema | Supported output |
| --- | --- |
| `object` | Up to eight properties in declaration order. Names are unique and at most 64 UTF-8 bytes. `required` names must be unique and declared. Other declared properties are optional. Omit `additionalProperties` or set it to `false`. |
| `array` | One `items` schema and integer bounds from 0 to 8. `minItems` defaults to 0. `maxItems` defaults to the greater of 4 and `minItems`. |
| `string`, `boolean`, `null` | JSON strings, `true` or `false`, or `null`. Strings allow valid UTF-8 and JSON escapes, including paired surrogate escapes. |
| `integer`, `number` | An optional minus and up to 18 integer digits. No leading zero on a multi-digit integer part. `number` also allows a decimal point and 1 to 12 fractional digits. Exponents are unsupported. |
| `enum` | One to 16 unique scalar choices. Choices may be strings of at most 128 UTF-8 bytes, integers accepted by the JSON parser, booleans, or null. A declared type must match. A `number` enum accepts integer entries. |

Output is compact JSON. Object keys follow `properties` order even if `required` lists them differently. The generator may omit optional properties and always omits undeclared ones. Annotations `title` and `description` allow strings up to 256 UTF-8 bytes. Unsupported constructs include `$ref`, unions such as `oneOf`, `type` arrays, `pattern`, and numeric bounds. Boolean schemas and unknown keywords are invalid.

The root counts as level one. A schema may have up to five levels and 64 nodes. The expanded pattern can use up to 8,192 bytes. Compilation allows up to 4,096 nondeterministic states and 1,024 deterministic states before minimization. After minimization, the 256-state and 4,194,304-token-transition limits apply. HTTP returns 400 for malformed schemas, unsupported keywords, compiler limits, and insufficient budget.

C++ callers can pass an ordered schema object to `TokenConstraint::compile_json_schema(tokenizer, schema)`. Parse the schema with `json::Syntax::ordered` to preserve property order.

### Selection and finish

Each request owns its cursor. The cursor excludes tokens that cannot lead to a match within the remaining budget. With up to 4,096 allowed IDs, CUDA gathers their BF16 scores. For larger sets it masks disallowed scores to negative infinity in a vocabulary-sized row. Both routes use the usual greedy tie rule or [sampling](text-serving.md#sampling) over the allowed scores. Selection runs after eager scoring or eligible graph replay.

A match with no continuation stops immediately. Acceptance at the last allowed token also stops. Neither case adds EOS. With budget left at a match that can continue, either model EOS ID can compete with valid continuations. A selected EOS counts in usage and diagnostic IDs. Its text is omitted. A nullable regex with `max_tokens: 0` returns no IDs and `finish_reason: "length"`. A completed match during decoding returns `stop`.

HTTP 400 covers conflicting constraints, invalid input, compiler limits, and an impossible token budget. Omit `stop_token_ids` entirely. Leave `ignore_eos` false. Omit nonempty text stops. Constraints use K0 even when the server has positive MTP depth. The benchmark model and HTTP media chat reject `regex` and `response_format`. A full prefix hit uses the new request's constraint and seed for its first eager selection. [Prefix reuse](prefix-cache.md#runtime-path)

C++ callers can use `TokenConstraint::compile_regex(tokenizer, pattern)` with the checkpoint tokenizer. Use the [shared C++ constraint options](text-serving.md#finite-choice-constraints) and set `mtp_steps = 0`.

## Verification

Run the release server with the pinned checkpoint. Save an example body as `request.json`. Set `ENDPOINT` to `/v1/completions` or `/v1/chat/completions` for that body.

```bash
curl -sS -N -H 'Content-Type: application/json' --data-binary @request.json \
    "${SERVER_URL:-http://127.0.0.1:17384}${ENDPOINT:?set the endpoint}"
```

Repeat with `stream: true`, `return_token_ids: true`, and `stream_options: {"include_usage": true}`. Join the streamed deltas. Compare the complete text with the pattern. Check usage and finish behavior against the committed IDs. Include shared prefixes, nullable patterns, split UTF-8, exact-budget acceptance, and both EOS IDs.

Use the pinned tokenizer for a CPU oracle of allowed continuations. Compare sparse and wide selection with a score oracle on identical BF16 rows. Include a fixed seed and ties. On a reserved B200, trace eager and graph routes with mixed K0 lanes. Change the constraint on a full prefix hit and compare with its cold control. Require HTTP 400 for invalid syntax, conflicts, and an impossible budget. Record compilation time separately from prefill and decode.

For JSON Schema, run the example on both endpoints. Parse the joined response text and verify the property order, required keys, and scalar values. Include optional keys before and after required keys, bounded arrays, and escaped Unicode. Try an unsupported keyword, `strict: false`, a conflicting constraint, and a budget too small for valid JSON. Require HTTP 400 before SSE begins for each invalid request.
