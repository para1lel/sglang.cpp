# Native output constraints

Native text and text-only chat accept [finite-choice EBNF](text-serving.md#finite-choice-constraints), a regular expression, or the JSON Schema subset below. Pick one constraint per request. JSON and SSE use the same constraint. These whole-output constraints keep reasoning and tool markers in chat `content`. Text chat can also select a [tool-call constraint](#tool-call-constraints) that feeds the chat output parsers.
`POST /generate` accepts the same subsets through [`sampling_params.regex`](#native-generate), [`sampling_params.json_schema`](#native-json-schema) and [`sampling_params.ebnf`](#native-ebnf).

## Responsibilities and sources of truth

HTTP compiles constraints with the checkpoint tokenizer. The decoder filters each score row and advances the cursor on committed IDs. [Request validation](../apps/server/text_request.cpp#L286)<!-- prefix: "void set_con" --> · [Tokenizer compiler](../src/tokenizer/regex_constraint.cpp#L151)<!-- prefix: "std::shared_" --> · [Schema compiler](../src/tokenizer/json_schema.cpp#L423)<!-- prefix: "std::shared_" --> · [Tool grammar cache](../apps/server/qwen/qwen_tool_constraints.cpp#L69)<!-- prefix: "std::shared_" --> · [Trigger compiler](../src/tokenizer/regex_constraint.cpp#L164)<!-- prefix: "std::shared_" --> · [Selection](../src/decoder_constraint.cpp#L28)<!-- prefix: "void select_" --> · [Cursor update](../src/tokenizer/constraint.cpp#L162)<!-- prefix: "bool Constra" -->

## Runtime path

The same [selection rules](#selection-and-finish) apply at [positive MTP depths](mtp.md#constrained-mtp).

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

For chat, replace `prompt` with a `messages` array. [Chat request example](https://github.com/para1lel/sglang.cpp/blob/49a48f98e44223bd619e7b6f43569b45175504c0/docs/output-constraints.md#regular-expressions)

The regex matches the entire generated text. The leading `^` and trailing `$` are optional. Grouping with `(...)` or `(?:...)` has the same effect.

- Use literals, concatenation, `|`, `.`, bracket classes, ranges, or negated classes. The dot matches one Unicode scalar other than line feed.
- Use `?`, `*`, `+`, `{n}`, `{n,}`, or `{m,n}`. Each explicit bound is at most 64, and groups can nest up to 32 levels.
- Use ASCII classes `\d`, `\w`, and `\s` or their uppercase complements. Use `\n`, `\r`, `\t`, `\f`, `\v`, escaped punctuation, or scalar escapes `\xHH`, `\uHHHH`, and `\UHHHHHHHH`.

The pattern must contain 1 to 1,024 UTF-8 bytes after JSON decoding. Escape a regex backslash as `\\` in the JSON string. Hex escapes denote Unicode scalars. Lone surrogates and malformed UTF-8 are invalid. Lookaround, backreferences, inline flags, lazy quantifiers, and Unicode property escapes are unsupported. [Syntax parser](../src/tokenizer/regex_parse.cpp#L287)<!-- prefix: "Expression p" -->

Compilation allows up to 4,096 nondeterministic states, 256 deterministic states, and 4,194,304 token transitions. It builds transitions from checkpoint token bytes, including pieces that split UTF-8 scalars. A token can span several pattern positions. Only complete UTF-8 matches can accept.

### Native generate

Set `sampling_params.regex` in `POST /generate`. Each item can activate one of `regex`, `json_schema` and `ebnf`; omit a field or use null or `""` to disable it. Each item owns its cursor and budget. Broadcast one sampling object to share compilation, or pass an array to mix constraints or constrained and unconstrained items. [Native validation](../apps/server/native_generate.cpp#L688)<!-- prefix: "void set_gen" --> · [Batch example](https://github.com/para1lel/sglang.cpp/blob/49a48f98e44223bd619e7b6f43569b45175504c0/docs/output-constraints.md#native-generate)

An accepting state with no continuation finishes immediately, including below `min_new_tokens`. An early grammar finish reports `stop` with the last generated ID as `matched` and includes its text. A match at the token budget reports `length` unless a text stop or model EOS wins that step. While a match can continue, `min_new_tokens` masks model EOS until the minimum is reached. An explicit text stop can finish earlier and trim the displayed match.

For any active native constraint, omit `stop_token_ids` and leave `ignore_eos` false. Invalid constraints, field conflicts, insufficient budget or same-item generated logprobs return HTTP 400 before decoder admission or SSE headers. Unconstrained siblings can request scores. Native JSON and cumulative SSE retain their [usual shapes and batch indices](native-generate.md#json-and-sse).

### JSON Schema

Send `response_format` with `POST /v1/chat/completions`. For `POST /v1/completions`, replace `messages` with `prompt`. Chat returns the generated JSON as a string in `message.content`. Text returns it in `choices[0].text`. For SSE, join the text deltas before parsing the JSON. The [Responses route](serving/openai-responses.md#json-schema-output) uses the same schema subset with a flat `text.format` descriptor.

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

`response_format` contains exactly `type` and `json_schema`. The descriptor requires `schema`. It accepts optional `strict: true`, a nonempty `name` of at most 64 bytes, and `description` of at most 256 bytes. A supplied `strict: false` is invalid. HTTP rejects `json_object`, `text`, and other format types. [Envelope validation](../apps/server/text_request.cpp#L252)<!-- prefix: "const json::" -->

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

### Native JSON Schema

Set `sampling_params.json_schema` to the serialized [schema](#json-schema) itself, for example `json.dumps({"enum":["ready","pending"]})` in Python. The response carries generated JSON as a string in `text`. The [native admission and finish rules](#native-generate) apply. Zero budget is invalid. Text stops can trim the JSON. [Request example](https://github.com/para1lel/sglang.cpp/blob/49a48f98e44223bd619e7b6f43569b45175504c0/docs/output-constraints.md#native-json-schema)

### Native EBNF

Set `sampling_params.ebnf` to a [finite-choice grammar](text-serving.md#finite-choice-constraints), for example `root ::= "ready" | "pending"`. The [native admission and finish rules](#native-generate) apply. Text stops can trim the displayed literal.

Run `tests/text/test_native_ebnf.py` with a release server and the checkpoint. It checks literal and UTF-8 boundaries, mixed constraints, per-item cursors, and the top-level OpenAI form.

### Selection and finish

Each request owns its cursor. For whole-output constraints, the cursor excludes tokens that cannot lead to a match within the remaining budget. With up to 4,096 allowed IDs, CUDA gathers their BF16 scores. For larger sets it masks disallowed scores to negative infinity in a vocabulary-sized row. With adjusted FP32 scores, CUDA also masks disallowed IDs to negative infinity. Every path respects the cursor mask for greedy selection, with lowest-ID ties, or [sampling](text-serving.md#sampling). Selection runs after eager scoring or eligible K0 graph replay. Positive-depth requests use [constrained MTP](mtp.md#constrained-mtp).

For OpenAI routes, a match with no continuation or acceptance at the token cap reports `stop` without adding EOS. A match with no continuation can finish before `min_tokens`. With budget left at a match that can continue, model EOS IDs can compete once `min_tokens` is reached. A selected EOS counts in usage and diagnostics and omits its text. A nullable regex with `max_tokens: 0` returns no IDs and `length`. For `/generate`, use the [native finish rules](#native-generate).

HTTP 400 covers conflicting constraints, invalid input, compiler limits, and an impossible token budget. Omit `stop_token_ids` entirely. Leave `ignore_eos` false. OpenAI routes require omitting nonempty text stops when using a constraint. `/generate` applies its [native text stops and finish metadata](#native-generate). The benchmark model and HTTP media chat reject `regex` and `response_format`. At K0, a full prefix hit uses the new request's constraint and seed for its first eager selection. [Prefix reuse](prefix-cache.md#runtime-path)

C++ callers can use `TokenConstraint::compile_regex(tokenizer, pattern)` with the checkpoint tokenizer. Use the [shared C++ constraint options](text-serving.md#finite-choice-constraints).

### Tool-call constraints

The [tool choice contract](chat-serving.md#tool-choice) defines which functions are advertised and when the grammar applies. In strict auto, generation can emit prose and enters the grammar at `<tool_call>`. Required and named choices force a call frame after any open reasoning prelude. After a complete frame, parallel mode permits prose or another frame. Serial and named mode allow only model EOS if budget remains. [Trigger states](../src/tokenizer/regex_constraint.cpp#L164)<!-- prefix: "std::shared_" -->

A non-strict function in the grammar constrains the wrapper and function name. With `function.strict: true`, `parameters` is a closed object with up to eight properties in declaration order. Omit `additionalProperties` or set it to `false`. Names must be unique XML labels of 1–64 UTF-8 bytes without angle brackets or line breaks. List declared keys in `required`. Omitted or null `parameters` permits an empty argument object. [Function grammar](../apps/server/qwen/qwen_tool_constraints.cpp#L23)<!-- prefix: "std::string " --> · [Parameter compiler](../src/tokenizer/json_schema.cpp#L387)<!-- prefix: "std::string " -->

Use the [JSON Schema subset](#json-schema) for nested objects and arrays, which use compact JSON inside XML parameter values. Direct strings use raw text. Direct string enums reject line breaks and tool delimiters. Direct integers accept inclusive `minimum` and `maximum` within the signed 18-digit range. An integer enum cannot also set bounds. The schema depth, node, and expanded pattern limits apply. [Schema nodes](../src/tokenizer/json_schema.cpp#L358)<!-- prefix: "std::string " -->

The triggered compiler allows up to 4,096 nondeterministic states, 4,096 deterministic subsets before minimization, 1,024 final states, and 8,388,608 token transitions. Unsupported schemas and exceeded compiler limits return HTTP 400. The tokenizer caches up to eight compiled grammars in least-recently-used order. The key includes the generated pattern, choice mode, parallel setting, and open reasoning state. Cache lookup and cold compilation share one mutex. [Automaton limits](../src/tokenizer/regex_automaton.cpp#L350)<!-- prefix: "std::vector<" --> · [Token compiler](../src/tokenizer/regex_constraint.cpp#L201)<!-- prefix: "std::shared_" --> · [Tool grammar cache](../apps/server/qwen/qwen_tool_constraints.cpp#L69)<!-- prefix: "std::shared_" -->

The triggered cursor allows a token-budget cut within a tool frame. At a length cut, the parser suppresses pending wrapper markup. It can return a partial call after recognizing the opening function tag. Partial arguments can omit required keys or contain incomplete values. If the last allowed token closes the frame, the finish reason is still `length` in JSON and SSE. [Cursor budget](../src/tokenizer/constraint.cpp#L116)<!-- prefix: "void Constra" --> · [Length-cut recovery](../apps/server/qwen/qwen_chat_output.cpp#L594)<!-- prefix: "bool ToolFra" -->

## Verification

Run the release server with the pinned checkpoint. Save an example body as `request.json`. Set `ENDPOINT` to `/v1/completions` or `/v1/chat/completions` for that body.

```bash
curl -sS -N -H 'Content-Type: application/json' --data-binary @request.json \
    "${SERVER_URL:-http://127.0.0.1:17384}${ENDPOINT:?set the endpoint}"
```

Repeat with `stream: true`, `return_token_ids: true`, and `stream_options: {"include_usage": true}`. Join the streamed deltas. Compare the complete text with the pattern. Check usage and finish behavior against the committed IDs. Include shared prefixes, nullable patterns, split UTF-8, exact-budget acceptance, and both EOS IDs.

Use the pinned tokenizer for a CPU oracle of allowed continuations. Compare sparse and wide selection with a score oracle on identical BF16 rows. Include a fixed seed and ties. On a reserved B200, trace eager and graph routes with mixed K0 lanes. Change the constraint on a full prefix hit and compare with its cold control. Require HTTP 400 for invalid syntax, conflicts, and an impossible budget. Record compilation time separately from prefill and decode.

For JSON Schema, run the example on both endpoints. Parse the joined response text and verify the property order, required keys, and scalar values. Include optional keys before and after required keys, bounded arrays, and escaped Unicode. Try an unsupported keyword, `strict: false`, a conflicting constraint, and a budget too small for valid JSON. Require HTTP 400 before SSE begins for each invalid request.

### Positive-depth checks

Use the release server and pinned checkpoint on a reserved B200. Follow the [constraint integration procedure](../tests/constraints/README.md) at capacities one and four. Compare K1 and K4 with exact-policy K0 on both native HTTP endpoints. Include finite-choice EBNF, regex, and JSON Schema. Check greedy and seeded selection, JSON and SSE, diagnostic IDs, finish reasons, and usage. Investigate differing IDs by comparing the scores on matching token histories.

Check the trace for draft proposals and target verification. Enable graphs and check that constrained verification stays eager. At wider capacity, check mixed lanes, stream reset, and slot reuse. Run the C++ constraint probe at depths zero through four. Check acceptance, rollback, terminal proposals, EOS with budget left, and completion at the budget.

### Tool constraint checks

In default Pixi, set `build_dir` to the matching XMake release build. Set `evidence` to a writable run directory.

```bash
bash tests/constraints/build_tool_probe.sh "$PWD" "$build_dir" "$evidence/tool-probe"
python -B tests/constraints/tool_mask.py "$evidence/tool-probe" "$evidence/mask"
python -B tests/constraints/strict_schema_probe.py "$evidence/tool-probe" "$evidence/strict"
```

Check optional and required frames, serial and parallel continuations, ordered properties, nested collections, and compiler limits. Build the [CPU replay server](../tests/text/README.md#qwen-chat-output), then set `cpu_server` to its executable path.

```bash
PYTHONPATH=tests/text CUDA_VISIBLE_DEVICES='' python -B tests/constraints/tool_cut_replay.py \
    --server "$cpu_server" --evidence "$evidence/cuts"
```

Compare JSON and SSE at cuts through tool wrappers, names, values, and closing tags. On a reserved B200, check the [chat wire contract](chat-serving.md#qwen-output-checks) with the pinned checkpoint.
