# Native output constraints

Native text and text-only chat accept [finite-choice EBNF](text-serving.md#finite-choice-constraints) or a regular expression. Pick one constraint per request. Both JSON and SSE use the same constraint.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP request | Validate the constraint fields and compile with the checkpoint tokenizer. | [Request validation](../apps/server/text_request.cpp#L121)<!-- prefix: "void set_con" --> |
| Regular expression compiler | Build the token transition table and compute the minimum budget. | [Tokenizer compiler](../src/tokenizer/regex_constraint.cpp#L67)<!-- prefix: "std::shared_" --> |
| Decoder | Apply the request cursor to target scores and commit the selected ID. | [Selection](../src/decoder_constraint.cpp#L28)<!-- prefix: "void select_" --> · [Cursor update](../include/sglang/decoder_constraint.h#L79)<!-- prefix: "bool commit(" --> |

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

### Selection and finish

Each request owns its cursor. The cursor excludes tokens that cannot lead to a match within the remaining budget. With up to 4,096 allowed IDs, CUDA gathers their BF16 scores. For larger sets it masks disallowed scores to negative infinity in a vocabulary-sized row. Both routes use the usual greedy tie rule or [sampling](text-serving.md#sampling) over the allowed scores. Selection runs after eager scoring or eligible graph replay.

A match with no continuation stops immediately. Acceptance at the last allowed token also stops. Neither case adds EOS. With budget left at a match that can continue, either model EOS ID can compete with valid continuations. A selected EOS counts in usage and diagnostic IDs. Its text is omitted. A nullable regex with `max_tokens: 0` returns no IDs and `finish_reason: "length"`. A completed match during decoding returns `stop`.

HTTP 400 covers conflicting constraints, an invalid pattern, compiler limits, and an impossible token budget. Omit `stop_token_ids` entirely. Leave `ignore_eos` false. Constraints use K0 even when the server has positive MTP depth. The benchmark model and HTTP media chat reject regex. A full prefix hit uses the new request's constraint and seed for its first eager selection. [Prefix reuse](prefix-cache.md#runtime-path)

C++ callers can use `TokenConstraint::compile_regex(tokenizer, pattern)` with the checkpoint tokenizer. Use the [shared C++ constraint options](text-serving.md#finite-choice-constraints) and set `mtp_steps = 0`.

## Verification

Run the release server with the pinned checkpoint. Save an example body as `request.json`. Set `ENDPOINT` to `/v1/completions` or `/v1/chat/completions` for that body.

```bash
curl -sS -N -H 'Content-Type: application/json' --data-binary @request.json \
    "${SERVER_URL:-http://127.0.0.1:17384}${ENDPOINT:?set the endpoint}"
```

Repeat with `stream: true`, `return_token_ids: true`, and `stream_options: {"include_usage": true}`. Join the streamed deltas. Compare the complete text with the pattern. Check usage and finish behavior against the committed IDs. Include shared prefixes, nullable patterns, split UTF-8, exact-budget acceptance, and both EOS IDs.

Use the pinned tokenizer for a CPU oracle of allowed continuations. Compare sparse and wide selection with a score oracle on identical BF16 rows. Include a fixed seed and ties. On a reserved B200, trace eager and graph routes with mixed K0 lanes. Change the constraint on a full prefix hit and compare with its cold control. Require HTTP 400 for invalid syntax, conflicts, and an impossible budget. Record compilation time separately from prefill and decode.
