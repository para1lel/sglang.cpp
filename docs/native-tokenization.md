# Native tokenization

Send `POST /tokenize` to encode text or one chat conversation. Send
`POST /detokenize` to decode checkpoint token IDs to text.
`/v1/tokenize` and `/v1/detokenize` are aliases.
Send `POST /v1/messages/count_tokens` to count a rendered Anthropic-format
prompt for the configured Qwen checkpoint.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP transport | Match the routes, validate the media type, and admit requests to the shared queue. | [Route aliases](../apps/server/http.cpp#L389)<!-- prefix: "Tokenization" --> · [Media type](../apps/server/http.cpp#L208)<!-- prefix: "if ((tokeniz" --> · [Queue admission](../apps/server/prefix_controls.cpp#L642)<!-- prefix: "void Connect" --> |
| Tokenization worker | Validate inputs and encode or decode with the checkpoint tokenizer. | [Request handler](../apps/server/tokenization.cpp#L900)<!-- prefix: "void serve_t" --> · [Tokenizer load](../apps/server/text_completion.cpp#L20)<!-- prefix: "const Tokeni" --> |
| Chat renderer | Render the Qwen3.8 checkpoint format before encoding chat. | [Chat tokenization](../apps/server/tokenization.cpp#L598)<!-- prefix: "std::vector<" --> · [Renderer](../apps/server/chat_template.cpp#L538)<!-- prefix: "std::string " --> |

## Runtime path

Send a JSON object with `Content-Type: application/json`. If you include
`model` on the encode or decode routes, use a string.
Requests share the [bounded HTTP queue](http-server.md#http-admission-and-shutdown).
An HTTP worker performs the conversion on the CPU without a decoder slot.
Each route returns one JSON response with HTTP 200 on success.

### Encode text

Supply one non-null field, `prompt` or `messages`.
Set `prompt` to a UTF-8 string or an array of strings. A string returns flat
`tokens` and an integer `count`. An array returns nested `tokens` and an array
of counts in the same order, including for a singleton. Empty strings and
arrays are valid. Omit `messages` or set it to `null`. `add_special_tokens`
defaults to `true`. The [pinned tokenizer](text-serving.md#tokenizer-assets)
inserts no extra tokens with either flag value. [Encoding and response](../apps/server/tokenization.cpp#L655)<!-- prefix: "void tokeniz" -->

```bash
curl --fail-with-body http://127.0.0.1:17384/tokenize \
  -H 'Content-Type: application/json' \
  --data-binary '{"prompt":"Hi 世界"}'
```

```json
{"tokens":[12675,220,96748],"count":3,"max_model_len":262144}
```

`{"prompt":["Hi","世界",""]}` returns
`{"tokens":[[12675],[96748],[]],"count":[1,1,0],"max_model_len":262144}`.

### Encode chat

Set `messages` to one nonempty array of message objects. Omit `prompt` or
set it to `null`. The response has flat `tokens`, an integer `count`, and
`max_model_len`. [Chat input](../apps/server/tokenization.cpp#L524)<!-- prefix: "ChatTokeniza" -->

```bash
curl --fail-with-body http://127.0.0.1:17384/v1/tokenize \
  -H 'Content-Type: application/json' \
  --data-binary '{"messages":[{"role":"user","content":"你好 🌍 café"}],"reasoning_effort":"medium"}'
```

```json
{"tokens":[248045,846,198,109266,10838,234,235,50203,248046,198,248045,74455,198,248068,198],"count":15,"max_model_len":262144}
```

The default opens an assistant turn. With `add_generation_prompt` omitted
or `null`, the server first treats a final string-content assistant turn as
a user turn. Explicit `true` opens a new assistant turn after the history.
Explicit `false` keeps the history and leaves out the new assistant turn.
For top-level `continue_final_message: true`, send a final assistant string
after at least one earlier message. The server encodes that string separately.
It appends the resulting IDs to the open assistant turn. Use
`add_generation_prompt: true` or the default for this setting.
[Chat normalization and encoding](../apps/server/tokenization.cpp#L598)<!-- prefix: "std::vector<" -->

Set `add_generation_prompt: false` to continue a message within the template.
Set `chat_template_kwargs.continue_final_message` to `true` to end the prompt
at the final message's string content or last text part. Set it to
`"reasoning_content"` to end the prompt at that field when the template renders it.
[Template continuation](../apps/server/tokenization.cpp#L472)<!-- prefix: "std::optiona" -->

The checkpoint defaults to thinking enabled with `xhigh` effort. With
thinking enabled, the template uses `xhigh`, `medium`, or `low`. Use
`chat_template_kwargs.reasoning_effort` to override the top-level effort.
Use `chat_template_kwargs.enable_thinking: false` to close the thinking
block in the assistant opener. Set `chat_template_kwargs.preserve_thinking`
to `false` to drop thinking from assistant turns before the last user query.
[Template options](../apps/server/tokenization.cpp#L397)<!-- prefix: "json::Ordere" -->

The renderer accepts text parts, assistant function calls, tool results,
and request-level function definitions. `tool_choice` can select a named
function or control which definitions appear in the prompt.

In assistant tool history, `function.arguments` can contain an object or a
string that encodes an object. The request parser keeps integers in objects
exact. For strings, integers outside both 64-bit integer ranges round to
binary64 if the result is finite. Floating-point underflow can produce zero.
[Argument parsing](../apps/server/tokenization.cpp#L305)<!-- prefix: "json::Value " --> · [Number parsing](../src/json.cpp#L195)<!-- prefix: "Value Parser" -->

Image and video parts produce template placeholders. Set
`chat_template_kwargs.add_vision_id: true` to number them. The HTTP worker
performs no media I/O or placeholder expansion.

Pinned Python SGLang converts the tokenization request to a chat request
and uses its chat message processing path. To forward `add_special_tokens: true`,
set that flag to `true` and supply a Boolean `add_generation_prompt`.
The native renderer emits the pinned checkpoint's chat markers.
Both `add_special_tokens` values produce the same IDs with this tokenizer.
[Python conversion](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/openai/protocol.py#L1917)<!-- prefix: "def to_chat_" --> · [Python tokenization](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/openai/serving_tokenize.py#L92)<!-- prefix: "def _tokeniz" -->

### Decode IDs

Set `tokens` to a flat array of IDs for a string result, or a nonempty array
of ID arrays for an ordered array of strings. An empty flat array returns
`{"text":""}` without loading the tokenizer, and `[[]]` returns
`{"text":[""]}`. `skip_special_tokens` defaults to `true`. Set it to `false`
to retain special tokens as text. Unmapped padding IDs from 248077 through
248319 contribute no text. [ID validation](../apps/server/tokenization.cpp#L775)<!-- prefix: "TokenId chec" --> · [Decoding and response](../apps/server/tokenization.cpp#L833)<!-- prefix: "void detoken" -->

```bash
curl --fail-with-body http://127.0.0.1:17384/v1/detokenize \
  -H 'Content-Type: application/json' \
  --data-binary '{"tokens":[[12675,220,96748],[248044],[]],"skip_special_tokens":false}'
```

```json
{"text":["Hi 世界","<|endoftext|>",""]}
```

`{"tokens":[12675,220,96748]}` returns `{"text":"Hi 世界"}`.

### Count Anthropic prompt tokens

Send a nonempty `model` string and a `messages` array. The server always uses
its configured checkpoint. Messages accept `user`, `assistant`, and `system`
roles with string or block-array content. The adapter merges top-level
`system` text and system turns into one system message.

```bash
curl --fail-with-body http://127.0.0.1:17384/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen3.8-27B-FP8","messages":[{"role":"user","content":"Hello, 你好 🌍 café"}]}'
```

```json
{"input_tokens":60}
```

`input_tokens` counts the complete rendered prompt. It includes role and
assistant-generation markers, text and search-result blocks, selected custom
tool definitions, supported tool history, and image placeholders. The server
does no media I/O, tool execution, or decoder work. `tool_choice` accepts
`auto`, `any`, `tool`, or `none`. Built-in server tool declarations contribute
no prompt tokens and cannot be selected. `betas` accepts strings and has no
prompt effect. In this request, `stream` has no effect.
[Anthropic conversion](../apps/server/anthropic_count.cpp#L693)<!-- prefix: "AnthropicCou" --> ·
[Render and count](../apps/server/tokenization.cpp#L625)<!-- prefix: "void serve_a" -->

The Qwen template enables thinking by default. With `--reasoning-parser qwen3`,
the adapter keeps assistant thinking history and accepts explicit `enabled`
or `adaptive` thinking. The default parser setting drops thinking history and
rejects those two explicit modes. For the example, `thinking: {"type":"disabled"}`
returns 20 with `qwen3` and 60 with the default parser setting.
`output_config.effort` can change the count. Redacted assistant thinking and
tool references forwarded from tool results fail validation for this checkpoint.
[Thinking options](../apps/server/anthropic_count.cpp#L621)<!-- prefix: "bool thinkin" -->

The route requires `Content-Type: application/json`. A supported request to
the trailing-slash form receives HTTP 307. Other methods on the canonical
path receive HTTP 405 with `Allow: POST`. Malformed content or invalid fields
receive HTTP 400 with an `invalid_request_error`. For example, omit `model`
to receive this body.
[Method check](../apps/server/prefix_controls.cpp#L700)<!-- prefix: "if ((count |" --> ·
[Anthropic error body](../apps/server/http.cpp#L651)<!-- prefix: "void write_a" -->

```json
{"type":"error","error":{"type":"invalid_request_error","message":"model: Field required"}}
```

Error messages can differ from the pinned Python SGLang wording. For the
malformed body `{`, native returns `Invalid JSON request body` and Python
SGLang returns `1: JSON decode error`. Queue overload returns HTTP 503 with
`api_error`. Internal tokenization failures return HTTP 500 with `api_error`.

## Constraints and failure behavior

Use integer IDs in `[0, 248320)` and JSON booleans for flags.
Detokenization also accepts booleans, integral floating-point values in the
signed 64-bit range, and decimal integer strings as IDs. The same vocabulary
check applies. The flag parser also accepts `0` and `1` and case-insensitive
Boolean spellings such as `"false"` and `"yes"`. `add_special_tokens` and
`skip_special_tokens` reject `null`.
`max_model_len` reports the checkpoint context of 262,144 tokens. The routes
apply the HTTP limits of 8 KiB for headers and 2 MiB for bodies. The
tokenizer returns all encoded IDs on encode routes. The JSON parser limits
nesting to 32 containers. Anthropic counting uses the last value of duplicate
keys. Encode and decode requests reject duplicate keys. [Vocabulary](../include/sglang/tokenizer/tokenizer.h#L13)<!-- prefix: "inline const" --> · [Context](../include/sglang/tokenizer/tokenizer.h#L16)<!-- prefix: "inline const" --> · [HTTP limits](../apps/server/http.cpp#L30)<!-- prefix: "constexpr st" --> · [JSON depth](../src/json.cpp#L86)<!-- prefix: "if (depth ==" -->

In chat tokenization bodies, decimal and exponent numbers can round to signed
zero. For example, `1e-999` and `-1e-999` produce `0.0` and `-0.0`, as in
pinned Python. The chat body parser rejects overflow. Raw prompt tokenization,
detokenization, and legacy chat completions reject both underflow to zero and
overflow. [Chat body parsing](../apps/server/tokenization.cpp#L885)<!-- prefix: "json::Value " --> ·
[Number parsing](../src/json.cpp#L195)<!-- prefix: "Value Parser" -->

Send at least one user query. If you include a system turn, put it first.
Chat tokenization is limited to the pinned Qwen3.8 template. It excludes
audio, message-level tools, `input_ids`, request-supplied templates,
other model adapters, and generation grammars. Native normalizes tool schema
aliases such as `varchar` and checks selected keywords such as `type` and
`required`. Pinned Python performs full JSON Schema Draft 2020-12 validation.
[Native tool schema validation](../apps/server/tokenization_validation.cpp#L276)<!-- prefix: "Value normal" --> ·
[Python tool schema validation](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/openai/serving_chat.py#L1101)<!-- prefix: "normalize_js" -->

Chat checks the form of messages, including media metadata and assistant
history fields, before it rejects streaming. For non-streaming chat, native
also checks the completion token budget and selected `response_format` schemas.
It requires `return_meta_info: true` when `return_sampling_mask` is enabled.
Native supports the template options described in [Encode chat](#encode-chat).
Validation diagnostics can differ from pinned Python.
[Chat input](../apps/server/tokenization.cpp#L524)<!-- prefix: "ChatTokeniza" --> · [Generation field checks](../apps/server/tokenization_validation.cpp#L244)<!-- prefix: "void validat" -->

Encode and decode route errors return a JSON object with `object: "error"`, `message`, `type`,
`param: null`, and an integer `code` matching the HTTP status. For example,
detokenizing `{"tokens":[248320]}` returns HTTP 400 with this body.
[Error writer](../apps/server/http.cpp#L630)<!-- prefix: "void write_t" -->

```json
{"object":"error","message":"Invalid input: token id 248320 is out of vocabulary; valid range is [0, 248320).","type":"BadRequestError","param":null,"code":400}
```

| HTTP status | When it occurs and what to do |
| --- | --- |
| 400 `Bad Request` | Fix malformed JSON, an invalid field or shape, or a missing or incorrect media type. |
| 400 `BadRequestError` | Correct the reported ID or chat formatting error and resend the whole request. |
| 413 or 431 | Reduce the request body or headers, respectively. |
| 501 `NotImplementedError` | Remove a truthy `stream` field from `/tokenize` to receive JSON. `/detokenize` ignores `stream`. |
| 503 `ServiceUnavailableError` | The HTTP queue is full. Retry when capacity is available. |
| 500 `InternalServerError` | Check the server log and checkpoint `tokenizer.json`. For chat, also check the [chat template assets](chat-serving.md#constraints-and-failure-behavior). Restore an unreadable or mismatched asset before resending. The next request that needs it retries loading. |

## Verification

[Start the server](../README.md#serve-token-ids-over-http) with the pinned
checkpoint. Run the commands in [Runtime path](#runtime-path). Require
HTTP 200 and the documented fields and values for each response.

Repeat the chat request on `/tokenize`. Require the same response. Repeat
with `add_generation_prompt: false`. Require the first ten IDs and
`count: 10`. Send the chat request with `prompt: ""`. Require HTTP 400.

Send this request body to both tokenization aliases.

```json
{"messages":[{"role":"user","content":" trailing spaces  \n"}],"add_generation_prompt":false,"chat_template_kwargs":{"continue_final_message":true}}
```

Require HTTP 200, `count: 48`, and final IDs `[371,13868,12258]`.
Remove `add_generation_prompt` and require HTTP 400 on both aliases.

Send the out-of-vocabulary example and require the documented HTTP 400 body.
Repeat a valid request on the same server. Require HTTP 200 and the same
result as before the error.

Run the [Anthropic count CPU test](../tests/test_anthropic_count_cpu.py#L42)<!-- prefix: "class Anthro" -->
against the pinned checkpoint. It checks exact counts under the default and
`qwen3` parser settings, the error envelope, method dispatch, and redirects.
