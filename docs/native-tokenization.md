# Native tokenization

Send `POST /tokenize` or `POST /detokenize` to convert between raw text and
checkpoint token IDs. `/v1/tokenize` and `/v1/detokenize` are aliases.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP transport | Match the routes, validate the media type, and admit requests to the shared queue. | [Route aliases](../apps/server/http.cpp#L297)<!-- prefix: "Tokenization" --> · [Media type](../apps/server/http.cpp#L192)<!-- prefix: "if (tokeniza" --> · [Queue admission](../apps/server/prefix_controls.cpp#L320)<!-- prefix: "void Connect" --> |
| Tokenization worker | Validate request fields and encode or decode with the checkpoint tokenizer. | [Request handler](../apps/server/tokenization.cpp#L300)<!-- prefix: "void serve_t" --> · [Tokenizer load](../apps/server/text_request.cpp#L471)<!-- prefix: "const Tokeni" --> |

## Runtime path

Send a JSON object with `Content-Type: application/json`. If you include
`model`, use a string. The server always uses its configured checkpoint.
Requests share the [bounded HTTP queue](http-server.md#http-admission-and-shutdown).
An HTTP worker performs the conversion on the CPU without a decoder slot.
Both routes return one JSON response with HTTP 200 on success.

### Encode text

Set `prompt` to a UTF-8 string or an array of strings. A string returns flat
`tokens` and an integer `count`. An array returns nested `tokens` and an array
of counts in the same order, including for a singleton. Empty strings and
arrays are valid. Omit `messages` or set it to `null`. `add_special_tokens`
defaults to `true`. The [pinned tokenizer](text-serving.md#tokenizer-assets)
inserts no extra tokens with either flag value. [Encoding and response](../apps/server/tokenization.cpp#L79)<!-- prefix: "void tokeniz" -->

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

### Decode IDs

Set `tokens` to a flat array of IDs for a string result, or a nonempty array
of ID arrays for an ordered array of strings. An empty flat array returns
`{"text":""}` without loading the tokenizer, and `[[]]` returns
`{"text":[""]}`. `skip_special_tokens` defaults to `true`. Set it to `false`
to retain special tokens as text. Unmapped padding IDs from 248077 through
248319 contribute no text. [ID validation](../apps/server/tokenization.cpp#L189)<!-- prefix: "TokenId chec" --> · [Decoding and response](../apps/server/tokenization.cpp#L247)<!-- prefix: "void detoken" -->

```bash
curl --fail-with-body http://127.0.0.1:17384/v1/detokenize \
  -H 'Content-Type: application/json' \
  --data-binary '{"tokens":[[12675,220,96748],[248044],[]],"skip_special_tokens":false}'
```

```json
{"text":["Hi 世界","<|endoftext|>",""]}
```

`{"tokens":[12675,220,96748]}` returns `{"text":"Hi 世界"}`.

## Constraints and failure behavior

Use integer IDs in `[0, 248320)` and JSON booleans for the two flags.
Detokenization also accepts booleans, integral floating-point values in the
signed 64-bit range, and decimal integer strings as IDs. The same vocabulary
check applies. The flag parser also accepts `0` and `1` and case-insensitive
Boolean spellings such as `"false"` and `"yes"`, and rejects `null`.
`max_model_len` reports the checkpoint context of 262,144 tokens. The routes
apply the HTTP limits of 8 KiB for headers and 2 MiB for bodies
without a generation context budget. The JSON parser rejects duplicate
keys and nesting beyond 32 containers. [Vocabulary](../include/sglang/tokenizer/tokenizer.h#L13)<!-- prefix: "inline const" --> · [Context](../include/sglang/tokenizer/tokenizer.h#L16)<!-- prefix: "inline const" --> · [HTTP limits](../apps/server/http.cpp#L29)<!-- prefix: "constexpr st" --> · [JSON depth](../src/json.cpp#L83)<!-- prefix: "if (depth ==" -->

Route errors return a JSON object with `object: "error"`, `message`, `type`,
`param: null`, and an integer `code` matching the HTTP status. For example,
detokenizing `{"tokens":[248320]}` returns HTTP 400 with this body.
[Error writer](../apps/server/http.cpp#L455)<!-- prefix: "void write_t" -->

```json
{"object":"error","message":"Invalid input: token id 248320 is out of vocabulary; valid range is [0, 248320).","type":"BadRequestError","param":null,"code":400}
```

| HTTP status | When it occurs and what to do |
| --- | --- |
| 400 `Bad Request` | Fix malformed JSON, an invalid field or shape, or a missing or incorrect media type. |
| 400 `BadRequestError` | Correct the reported ID and resend the whole request. |
| 413 or 431 | Reduce the request body or headers, respectively. |
| 501 `NotImplementedError` | Remove a truthy `stream` field from `/tokenize` to receive JSON. `/detokenize` ignores `stream`. |
| 503 `ServiceUnavailableError` | The HTTP queue is full. Retry when capacity is available. |
| 500 `InternalServerError` | Check the server log and checkpoint `tokenizer.json`. After a load failure, restore the file and resend. The next request that needs it retries loading. |

## Verification

[Start the server](../README.md#serve-token-ids-over-http) with the pinned
checkpoint. Run both commands in [Runtime path](#runtime-path). Require
HTTP 200 and the documented fields and values for each response.

Send the out-of-vocabulary example and require the documented HTTP 400 body.
Repeat a valid request on the same server. Require HTTP 200 and the same
result as before the error.
