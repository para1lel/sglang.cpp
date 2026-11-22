# Native model information

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP control | Match `GET /model_info` and its deprecated alias `GET /get_model_info`. | [Control route](../apps/server/prefix_controls.cpp#L313)<!-- prefix: "if (!model_i" --> |
| Model discovery | List or retrieve the served model card. | [List dispatch](../apps/server/prefix_controls.cpp#L340)<!-- prefix: "if (request." --> · [Retrieval dispatch](../apps/server/prefix_controls.cpp#L325)<!-- prefix: "constexpr st" --> · [Shared card writer](../apps/server/prefix_controls.cpp#L244)<!-- prefix: "std::string " --> |
| Server | Report the selected checkpoint and configured serving properties. | [Metadata writer](../apps/sglang-server.cpp#L146)<!-- prefix: "std::string " --> |
| Checkpoint | Supply the supported model type and architecture. | [Checkpoint identity](../include/sglang/checkpoint.h#L47)<!-- prefix: "inline const" --> |

## Runtime path

The server prepares the response at startup after decoder construction. Both routes return the same HTTP 200 JSON object. They bypass the completion queue with prefix reuse off or on.

| Field | Native value |
| --- | --- |
| `model_path`, `tokenizer_path` | The directory supplied to `--checkpoint`. The [tokenizer](native-tokenization.md) loads `tokenizer.json` from this directory on demand. |
| `served_model_name` | `Qwen3.8-27B-FP8`, also the ID in [model discovery](#model-discovery). |
| `is_generation`, `has_image_understanding`, `has_audio_understanding` | JSON booleans `true`, `true`, and `false`, respectively. See [chat image input](chat-serving.md). |
| `preferred_sampling_params` | JSON `null`. [Chat sampling defaults](chat-serving.md#sampling-defaults-and-output-budget) still apply. |
| `weight_version`, `load_format` | Strings `"default"` and `"safetensors"`. |
| `reasoning_parser`, `tool_call_parser` | JSON `null` by default. When enabled, the fields show `"qwen3"` or `"qwen3_coder"`, respectively. |
| `disaggregation_mode` | String `"null"`. The native server runs prefill and decode together. |
| `model_type`, `architectures` | `"qwen3_5"` and `["Qwen3_5ForConditionalGeneration"]`. |

### Model discovery

Send an empty `GET /v1/models` to get HTTP 200 with `object: "list"` and one card in `data`. Send `GET /v1/models/Qwen3.8-27B-FP8` to get the card directly as an HTTP 200 JSON object.

The card uses `Qwen3.8-27B-FP8` for `id` and `root`. Use this ID for native chat and text completions. The card sets `object` to `model`, `owned_by` to `sglang`, and `parent` to `null`. The `created` field gives the response creation time in integer Unix seconds. The `max_model_len` field gives the 262,144-token context shared by prompt and output. Available GPU memory can reduce admitted capacity. The `benchmark` alias belongs to the [token-ID API](../benchmarks/streaming/README.md#wire-contract).

Pinned SGLang can list loaded LoRA adapters and retrieves its configured base served name. Native discovery exposes the single fixed checkpoint ID. [Pinned model list](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1998)<!-- prefix: "@app.get(\"/v" --> · [Pinned model retrieval](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L2030)<!-- prefix: "@app.get(\"/v" -->

## Constraints and failure behavior

Send an empty GET to the exact target. A query, trailing slash, or other unmatched target returns HTTP 404. An exact GET with a body also returns HTTP 404. An exact target with a different valid method returns HTTP 405 and `Allow: GET`. The response is `{"detail":"Method Not Allowed"}`. For HEAD, the server sends only the headers. Malformed HTTP still fails in the reader. [Method response](../apps/server/http.cpp#L465)<!-- prefix: "void write_g" --> · [Request line](../apps/server/http.cpp#L116)<!-- prefix: "HttpRequest " -->

Use [`/ready`](http-server.md) to check the decoder worker. Use a [tokenization request](native-tokenization.md#verification) to check tokenizer access.

### Discovery targets and errors

After HTTP parsing, the list route returns HTTP 404 for other methods, a nonempty body, or a query suffix.

Retrieval ignores the query and any successfully parsed body. The ID is the entire path suffix after `/v1/models/`. The server percent-decodes it once, keeps literal `+` and slashes, and replaces invalid UTF-8. IDs match case-sensitively. An empty or unknown ID returns HTTP 404 with an `error` object. It contains `message` (`The model '<id>' does not exist`), `type` (`invalid_request_error`), `param` (`model`), and `code` (`model_not_found`). Other methods return HTTP 405 with `Allow: GET` and `{"detail":"Method Not Allowed"}`. For `HEAD`, the reader sends only the headers. [ID decoding](../apps/server/prefix_controls.cpp#L258)<!-- prefix: "std::string " --> · [Lookup and errors](../apps/server/prefix_controls.cpp#L281)<!-- prefix: "void retriev" --> · [Method response](../apps/server/http.cpp#L458)<!-- prefix: "void write_m" -->

## Verification

With the native server on its default loopback port, send both GET requests.

```bash
curl --fail-with-body -i http://127.0.0.1:17384/model_info
curl --fail-with-body -i http://127.0.0.1:17384/get_model_info
```

Require HTTP 200, matching JSON objects, and the values specified above. If the values differ, check the checkpoint path and parser options used at launch.

Send a POST to check the method response.

```bash
curl -i -X POST http://127.0.0.1:17384/model_info
```

Require HTTP 405 and `Allow: GET`. If another route responds, check the server address and the exact request target.

Request the discovery list and retrieve the model by its encoded ID.

```bash
curl --fail-with-body -i http://127.0.0.1:17384/v1/models
curl --fail-with-body -i 'http://127.0.0.1:17384/v1/models/%51wen3.8-27B-FP8?test=1'
curl -i http://127.0.0.1:17384/v1/models/unknown
curl -I http://127.0.0.1:17384/v1/models/Qwen3.8-27B-FP8
```

Require HTTP 200 for the first two requests and compare the list's only card with the bare object, allowing different `created` values. Require HTTP 404 with `model_not_found` for the unknown ID and HTTP 405 with `Allow: GET` and no body for `HEAD`.
