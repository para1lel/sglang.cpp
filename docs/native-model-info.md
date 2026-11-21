# Native model information

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP control | Match `GET /model_info` and its deprecated alias `GET /get_model_info`. | [Control route](../apps/server/prefix_controls.cpp#L239)<!-- prefix: "void Connect" --> |
| Server | Report the selected checkpoint and configured serving properties. | [Metadata writer](../apps/sglang-server.cpp#L146)<!-- prefix: "std::string " --> |
| Checkpoint | Supply the supported model type and architecture. | [Checkpoint identity](../include/sglang/checkpoint.h#L47)<!-- prefix: "inline const" --> |

## Runtime path

The server prepares the response at startup after decoder construction. Both routes return the same HTTP 200 JSON object. They bypass the completion queue with prefix reuse off or on.

| Field | Native value |
| --- | --- |
| `model_path`, `tokenizer_path` | The directory supplied to `--checkpoint`. The [tokenizer](native-tokenization.md) loads `tokenizer.json` from this directory on demand. |
| `served_model_name` | `Qwen3.8-27B-FP8`, also the ID in [`/v1/models`](http-server.md#model-discovery). |
| `is_generation`, `has_image_understanding`, `has_audio_understanding` | JSON booleans `true`, `true`, and `false`, respectively. See [chat image input](chat-serving.md). |
| `preferred_sampling_params` | JSON `null`. [Chat sampling defaults](chat-serving.md#sampling-defaults-and-output-budget) still apply. |
| `weight_version`, `load_format` | Strings `"default"` and `"safetensors"`. |
| `reasoning_parser`, `tool_call_parser` | JSON `null` by default. When enabled, the fields show `"qwen3"` or `"qwen3_coder"`, respectively. |
| `disaggregation_mode` | String `"null"`. The native server runs prefill and decode together. |
| `model_type`, `architectures` | `"qwen3_5"` and `["Qwen3_5ForConditionalGeneration"]`. |

## Constraints and failure behavior

Send an empty GET to the exact target. A query, trailing slash, or other unmatched target returns HTTP 404. An exact GET with a body also returns HTTP 404. An exact target with a different valid method returns HTTP 405 and `Allow: GET`. The response is `{"detail":"Method Not Allowed"}`. For HEAD, the server sends only the headers. Malformed HTTP still fails in the reader. [Method response](../apps/server/http.cpp#L450)<!-- prefix: "void write_g" --> · [Request line](../apps/server/http.cpp#L116)<!-- prefix: "HttpRequest " -->

Use [`/ready`](http-server.md) to check the decoder worker. Use a [tokenization request](native-tokenization.md#verification) to check tokenizer access.

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
