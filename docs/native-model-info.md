# Native model and server information

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP control | Match `GET /model_info` and its deprecated alias `GET /get_model_info`. | [Control route](../apps/server/prefix_controls.cpp#L534)<!-- prefix: "if (!model_i" --> |
| Model discovery | List or retrieve the served model card. | [List dispatch](../apps/server/prefix_controls.cpp#L559)<!-- prefix: "if (path == " --> · [Retrieval dispatch](../apps/server/prefix_controls.cpp#L546)<!-- prefix: "if (model_re" --> · [Shared card writer](../apps/server/prefix_controls.cpp#L481)<!-- prefix: "std::string " --> |
| Server | Report the selected checkpoint and configured serving properties. | [Metadata writer](../apps/sglang-server.cpp#L189)<!-- prefix: "std::string " --> |
| Checkpoint | Supply the supported model type and architecture. | [Checkpoint identity](../include/sglang/checkpoint.h#L47)<!-- prefix: "inline const" --> |
| Server information | Match both server-info paths on the HTTP reader and serialize one snapshot per GET. | [Route](../apps/server/prefix_controls.cpp#L714)<!-- prefix: "if (server_i" --> · [Callback](../apps/sglang-server.cpp#L327)<!-- prefix: "return serve" --> · [Response fields](../apps/sglang-server.cpp#L211)<!-- prefix: "std::string " --> |
| Launcher | Record requested options and pass effective settings to the batch service. | [Command sanitization](../apps/sglang-server.cpp#L67)<!-- prefix: "void append_" --> · [Scheduler construction](../apps/sglang-server.cpp#L298)<!-- prefix: "BatchService" --> |
| Batch scheduler | Snapshot admitted user jobs, published output tokens, and worker readiness under the scheduler mutex. | [Runtime snapshot](../apps/server/batch_service.cpp#L834)<!-- prefix: "BatchRuntime" --> · [Token accounting](../apps/server/batch_service.cpp#L346)<!-- prefix: "counters.gen" --> |

## Runtime path

### Model information

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

### Server information

`GET /server_info` and its alias `GET /get_server_info` return the same JSON schema. Upstream deprecates the alias. The reader calls the batch service directly, even with all generation lanes occupied or prefix reuse disabled. Each call returns the startup configuration and a fresh scheduler snapshot. [Reader dispatch](../apps/server/prefix_controls.cpp#L716)<!-- prefix: "write_json(c" -->

| Top-level field | JSON type and meaning |
| --- | --- |
| `frontend`, `version`, `supports_inline_system` | String `"cpp"`, JSON `null`, and boolean `false`. The native chat template requires a system turn to be first. |
| `host`, `port`, `device` | Loopback address `"127.0.0.1"`, integer TCP port, and integer CUDA ordinal among devices visible to the process. |
| `dp_size`, `tp_size`, `pp_size` | Integers, each `1`. The server runs one data, tensor, and pipeline parallel rank. |
| `model_path`, `tokenizer_path`, `served_model_name` | Strings. Both paths expose the effective checkpoint directory. The served ID is `"Qwen3.8-27B-FP8"`. |
| `launch_command` | String starting with `sglang-server`. It records explicit options in input order, including repeats. |
| `max_running_requests`, `max_num_batched_tokens` | Integer lane limit from `--max-active` and aggregate target-token budget per scheduler step. Defaults are 32 and 8,192. |
| `mtp_steps`, `prefix_cache_entries` | Integer [server draft depth](mtp.md#runtime-path) and prefix entry capacity. Zero disables the corresponding feature. |
| `fp8_prefill`, `sampling_defaults` | Strings selecting the prefill policy (`"exact"` or `"tensor-core"`) and sampling defaults (`"model"` or `"openai"`). |
| `reasoning_parser`, `tool_call_parser` | JSON `null` when disabled. Enabled values are `"qwen3"` and `"qwen3_coder"`, respectively. |
| `eager`, `trace_batches`, `batch_cuda_graphs` | Booleans for eager execution, batch tracing, and the effective [batch graph policy](batch-graphs.md). `batch_cuda_graphs` defaults to `true`. It is `false` with `--eager`, including when the command also includes `--batch-cuda-graphs`. |
| `startup_time` | Object with one numeric field, `server_e2e`. It gives elapsed monotonic seconds from before argument parsing through listener creation, fixed for the process lifetime. |
| `internal_states` | Array with one object for the native scheduler rank. |

| Field in `internal_states[0]` | JSON type and meaning |
| --- | --- |
| `dp_rank`, `world_size` | Integers `0` and `1`. |
| `effective_max_running_requests_per_dp` | Integer lane capacity for this rank, equal to `max_running_requests`. |
| `ready` | Boolean. It becomes `false` on permanent decoder worker failure. This uses the same flag as `/ready`. |
| `num_running_reqs` | Integer count of admitted user generation items still in scheduler slots. It excludes waiting HTTP work and internal health jobs. |
| `generation_tokens_total` | Cumulative integer count of published user output IDs, including additional MTP IDs. It advances at scheduler publication, independently of socket delivery and prefix reuse. Health generation contributes zero. |

The command retains switches and validated numeric or enumerated operands. It replaces every `--checkpoint` operand, including superseded paths, with `'<redacted>'`. The two public path fields still identify the selected checkpoint. Environment variables, executable paths, HTTP headers, query values, and request bodies do not supply command text. [Argument recording](../apps/sglang-server.cpp#L86)<!-- prefix: "Arguments pa" --> · [Startup interval](../apps/sglang-server.cpp#L286)<!-- prefix: "const auto s" --> · [Listener timestamp](../apps/sglang-server.cpp#L308)<!-- prefix: "const std::c" -->

Pinned SGLang exposes resolved settings and a launch command at the top level, with startup durations, per-DP state, version, frontend, and inline-system capability. The native schema reports its own settings and counters. It omits Python-specific options, `kv_events`, and upstream `last_gen_throughput` and `memory_usage` measurements. [Pinned Python routes](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L888)<!-- prefix: "@app.get(\"/g" --> · [Resolved fields](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/server_utils.py#L206)<!-- prefix: "def build_se" --> · [Scheduler measurements](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler_components/internal_state.py#L75)<!-- prefix: "state[\"last_" -->

## Constraints and failure behavior

### Model-info targets

Send an empty GET to either model-info path. Both ignore the query and follow the [shared redirect rules](http-server.md#request-targets-and-redirects). A GET with a body or an unknown path returns HTTP 404. Other valid methods on either model-info path return HTTP 405 and `Allow: GET`. The response is `{"detail":"Method Not Allowed"}`. For HEAD, the server sends only the headers. Malformed HTTP still fails in the reader. [Method response](../apps/server/http.cpp#L626)<!-- prefix: "void write_g" --> · [Request line](../apps/server/http.cpp#L128)<!-- prefix: "HttpRequest " -->

Use [`/ready`](http-server.md) to check the decoder worker. Use a [tokenization request](native-tokenization.md#verification) to check tokenizer access.

### Discovery targets and errors

The list route ignores the query. After HTTP parsing, methods other than GET or a nonempty body return HTTP 404.

Retrieval ignores the query and any successfully parsed body. After shared path decoding and route matching, the ID is the entire suffix after `/v1/models/`, including any trailing slashes. An interior line feed prevents the retrieval route from matching. IDs match case-sensitively. An empty or unknown ID returns HTTP 404 with an `error` object. It contains `message` (`The model '<id>' does not exist`), `type` (`invalid_request_error`), `param` (`model`), and `code` (`model_not_found`). Other methods return HTTP 405 with `Allow: GET` and `{"detail":"Method Not Allowed"}`. For `HEAD`, the reader sends only the headers. [ID decoding](../apps/server/http.cpp#L346)<!-- prefix: "std::string " --> · [Lookup and errors](../apps/server/prefix_controls.cpp#L495)<!-- prefix: "void retriev" --> · [Method response](../apps/server/http.cpp#L555)<!-- prefix: "void write_m" -->

### Server-info targets

Both aliases use the shared decoded path. A GET ignores the query and any successfully framed body. It returns HTTP 200 even when `ready` is `false`. A trailing slash redirects to the corresponding alias. An unknown path returns HTTP 404. Other valid methods on either canonical path return HTTP 405 with `Allow: GET` and `{"detail":"Method Not Allowed"}`. HEAD sends only the headers. Invalid HTTP framing fails before the reader calls the handler. [Query split](../apps/server/http.cpp#L143)<!-- prefix: "const auto s" --> · [Method reply](../apps/server/http.cpp#L555)<!-- prefix: "void write_m" --> · [Worker flag](../apps/server/batch_service.cpp#L775)<!-- prefix: "bool BatchSe" -->

## Verification

### Model information and discovery

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

### Server information at idle and under load

Build the server and select a reserved B200 with the [launch procedure](../README.md#serve-token-ids-over-http). Start a fresh process with no other clients.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" INFO_TEST=env-canary \
    host ./build/bin/sglang-server --checkpoint /superseded-canary \
    --checkpoint /data0/shared/Qwen3.8-27B-FP8 --device 0 --port 17384 \
    --max-active 2 --max-num-batched-tokens 4096 --mtp-k 0 --prefix-cache-entries 0 \
    --batch-cuda-graphs --eager --sampling-defaults openai
```

In a second Bash shell with `curl` and `jq`, compare both responses at idle.

```bash
set -euo pipefail
base=http://127.0.0.1:17384
samples=$(mktemp -d)
for route in server_info get_server_info; do
    curl -fsS -X GET -d body-canary -H 'Authorization: Bearer header-canary' \
        "$base/$route?test=query-canary" | jq -S . > "$samples/$route"
    for method in HEAD POST PUT; do
        status=$(curl -sS --ignore-content-length -X "$method" -D "$samples/headers" \
            -o "$samples/body" -w '%{http_code}' "$base/$route?test=1")
        test "$status" = 405
        tr -d '\r' < "$samples/headers" | grep -ix 'Allow: GET'
        if [[ $method == HEAD ]]; then test ! -s "$samples/body"
        else jq -e '. == {"detail":"Method Not Allowed"}' "$samples/body"; fi
    done
done
diff -u "$samples/server_info" "$samples/get_server_info"
jq -e '.port == 17384 and .device == 0 and .max_running_requests == 2 and
    .max_num_batched_tokens == 4096 and .prefix_cache_entries == 0 and .mtp_steps == 0 and
    .sampling_defaults == "openai" and .eager and (.batch_cuda_graphs == false) and
    .model_path == "/data0/shared/Qwen3.8-27B-FP8" and .tokenizer_path == .model_path and
    ([.launch_command | scan("<redacted>")] | length) == 2 and
    (.launch_command | contains("/data0/") | not) and (tostring | contains("canary") | not) and
    .startup_time.server_e2e > 0 and (.internal_states | length) == 1 and
    (.internal_states[0] | .ready and .effective_max_running_requests_per_dp == 2 and
    .num_running_reqs == 0 and .generation_tokens_total == 0)' "$samples/server_info"
for route in server_info get_server_info; do
    status=$(curl -sS -X PUT -D "$samples/headers" -o "$samples/body" \
        -w '%{http_code}' "$base/$route//?test=1")
    test "$status" = 307
    test ! -s "$samples/body"
    tr -d '\r' < "$samples/headers" | grep -Fxi "Location: $base/$route?test=1"
done
test "$(curl -sS -o /dev/null -w '%{http_code}' "$base/server_info_unknown")" = 404
curl -fsS "$base/health_generate"
curl -fsS "$base/server_info" | jq -S . | diff - "$samples/server_info"
```

Both aliases must return identical objects at idle. The health probe must preserve both counters.

Send two concurrent requests and sample both routes until generation finishes.

```bash
generate() {
    curl -fsS --max-time 180 -H 'Content-Type: application/json' \
        -d '{"input_ids":[220],"sampling_params":{"max_new_tokens":96,"temperature":0,"ignore_eos":true}}' \
        "$base/generate"
}
generate > "$samples/first" & first=$!
generate > "$samples/second" & second=$!
while kill -0 "$first" 2>/dev/null || kill -0 "$second" 2>/dev/null; do
    for route in server_info get_server_info; do
        curl -fsS --max-time 5 "$base/$route" | jq -c '.internal_states[0]' >> "$samples/busy"
    done
done
wait "$first"
wait "$second"
jq -es 'all(.[]; (.output_ids | length) == 96 and .meta_info.completion_tokens == 96)' \
    "$samples/first" "$samples/second"
jq -es 'any(.[]; .num_running_reqs == 2) and
    any(.[]; .generation_tokens_total > 0 and .generation_tokens_total < 192) and
    ([.[].generation_tokens_total] | . == sort)' "$samples/busy"
curl -fsS "$base/get_server_info" | jq -e '.internal_states[0] |
    .ready and .num_running_reqs == 0 and .generation_tokens_total == 192'
```

Both lanes must be visible together, and the final token count must equal the two response lengths. If polling misses the overlap, restart with longer requests and adjust the expected counts. After an accounting change, repeat with prefix reuse enabled and with MTP output. Omit `--eager` when enabling prefix reuse.
