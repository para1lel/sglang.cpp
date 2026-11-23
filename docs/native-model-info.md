# Native model and server information

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP control | Match `GET /model_info` and its deprecated alias `GET /get_model_info`. | [Control route](../apps/server/prefix_controls.cpp#L508)<!-- prefix: "if (!model_i" --> |
| Model discovery | List or retrieve the served model card. | [List dispatch](../apps/server/prefix_controls.cpp#L533)<!-- prefix: "if (path == " --> · [Retrieval dispatch](../apps/server/prefix_controls.cpp#L520)<!-- prefix: "if (model_re" --> · [Shared card writer](../apps/server/prefix_controls.cpp#L455)<!-- prefix: "std::string " --> |
| Server | Report the selected checkpoint and configured serving properties. | [Metadata writer](../apps/sglang-server.cpp#L173)<!-- prefix: "std::string " --> |
| Checkpoint | Supply the supported model type and architecture. | [Checkpoint identity](../include/sglang/runtime/model/qwen_text_model.h#L8)<!-- prefix: "static const" --> |
| Server information | Match both server-info paths on the HTTP reader and serialize one snapshot per GET. | [Route](../apps/server/prefix_controls.cpp#L692)<!-- prefix: "if (server_i" --> · [Callback](../apps/sglang-server.cpp#L332)<!-- prefix: "return serve" --> · [Response fields](../apps/sglang-server.cpp#L195)<!-- prefix: "std::string " --> |
| Launcher | Record requested options and pass effective settings to the batch service. | [Command sanitization](../apps/sglang-server.cpp#L66)<!-- prefix: "void append_" --> · [Scheduler construction](../apps/sglang-server.cpp#L303)<!-- prefix: "BatchService" --> |
| Batch scheduler | Snapshot admitted user jobs, published output tokens, and worker readiness under the scheduler mutex. | [Runtime snapshot](../apps/server/batch_service.cpp#L827)<!-- prefix: "BatchRuntime" --> · [Token accounting](../apps/server/batch_service.cpp#L338)<!-- prefix: "counters.gen" --> |

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

`GET /server_info` and its alias `GET /get_server_info` return the same JSON schema. Upstream deprecates the alias. The reader calls the batch service directly, even with all generation lanes occupied or prefix reuse disabled. Each call returns the startup configuration and a fresh scheduler snapshot. [Reader dispatch](../apps/server/prefix_controls.cpp#L694)<!-- prefix: "write_json(c" -->

| Top-level field | JSON type and meaning |
| --- | --- |
| `frontend`, `version`, `supports_inline_system` | String `"cpp"`, JSON `null`, and boolean `false`. The native chat template requires a system turn to be first. |
| `host`, `port`, `device` | Loopback address `"127.0.0.1"`, integer TCP port, and integer CUDA ordinal among devices visible to the process. |
| `dp_size`, `tp_size`, `pp_size` | Integers reported by the batch executor. The single-device executor sets each to `1`. [Topology](../src/runtime/server/single_device_batch_executor.cpp#L74)<!-- prefix: "BatchTopolog" --> |
| `model_path`, `tokenizer_path`, `served_model_name` | Strings. Both paths expose the effective checkpoint directory. The served ID is `"Qwen3.8-27B-FP8"`. |
| `launch_command` | String starting with `sglang-server`. It records explicit options in input order, including repeats. |
| `max_running_requests`, `max_num_batched_tokens` | Integer lane limit from `--max-active` and aggregate target-token budget per scheduler step. Defaults are 32 and 8,192. |
| `mtp_steps`, `prefix_cache_entries` | Integer [server draft depth](mtp.md#runtime-path) and prefix entry capacity. Zero disables the corresponding feature. |
| `mtp_online_gqa` | Boolean. True on B200 with positive server draft depth and capacity for at least two lanes. It reports availability. Each scored draft wave checks its lanes. [Metadata check](../apps/sglang-server.cpp#L222)<!-- prefix: "body += back" --> |
| `mtp_int8_head` | Boolean indicating availability of the [INT8 draft head](#int8-draft-head). |
| `certified_int8_verify_head` | Boolean indicating availability of the [certified Tensor Core head](#certified-tensor-core-target-head). |
| `fp8_prefill` | String `"tensor-core"` for B200 with prefix reuse off, or `"exact"` otherwise. Media and label scoring retain exact prefill. [Backend selection](../apps/sglang-server.cpp#L228)<!-- prefix: "const bool t" --> |
| `target_gqa` | String `"grouped-split"` on B200, or `"exact"` otherwise. Per-row shape and numerical guards still apply. [Backend selection](../apps/sglang-server.cpp#L231)<!-- prefix: "sglang::json" --> |
| `paired_prefill` | Boolean indicating availability of [whole-tile MTP projection pairs](native-prefill-workspace.md#projection-pairs-for-mtp). True on B200 with prefix reuse off, positive MTP depth, capacity for at least two lanes, and a step budget of 2,048–8,192. The scheduler checks each step separately. [Partial-grant projection groups](native-prefill-workspace.md#partial-grant-projection-groups) have independent eligibility. [Metadata check](../apps/sglang-server.cpp#L233)<!-- prefix: "const bool p" --> |
| `mtp_seed_fc` | Null at K0, otherwise the configured [draft seed FC policy](mtp.md#runtime-path). This field is exclusive to the server information routes. [Metadata builder](../apps/sglang-server.cpp#L358)<!-- prefix: "std::string " --> |
| `sampling_defaults` | String `"model"` or `"openai"`, according to the serving configuration. |
| `reasoning_parser`, `tool_call_parser` | JSON `null` when disabled. Enabled values are `"qwen3"` and `"qwen3_coder"`, respectively. |
| `eager`, `trace_batches`, `batch_cuda_graphs` | Booleans for eager execution, batch tracing, and the effective [batch graph policy](batch-graphs.md). `batch_cuda_graphs` defaults to `true`. It is `false` with `--eager`, including when the command also includes `--batch-cuda-graphs`. |
| `startup_time` | Object with one numeric field, `server_e2e`. It gives elapsed monotonic seconds from before argument parsing through listener creation, fixed for the process lifetime. |
| `internal_states` | Array with one object for the native scheduler rank. |

For B200 Tensor Core prefill with positive depth and prefix reuse off, `mtp_seed_fc` reports `preferred: "tma_tcgen05_bf16_fp32"`, `fallback: "wmma_bf16_fp32"`, and `outside_shape_limits: "simt_bf16_fp32"`. The limits are `min_prompt_tokens: 4096`, `min_rows: 64`, and `max_rows: 128`. Other positive-depth configurations report `preferred: "simt_bf16_fp32"`. These fields describe startup policy. Actual launches depend on the request, operands, and CUDA preflight. [Policy attachment](../apps/sglang-server.cpp#L257)<!-- prefix: "return body " -->

| Field in `internal_states[0]` | JSON type and meaning |
| --- | --- |
| `dp_rank`, `world_size` | Integers `0` and `1`. |
| `effective_max_running_requests_per_dp` | Integer lane capacity for this rank, equal to `max_running_requests`. |
| `ready` | Boolean. It becomes `false` on permanent decoder worker failure. This uses the same flag as `/ready`. |
| `num_running_reqs` | Integer count of admitted user generation items still in scheduler slots. It excludes waiting HTTP work and internal health jobs. |
| `generation_tokens_total` | Cumulative integer count of published user output IDs, including additional MTP IDs. It advances at scheduler publication, independently of socket delivery and prefix reuse. Health generation contributes zero. |

The command retains switches and validated numeric or enumerated operands. It replaces every `--checkpoint` operand, including superseded paths, with `'<redacted>'`. The two public path fields still identify the selected checkpoint. Environment variables, executable paths, HTTP headers, query values, and request bodies do not supply command text. [Argument recording](../apps/sglang-server.cpp#L85)<!-- prefix: "Arguments pa" --> · [Startup interval](../apps/sglang-server.cpp#L291)<!-- prefix: "const auto s" --> · [Listener timestamp](../apps/sglang-server.cpp#L313)<!-- prefix: "const std::c" -->

Pinned SGLang exposes resolved settings and a launch command at the top level, with startup durations, per-DP state, version, frontend, and inline-system capability. The native schema reports its own settings and counters. It omits Python-specific options, `kv_events`, and upstream `last_gen_throughput` and `memory_usage` measurements. [Pinned Python routes](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L888)<!-- prefix: "@app.get(\"/g" --> · [Resolved fields](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/server_utils.py#L206)<!-- prefix: "def build_se" --> · [Scheduler measurements](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler_components/internal_state.py#L75)<!-- prefix: "state[\"last_" -->

### INT8 draft head

With draft weights loaded on B200, unconstrained draft proposals use signed INT8 scoring. Constrained drafts use BF16. The INT8 copy adds 1,272,391,680 device bytes per model. A nonfinite head weight aborts startup with its token ID. With `--mtp-steps 0`, the head stays inactive. [Weight preparation](../src/decoder_mtp.cpp#L189)<!-- prefix: "if (capabili" -->

Quantization uses the maximum absolute value per weight row and activation. Zero rows give zero logits for finite operands. Rescaling uses FP64 intermediates before rounding to BF16. Ties choose the lowest token ID. An invalid INT8 score triggers BF16 scoring on the same activation. If the retry has no finite winner, that lane stops drafting for the round. The target verifies the available rows and the normal commit path repairs the draft KV. [Quantization](../src/text_decode_bandwidth.cu#L145)<!-- prefix: "__device__ v" --> · [Retry](../src/decoder_mtp.cpp#L747)<!-- prefix: "bool MtpDraf" -->

An INT8 wave of four unconstrained draft lanes uses four vocabulary rows per warp. The warp reuses each weight vector across the four quantized activations and keeps separate signed INT32 sums for each lane and row. It uses the same FP64 rescale and BF16 rounding. One to three lanes use the existing INT8 kernels. The batch entry point requires 16-byte-aligned weights and quantized activation buffers. Each lane writes its own logits before winner selection. [Wave scoring](../src/decoder_mtp.cpp#L421)<!-- prefix: "void MtpDraf" --> · [Four-lane kernel](../src/text_decode_bandwidth.cu#L273)<!-- prefix: "__global__ _" --> · [Batch dispatch](../src/text_decode_bandwidth.cu#L632)<!-- prefix: "cudaError_t " -->

### Certified Tensor Core target head

The certified greedy vocabulary head uses a native `sm_100a` kernel with TMA
and `tcgen05`. It requires loaded draft INT8 weights on B200 and valid tensor
maps for the weights and quantized inputs. The fast path serves eager packets
of four greedy lanes with five rows each. Each lane must use unconstrained
selection and request only the winner. Wide MTP verification scores each
four-lane packet separately. Unsupported tensor maps, unaligned weights, other packet
shapes, and K0 use the BF16 head. CUDA configuration and launch errors propagate.
[Score kernel](../src/text/tc_vocab.cu#L77)<!-- prefix: "__global__ _" --> · [Weight setup](../src/text/tc_vocab.cu#L142)<!-- prefix: "std::shared_" --> · [Scratch setup](../src/text/tc_vocab.cu#L168)<!-- prefix: "std::shared_" --> · [Dispatch guard](../src/decoder_verify.cpp#L480)<!-- prefix: "if (round.co" --> · [Build target](../xmake.lua#L69)<!-- prefix: "target(\"sgla" -->
· [Packet dispatch](../src/decoder_verify.cpp#L608)<!-- prefix: "if (packed_a" -->

The verifier reads the existing draft INT8 weights and adds 3,973,120 model bytes
for statistics and 21,544,480 workspace bytes per decoder batch before allocator rounding.
When the provider is available, batches with capacity for four lanes or more
prepare workspace before request admission.
[Batch allocation](../src/decoder_batch.cpp#L235)<!-- prefix: "storage-\u003etc_" -->

The Tensor Core dot accumulates signed INT8 products exactly in INT32. The interval
includes FP32 subnormal rounding error in the original BF16 head. Refinement repeats
the ordered FMA reduction and BF16 rounding, skips NaN scores and chooses the lowest
token ID on ties. Nonfinite or oversized operands get unbounded intervals. Empty
or overflowing candidate lists trigger a complete original-order scan.

### Draft attention

Visible length includes the current token. Draft forwards use tiled MXFP8 attention
at 8–256 visible tokens. At 257–5,173 tokens with standard scratch, they compute
exact scores in parallel and replay values in position order.
[Draft dispatch](../include/sglang/text/mtp/gqa_attention.cuh#L18)<!-- prefix: "constexpr in" -->

On B200, a scored draft wave of two to four rows automatically selects grouped
online split GQA for eligible lanes. A lane needs a visible length greater than
its score capacity, starting at 5,174 tokens with standard scratch. It also needs
valid KV and page coverage, aligned query and KV buffers, and sufficient aligned
scratch. Each lane keeps its own position, page table, and scratch. One eligible
lane can use split GQA when the other rows in the wave take different routes.
[Scored wave call](../src/decoder_mtp.cpp#L595)<!-- prefix: "if (score) {" --> · [Lane selection](../include/sglang/text/mtp/gqa_attention.cuh#L44)<!-- prefix: "cudaError_t " --> · [Alignment](../include/sglang/text/gqa_decode_split.cuh#L69)<!-- prefix: "__host__ __d" -->

The online route uses the same grouped split operator as target attention.
It can change draft attention values and proposals through a different reduction order.
The target still verifies proposals with its existing route. If the online combine
cannot produce a valid result, the operator replays the affected query head in FP64.
[Split launch](../include/sglang/text/gqa_target_split.cuh#L108)<!-- prefix: "cudaError_t " --> · [Head recovery](../include/sglang/text/gqa_target_split.cuh#L13)<!-- prefix: "if (!gqa_dec" --> · [Target round](mtp.md#runtime-path)

Other long draft calls score 4,096 positions at a time. They carry the FP32
replay maximum, denominator, and weighted sum between chunks. Ordered paths
use four-byte V loads when aligned and assemble the same values otherwise.
Lengths below eight and calls with insufficient score scratch use direct attention.
Up to four unconstrained lanes share projection and vocabulary weight reads.
Single-row waves use the row dispatch. Batched KV-only repairs skip attention.
[Chunked replay](../include/sglang/text/mtp/gqa_long.cuh#L151)<!-- prefix: "cudaError_t " --> · [Single-row wave](../src/decoder_mtp.cpp#L546)<!-- prefix: "if (count ==" --> · [Repair waves](../src/decoder_mtp.cpp#L625)<!-- prefix: "void repair_" -->

Each draft lane reserves 496,640 bytes for vocabulary logits. Ordered long
attention uses 419,328 bytes for scores and carry. Online split partials use at
most 399,360 bytes within the same requirement. Neither path adds request
memory. The lane stream orders attention before vocabulary scoring overwrites
the scratch. [Score storage](../src/decoder_mtp.cpp#L119)<!-- prefix: "ExactGqaWork" --> · [Scratch bound](../include/sglang/text/gqa_decode_split.cuh#L46)<!-- prefix: "static_asser" -->

Compare the current B200 K4 runtime with a fresh K0 control on the same target dispatch. See pinned Git history at `8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3` for the settings and evidence from earlier runtime profiles.

### Migrate old launch commands

| Earlier launch setting | Current launch |
| --- | --- |
| `--fp8-prefill exact`, `tensor-core`, `tensor-core-mtp`, or `tensor-core-mtp-target` | Remove the option from both executables. The runtime selects text prefill from hardware and cache eligibility. |
| `--mtp-int8-head` | Remove the option from `sglang-server`. The server reports INT8 draft head availability. |
| `SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA`, `SGLANG_EXPERIMENTAL_BULK_PREFILL`, `SGLANG_EXPERIMENTAL_VERIFY_SHARED_INT8_VOCAB` | Unset these variables. Target split attention, paired projections, and the certified vocabulary head use automatic dispatch. |
| `SGLANG_PREFILL_ATTN`, `SGLANG_EXPERIMENTAL_TMEM_PREFILL`, `SGLANG_SERIAL_PREFILL_ROWS` | Unset these variables. Prefill selects tiles and kernels from the request and device. |
| `--mtp-online-gqa` | Remove the retired switch. Eligible B200 draft waves select online attention automatically. |

`--mtp-steps 0` still selects target-only generation. `--eager` still disables
server graph replay. Record the effective configuration from `/server_info`.
Reproduce archived timings with the measured revision, command, and environment.

## Constraints and failure behavior

### Model-info targets

Send an empty GET to either model-info path. Both ignore the query and follow the [shared redirect rules](http-server.md#request-targets-and-redirects). A GET with a body or an unknown path returns HTTP 404. Other valid methods on either model-info path return HTTP 405 and `Allow: GET`. The response is `{"detail":"Method Not Allowed"}`. For HEAD, the server sends only the headers. Malformed HTTP still fails in the reader. [Method response](../apps/server/http.cpp#L627)<!-- prefix: "void write_g" --> · [Request line](../apps/server/http.cpp#L128)<!-- prefix: "HttpRequest " -->

Use [`/ready`](http-server.md) to check the decoder worker. Use a [tokenization request](native-tokenization.md#verification) to check tokenizer access.

### Discovery targets and errors

The list route ignores the query. After HTTP parsing, methods other than GET or a nonempty body return HTTP 404.

Retrieval ignores the query and any successfully parsed body. After shared path decoding and route matching, the ID is the entire suffix after `/v1/models/`, including any trailing slashes. An interior line feed prevents the retrieval route from matching. IDs match case-sensitively. An empty or unknown ID returns HTTP 404 with an `error` object. It contains `message` (`The model '<id>' does not exist`), `type` (`invalid_request_error`), `param` (`model`), and `code` (`model_not_found`). Other methods return HTTP 405 with `Allow: GET` and `{"detail":"Method Not Allowed"}`. For `HEAD`, the reader sends only the headers. [ID decoding](../apps/server/http.cpp#L346)<!-- prefix: "std::string " --> · [Lookup and errors](../apps/server/prefix_controls.cpp#L469)<!-- prefix: "void retriev" --> · [Method response](../apps/server/http.cpp#L556)<!-- prefix: "void write_m" -->

### Server-info targets

Both aliases use the shared decoded path. A GET ignores the query and any successfully framed body. It returns HTTP 200 even when `ready` is `false`. A trailing slash redirects to the corresponding alias. An unknown path returns HTTP 404. Other valid methods on either canonical path return HTTP 405 with `Allow: GET` and `{"detail":"Method Not Allowed"}`. HEAD sends only the headers. Invalid HTTP framing fails before the reader calls the handler. [Query split](../apps/server/http.cpp#L143)<!-- prefix: "const auto s" --> · [Method reply](../apps/server/http.cpp#L556)<!-- prefix: "void write_m" --> · [Worker flag](../apps/server/batch_service.cpp#L768)<!-- prefix: "bool BatchSe" -->

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
    has("mtp_seed_fc") and .mtp_seed_fc == null and
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
