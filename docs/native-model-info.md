# Native model and server information

## Responsibilities and sources of truth

HTTP handles [model discovery](#model-discovery) and [server information](#server-information) outside the completion queue. The [startup writer](../apps/sglang-server.cpp#L243)<!-- prefix: "std::string " --> supplies metadata, and the [scheduler snapshot](../apps/server/batch_service.cpp#L825)<!-- prefix: "BatchRuntime" --> supplies live state under its mutex. [Implementation map](https://github.com/para1lel/sglang.cpp/blob/49a48f98e44223bd619e7b6f43569b45175504c0/docs/native-model-info.md#responsibilities-and-sources-of-truth)

## Runtime path

### Model information

Both model-info aliases bypass the queue and return HTTP 200 with identical startup JSON after decoder construction, regardless of prefix reuse.

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

`GET /v1/models` returns 200 with `object: "list"` and one card in `data`. `GET /v1/models/Qwen3.8-27B-FP8` returns the card. Its `id` and `root` are that model ID. Other fields are `object: "model"`, `owned_by: "sglang"`, `parent: null`, integer Unix seconds in `created`, and `max_model_len: 262144`. See [resident admission](batch-graphs.md#resident-memory-admission) and the [benchmark alias](../benchmarks/streaming/README.md#wire-contract). [Pinned SGLang discovery comparison](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/native-model-info.md#model-discovery)

### Server information

Both `GET /server_info` and deprecated `GET /get_server_info` return startup settings and a live scheduler snapshot. Like [model discovery](#model-discovery), they bypass the completion queue regardless of prefix reuse. [Reader dispatch](../apps/server/prefix_controls.cpp#L692)<!-- prefix: "write_json(c" -->

| Top-level field | JSON type and meaning |
| --- | --- |
| `frontend`, `version`, `supports_inline_system` | `"cpp"`, `null`, and `false`. A system turn must be first. |
| `host`, `port`, `device` | `"127.0.0.1"`, TCP port, and process-visible CUDA ordinal. |
| `dp_size`, `tp_size`, `pp_size` | Integers reported by the batch executor. The single-device executor sets each to `1`. [Topology](../src/runtime/server/single_device_batch_executor.cpp#L77)<!-- prefix: "BatchTopolog" --> |
| `model_path`, `tokenizer_path`, `served_model_name` | Strings. Both paths expose the effective checkpoint directory. The served ID is `"Qwen3.8-27B-FP8"`. |
| `launch_command` | String starting with `sglang-server`. It records explicit options in input order, including repeats. |
| `max_running_requests`, `max_num_batched_tokens` | Lane limit and aggregate target tokens per step. Defaults are 8 and 32,768. |
| `mtp_steps`, `prefix_cache_entries` | Integer [server draft depth](mtp.md#runtime-path), default 4, and the former slot-count field, always 0. |
| `lmcache_enabled`, `lmcache_available` | Booleans for [configured reuse and daemon availability](prefix-cache.md#lmcache-daemon). |
| `gpu_memory_utilization`, `prefix_gqa_pages`, `prefix_gdn_slots` | Number and two integers. Defaults are `0.88`, `0`, and `0`. Zero selects automatic cache sizing. [Pool configuration](../apps/sglang-server.cpp#L298)<!-- prefix: "body += R\u0022(," --> |
| `resident_prefix_cache`, `device_memory_pools` | Reuse toggle and one object per rank. Each object contains process-visible `device`, total arena `budget_bytes`, general `capacity_bytes` and `available_bytes`, and `cache_bytes` and `cache_available_bytes`. [Rank fields](../apps/sglang-server.cpp#L304)<!-- prefix: "for (int ran" --> · [Memory pools](serving/request-memory.md) |
| `enable_multimodal` | Boolean. `true` for [image and video inputs](chat-serving.md#image-and-video-inputs); the vision encoder loads on first use. [Capability field](../apps/sglang-server.cpp#L282)<!-- prefix: "body += R\"(," --> |
| `mtp_online_gqa` | Boolean. True on B200 with positive server draft depth and capacity for at least two lanes. Each scored draft wave checks its lanes. [Metadata check](../apps/sglang-server.cpp#L324)<!-- prefix: "body += back" --> |
| `mtp_int8_head` | Boolean indicating availability of the [INT8 draft head](#int8-draft-head). |
| `certified_int8_verify_head` | Boolean indicating availability of the [certified Tensor Core head](#certified-tensor-core-target-head). |
| `fp8_prefill` | String `"tensor-core"` on B200 or `"exact"` otherwise. This reports backend policy; individual grants still apply shape and resource guards. [Backend selection](../apps/sglang-server.cpp#L334)<!-- prefix: "const bool t" --> |
| `target_gqa` | String `"grouped-split"` on B200, or `"exact"` otherwise. Per-row shape and numerical guards still apply. [Backend selection](../apps/sglang-server.cpp#L337)<!-- prefix: "sglang::json" --> |
| `paired_prefill` | Boolean. True on B200 with at least two lanes and a step budget of at least 2,048. The [whole-tile selector](native-prefill-workspace.md#projection-pairs-for-mtp) determines eligibility and workspace fallback per step. [Metadata check](../apps/sglang-server.cpp#L339)<!-- prefix: "const bool p" --> |
| `mtp_seed_fc` | Null at K0, otherwise the configured [draft seed FC policy](mtp.md#runtime-path). This field is exclusive to the server information routes. [Metadata builder](../apps/sglang-server.cpp#L515)<!-- prefix: "std::string " --> |
| `sampling_defaults` | String `"model"` or `"openai"`, according to the serving configuration. |
| `reasoning_parser`, `tool_call_parser` | JSON `null` when disabled. Enabled values are `"qwen3"` and `"qwen3_coder"`, respectively. |
| `eager`, `trace_batches`, `batch_cuda_graphs` | Booleans for eager execution, tracing, and [TP1 batch graphs](batch-graphs.md). `batch_cuda_graphs` defaults to `true` on TP1 and is `false` with `--eager` or TP2. |
| `startup_time` | Object with one numeric field, `server_e2e`: elapsed monotonic seconds from before argument parsing through listener creation, fixed for the process lifetime. |
| `internal_states` | Array with one object for the native scheduler rank. |

On B200 with Tensor Core prefill and positive depth, this policy object reports `preferred: "tma_tcgen05_bf16_fp32"`, `fallback: "wmma_bf16_fp32"`, `outside_shape_limits: "simt_bf16_fp32"`, and limits of `min_prompt_tokens: 4096`, `min_rows: 64`, and `max_rows: 128`. Other positive-depth configurations report `preferred: "simt_bf16_fp32"`. See [launch eligibility](mtp.md#runtime-path). [Policy attachment](../apps/sglang-server.cpp#L361)<!-- prefix: "return body " -->

| Field in `internal_states[0]` | JSON type and meaning |
| --- | --- |
| `dp_rank`, `world_size` | Integers `0` and `1`. |
| `effective_max_running_requests_per_dp` | Integer lane capacity for this rank, equal to `max_running_requests`. |
| `ready` | Boolean. It becomes `false` on permanent decoder worker failure, using the same flag as `/ready`. |
| `num_running_reqs` | Integer count of admitted user generation items still in scheduler slots. It excludes waiting HTTP work and internal health jobs. |
| `generation_tokens_total` | Cumulative integer count of published user output IDs, including additional MTP IDs. It advances at scheduler publication, independently of socket delivery and prefix reuse. Health generation contributes zero. |

The command records switches and validated numeric or enumerated operands. Every `--checkpoint` value, including superseded values, becomes `'<redacted>'`. Public path fields identify the selected checkpoint. Environment, executable paths, and HTTP inputs supply no command text. [Argument recording](../apps/sglang-server.cpp#L138)<!-- prefix: "Arguments pa" --> · [Startup interval](../apps/sglang-server.cpp#L448)<!-- prefix: "const auto s" --> · [Listener timestamp](../apps/sglang-server.cpp#L470)<!-- prefix: "const std::c" -->

Native omits upstream Python settings, `kv_events`, `last_gen_throughput`, and `memory_usage`. [Pinned Python routes](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L888)<!-- prefix: "@app.get(\"/g" --> · [Resolved fields](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/server_utils.py#L206)<!-- prefix: "def build_se" --> · [Scheduler measurements](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler_components/internal_state.py#L75)<!-- prefix: "state[\"last_" -->

### INT8 draft head

The B200 INT8 head scores unconstrained draft rows. K0 skips drafting and constrained draft rows use BF16. Invalid INT8 scores retry in BF16, with lowest ID winning ties. A lane stops drafting if no finite winner remains. See [draft dispatch](mtp.md#runtime-path) and [INT8 layouts, eligibility and fallback](https://github.com/para1lel/sglang.cpp/blob/49a48f98e44223bd619e7b6f43569b45175504c0/docs/native-model-info.md#int8-draft-head).

### Grouped draft projections

Complete draft depths can share normalization and projection. Sparse depths and clean declines use ordinary four-lane packets. See the [wide-round gate](mtp.md#target-round), [TP2 repair](batch-graphs.md#eight-row-tp2-draft-repair) and [projection and repair implementation](https://github.com/para1lel/sglang.cpp/blob/49a48f98e44223bd619e7b6f43569b45175504c0/docs/native-model-info.md#grouped-draft-projections).

### Certified Tensor Core target head

The B200 certified head uses INT8 bounds and BF16 refinement to preserve target IDs, BF16 scores and lowest-ID ties. It supports eager unscored greedy MTP singletons and eligible verification packets, including M20 score retention and eager unscored M40. A complete TP2 group of sixteen eligible K4 lanes shares one M80 projection and certifies each half with the M40 kernels. K0, sampling, constraints, raw logits and scored singletons use BF16. Two-row graphs retain their graph head. See [target dispatch](mtp.md#target-round) and [eligibility, memory, fallback and exactness checks](https://github.com/para1lel/sglang.cpp/blob/49a48f98e44223bd619e7b6f43569b45175504c0/docs/native-model-info.md#certified-tensor-core-target-head).

### Draft attention

Visibility includes the current token. KV-only repair skips attention. Eligible B200 scored draft rows use split online attention from 2,048 through 262,144 visible tokens. A geometry mismatch rejects the batch even after partial submission. Invalid combines replay affected heads in FP64. See [grouped draft reads](singleton-gqa-decode.md#grouped-long-draft-reads) and [scalar thresholds, packet eligibility and scratch layout](https://github.com/para1lel/sglang.cpp/blob/49a48f98e44223bd619e7b6f43569b45175504c0/docs/native-model-info.md#draft-attention).

### Migrate old launch commands

| Earlier launch setting | Current launch |
| --- | --- |
| `--fp8-prefill exact`, `tensor-core`, `tensor-core-mtp`, or `tensor-core-mtp-target` | Remove the option from both executables. Hardware and tile eligibility select the prefill kernels. |
| `--mtp-int8-head` | Remove the option from `sglang-server`. The server reports INT8 draft head availability. |
| `SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA`, `SGLANG_EXPERIMENTAL_BULK_PREFILL`, `SGLANG_EXPERIMENTAL_VERIFY_SHARED_INT8_VOCAB` | Unset these variables. Target split attention, paired projections, and the certified vocabulary head use automatic dispatch. |
| `SGLANG_PREFILL_ATTN`, `SGLANG_EXPERIMENTAL_TMEM_PREFILL`, `SGLANG_SERIAL_PREFILL_ROWS` | Unset these variables. Prefill selects tiles and kernels from the request and device. |
| `--mtp-online-gqa` | Remove the retired switch. Eligible B200 draft waves select online attention automatically. |

`--mtp-steps 0` still selects target-only generation. `--eager` still disables
server graph replay. Record the effective configuration from `/server_info`.
Reproduce archived timings with the measured revision, command, and environment.

## Constraints and failure behavior

### Model-info targets

Both routes accept empty GETs, ignore the query, and follow [shared redirects](http-server.md#request-targets-and-redirects). A GET with a body or an unknown path returns 404. Unsupported methods return 405 with `Allow: GET` and `{"detail":"Method Not Allowed"}`; HEAD omits the body. The reader rejects malformed HTTP first. Use [readiness](http-server.md#health-and-readiness) for the decoder and [tokenization](native-tokenization.md) for tokenizer access. [Method response](../apps/server/http.cpp#L656)<!-- prefix: "void write_g" --> · [Request line](../apps/server/http.cpp#L128)<!-- prefix: "HttpRequest " -->

### Discovery targets and errors

The list ignores the query and returns 404 on a non-GET or nonempty body. Retrieval ignores query and framed body. The case-sensitive ID is the decoded suffix, including any trailing slash. An interior line feed prevents matching. Empty or unknown IDs return 404 with `error.message: "The model '<id>' does not exist"`, `type: "invalid_request_error"`, `param: "model"`, and `code: "model_not_found"`. Unsupported retrieval methods follow the [model-info 405 rule](#model-info-targets). [ID decoding](../apps/server/http.cpp#L353)<!-- prefix: "std::string " --> · [Lookup and errors](../apps/server/prefix_controls.cpp#L467)<!-- prefix: "void retriev" --> · [Method response](../apps/server/http.cpp#L585)<!-- prefix: "void write_m" -->

### Server-info targets

Both aliases ignore queries and framed GET bodies, return 200 even with `ready: false`, and redirect trailing slashes to the corresponding alias. Unknown paths return 404. Apply the [same 405/HEAD rule](#model-info-targets). Invalid framing fails before the handler. [Query split](../apps/server/http.cpp#L143)<!-- prefix: "const auto s" --> · [Method reply](../apps/server/http.cpp#L585)<!-- prefix: "void write_m" --> · [Worker flag](../apps/server/batch_service.cpp#L752)<!-- prefix: "bool BatchSe" -->

## Verification

### Model information and discovery

Run against the native server on the default loopback port.

```bash
base=http://127.0.0.1:17384
curl --fail-with-body -i "$base/model_info"
curl --fail-with-body -i "$base/get_model_info"
curl -i -X POST "$base/model_info"
curl --fail-with-body -i "$base/v1/models"
curl --fail-with-body -i "$base/v1/models/%51wen3.8-27B-FP8?test=1"
curl -i "$base/v1/models/unknown"
curl -I "$base/v1/models/Qwen3.8-27B-FP8"
```

Require the two model-info GETs to return identical HTTP 200 objects and POST to return HTTP 405 with `Allow: GET`. Compare the discovery list's only card with the encoded retrieval result, allowing different `created` values. Check the unknown ID returns HTTP 404 with `model_not_found` and HEAD returns HTTP 405 with `Allow: GET` and no body.

### Server information at idle and under load

Start a fresh release server on a reserved B200 with no other clients. The duplicate checkpoint and launch canaries exercise redaction and effective option resolution.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" INFO_TEST=env-canary \
    host ./build/bin/sglang-server --checkpoint /superseded-canary \
    --checkpoint /data0/shared/Qwen3.8-27B-FP8 --device 0 --port 17384 \
    --max-active 2 --max-num-batched-tokens 4096 --mtp-k 0 --prefix-cache-entries 0 \
    --batch-cuda-graphs --eager --sampling-defaults openai
```

Run the [copyable idle and load probes](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/docs/native-model-info.md#server-information-at-idle-and-under-load) with Bash, `curl`, and `jq`. At idle require alias equality, effective options and paths, exactly two redacted checkpoint flags, no canaries, positive startup time, and zero running requests and tokens. Verify both aliases ignore a GET body, query, and auth header. Require the specified 405/HEAD, redirect, and 404 behavior. `/health_generate` must leave counters unchanged.

Run both concurrent 96-token requests. Require overlap and a nondecreasing token count across aliases, with a sample strictly between 0 and 192. Final state is ready, idle, and exactly 192 tokens. If overlap goes unobserved, use longer requests and adjust the total. After accounting changes, repeat with prefix reuse and MTP output, omitting `--eager` for reuse.

[Full verification appendix](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/docs/native-model-info.md#verification)
