# Native model and server information

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP controls | Dispatch model-info, discovery, and server-info routes. | [Control route](../apps/server/prefix_controls.cpp#L506)<!-- prefix: "if (!model_i" --> · [List dispatch](../apps/server/prefix_controls.cpp#L531)<!-- prefix: "if (path == " --> · [Retrieval dispatch](../apps/server/prefix_controls.cpp#L518)<!-- prefix: "if (model_re" --> · [Route](../apps/server/prefix_controls.cpp#L690)<!-- prefix: "if (server_i" --> · [Shared card writer](../apps/server/prefix_controls.cpp#L453)<!-- prefix: "std::string " --> |
| Startup profile | Supply model identity, response schema, topology, and redacted launch options. | [Metadata writer](../apps/sglang-server.cpp#L219)<!-- prefix: "std::string " --> · [Qwen serving profile](../apps/server/qwen/qwen_serving_profile.cpp#L134)<!-- prefix: "const Servin" --> · [Checkpoint identity](../include/sglang/runtime/model/qwen_text_model.h#L8)<!-- prefix: "static const" --> · [Callback](../apps/sglang-server.cpp#L435)<!-- prefix: "return serve" --> · [Response fields](../apps/sglang-server.cpp#L249)<!-- prefix: "std::string " --> · [Command sanitization](../apps/sglang-server.cpp#L103)<!-- prefix: "void append_" --> · [Startup composition](../apps/sglang-server.cpp#L355)<!-- prefix: "auto executo" --> |
| Scheduler | Snapshot user jobs, published IDs, and readiness under the mutex. | [Runtime snapshot](../apps/server/batch_service.cpp#L788)<!-- prefix: "BatchRuntime" --> · [Token accounting](../apps/server/batch_service.cpp#L279)<!-- prefix: "counters.gen" --> |

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
| `dp_size`, `tp_size`, `pp_size` | Integers reported by the batch executor. The single-device executor sets each to `1`. [Topology](../src/runtime/server/single_device_batch_executor.cpp#L74)<!-- prefix: "BatchTopolog" --> |
| `model_path`, `tokenizer_path`, `served_model_name` | Strings. Both paths expose the effective checkpoint directory. The served ID is `"Qwen3.8-27B-FP8"`. |
| `launch_command` | String starting with `sglang-server`. It records explicit options in input order, including repeats. |
| `max_running_requests`, `max_num_batched_tokens` | Lane limit and aggregate target tokens per step. Defaults are 8 and 32,768. |
| `mtp_steps`, `prefix_cache_entries` | Integer [server draft depth](mtp.md#runtime-path), default 4, and the former slot-count field, always 0. |
| `lmcache_enabled`, `lmcache_available` | Booleans for [configured reuse and daemon availability](prefix-cache.md#lmcache-daemon). |
| `enable_multimodal` | Boolean. `true` for [image and video inputs](chat-serving.md#image-and-video-inputs); the vision encoder loads on first use. [Capability field](../apps/sglang-server.cpp#L258)<!-- prefix: "body += R\u0022(," --> |
| `mtp_online_gqa` | Boolean. True on B200 with positive server draft depth and capacity for at least two lanes. Each scored draft wave checks its lanes. [Metadata check](../apps/sglang-server.cpp#L280)<!-- prefix: "body += back" --> |
| `mtp_int8_head` | Boolean indicating availability of the [INT8 draft head](#int8-draft-head). |
| `certified_int8_verify_head` | Boolean indicating availability of the [certified Tensor Core head](#certified-tensor-core-target-head). |
| `fp8_prefill` | String `"tensor-core"` on B200 or `"exact"` otherwise. This reports backend policy; individual grants still apply shape and resource guards. [Backend selection](../apps/sglang-server.cpp#L290)<!-- prefix: "const bool t" --> |
| `target_gqa` | String `"grouped-split"` on B200, or `"exact"` otherwise. Per-row shape and numerical guards still apply. [Backend selection](../apps/sglang-server.cpp#L293)<!-- prefix: "sglang::json" --> |
| `paired_prefill` | Boolean. True on B200 with at least two lanes and a step budget of at least 2,048. The [whole-tile selector](native-prefill-workspace.md#projection-pairs-for-mtp) determines eligibility and workspace fallback per step. [Metadata check](../apps/sglang-server.cpp#L295)<!-- prefix: "const bool p" --> |
| `mtp_seed_fc` | Null at K0, otherwise the configured [draft seed FC policy](mtp.md#runtime-path). This field is exclusive to the server information routes. [Metadata builder](../apps/sglang-server.cpp#L461)<!-- prefix: "std::string " --> |
| `sampling_defaults` | String `"model"` or `"openai"`, according to the serving configuration. |
| `reasoning_parser`, `tool_call_parser` | JSON `null` when disabled. Enabled values are `"qwen3"` and `"qwen3_coder"`, respectively. |
| `eager`, `trace_batches`, `batch_cuda_graphs` | Booleans for eager execution, tracing, and [TP1 batch graphs](batch-graphs.md). `batch_cuda_graphs` defaults to `true` on TP1 and is `false` with `--eager` or TP2. |
| `startup_time` | Object with one numeric field, `server_e2e`: elapsed monotonic seconds from before argument parsing through listener creation, fixed for the process lifetime. |
| `internal_states` | Array with one object for the native scheduler rank. |

On B200 with Tensor Core prefill and positive depth, this policy object reports `preferred: "tma_tcgen05_bf16_fp32"`, `fallback: "wmma_bf16_fp32"`, `outside_shape_limits: "simt_bf16_fp32"`, and limits of `min_prompt_tokens: 4096`, `min_rows: 64`, and `max_rows: 128`. Other positive-depth configurations report `preferred: "simt_bf16_fp32"`. See [launch eligibility](mtp.md#runtime-path). [Policy attachment](../apps/sglang-server.cpp#L317)<!-- prefix: "return body " -->

| Field in `internal_states[0]` | JSON type and meaning |
| --- | --- |
| `dp_rank`, `world_size` | Integers `0` and `1`. |
| `effective_max_running_requests_per_dp` | Integer lane capacity for this rank, equal to `max_running_requests`. |
| `ready` | Boolean. It becomes `false` on permanent decoder worker failure, using the same flag as `/ready`. |
| `num_running_reqs` | Integer count of admitted user generation items still in scheduler slots. It excludes waiting HTTP work and internal health jobs. |
| `generation_tokens_total` | Cumulative integer count of published user output IDs, including additional MTP IDs. It advances at scheduler publication, independently of socket delivery and prefix reuse. Health generation contributes zero. |

The command records switches and validated numeric or enumerated operands. Every `--checkpoint` value, including superseded values, becomes `'<redacted>'`. Public path fields identify the selected checkpoint. Environment, executable paths, and HTTP inputs supply no command text. [Argument recording](../apps/sglang-server.cpp#L122)<!-- prefix: "Arguments pa" --> · [Startup interval](../apps/sglang-server.cpp#L395)<!-- prefix: "const auto s" --> · [Listener timestamp](../apps/sglang-server.cpp#L416)<!-- prefix: "const std::c" -->

Native omits upstream Python settings, `kv_events`, `last_gen_throughput`, and `memory_usage`. [Pinned Python routes](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L888)<!-- prefix: "@app.get(\"/g" --> · [Resolved fields](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/server_utils.py#L206)<!-- prefix: "def build_se" --> · [Scheduler measurements](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler_components/internal_state.py#L75)<!-- prefix: "state[\"last_" -->

### INT8 draft head

B200 prepares 1,272,391,680 device bytes of INT8 weights for unconstrained drafting. Nonfinite weights fail startup with the token ID. Row maxima scale signed INT32 dots through FP64 to BF16. Finite zero rows score zero. Lowest ID wins ties. Invalid INT8 results retry in BF16; no finite winner ends drafting for the lane. Constraints use BF16 and K0 skips draft scoring.

Four lanes can share scoring with 16-byte-aligned activations and weights. Callback-free K4 waves of 8/16/32 can use TMA and verifier scratch, with wide activation reserve at TP2. Rank zero can fuse the full TP2 16-lane depth with mapped argmax. TP1, partial depths, and clean setup or mapping declines keep ordinary scoring. [Head layouts and scratch sizes](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/native-model-info.md#int8-draft-head) [Weight preparation](../src/decoder_mtp.cpp#L110)<!-- prefix: "if (capabili" --> · [Quantization](../src/text_decode_bandwidth.cu#L171)<!-- prefix: "__device__ v" --> · [Retry](../src/decoder_mtp.cpp#L804)<!-- prefix: "bool MtpDraf" --> · [Dispatch](../src/decoder_batch.cpp#L631)<!-- prefix: "const bool s" --> · [Setup cache](../src/decoder_mtp.cpp#L549)<!-- prefix: "bool MtpDraf" --> · [Eligibility](../src/decoder_mtp.cpp#L626)<!-- prefix: "if (count ==" --> · [Fused launch](../src/text/draft_depth.cu#L294)<!-- prefix: "cudaError_t " -->

### Grouped draft projections

Full draft depths pack 8 rows for an 8-lane wave or 16 for wider waves for normalization, BF16 FC, FP8 and activation. Sparse depths, attention, and clean declines use four-lane packets. Ready TP1 8-lane or callback-free 16-lane groups can commit together with shared repair FC/KV and in-range scalar rotary. Uncaptured B200 singletons and pairs can join BF16 FC and FP8 K/V with matching weights. Capture, unsupported shapes, and clean setup declines use ordinary projection. CUDA errors propagate with terminal precedence. See the [wide-round gate](mtp.md#target-round) and [TP2 repair rules](batch-graphs.md#eight-row-tp2-draft-repair). [Full packets](../src/decoder_mtp.cpp#L599)<!-- prefix: "const int pr" --> · [FC dispatch](../src/text/draft_projection.cpp#L53)<!-- prefix: "if (const au" --> · [Setup status](../src/text/draft_depth.cu#L208)<!-- prefix: "std::optiona" --> · [Setup](../src/text/fp8_projection.cu#L947)<!-- prefix: "if (attribut" --> · [Single row](../src/decoder_mtp.cpp#L396)<!-- prefix: "const bool s" --> · [Two rows](../src/decoder_mtp.cpp#L488)<!-- prefix: "const bool s" -->

### Certified Tensor Core target head

B200 `sm_100a` reuses INT8 weights and refines score intervals with original BF16 arithmetic. Eligible eager formats are unscored greedy MTP singletons (1–5 rows) and 2–20-row verification packets, including BF16 score retention at M20. M20 uses 9.7 MiB of staging. K0, sampling, constraints, raw logits, and scored singletons use BF16. Two-row graphs retain their graph head. Align branch, residual, and gain to 16 bytes and reserve M40 scratch to combine two full M20 packets in eligible eager unscored K4. Final normalization yields matching BF16 rows.

Weights add 3,973,120 bytes. Reserve M5/M20 and eligible TP M10 scratch, plus one lazy M40 slab. Clean weight setup decline keeps the ordinary head. M40 reserve, setup or map rejection, or OOM selects M20. Rejected maps retain the slab. Unexpected CUDA errors propagate.

Inclusive bounds preserve lowest-ID ties. Wide or nonfinite intervals can replay exactly. M10 tightens bounds using one exact tile and replays all tokens outside 1–16,384 candidates. Require exact IDs and BF16 scores at M1–5/M10/M20/M40 across ambiguous, scored, raw, sampled, constrained, graph, decline, and commit cases. [Certification assumptions, bounds, and scratch sizes](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/native-model-info.md#certified-tensor-core-target-head) [Dispatch guard](../src/decoder_verify.cpp#L679)<!-- prefix: "if (round.co" --> · [Score selection](../src/decoder_verify.cpp#L584)<!-- prefix: "if (scored_r" --> · [Graph path](../src/decoder_verify.cpp#L881)<!-- prefix: "if (graph &&" --> · [M40 guard](../src/decoder_verify.cpp#L947)<!-- prefix: "if (vocabula" --> · [Weight setup](../src/text/tc_vocab.cpp#L168)<!-- prefix: "std::shared_" --> · [Scratch setup](../src/text/tc_vocab.cpp#L196)<!-- prefix: "std::shared_" --> · [Batch allocation](../src/runtime/execution/decoder_batch_state.cpp#L71)<!-- prefix: "storage-\u003etc_" --> · [Lazy slab](../src/text/tc_vocab.cpp#L227)<!-- prefix: "bool prepare" --> · [Packed bound](../src/text/tc_vocab.cu#L299)<!-- prefix: "__device__ i" --> · [Full replay guard](../src/text/tc_vocab.cu#L411)<!-- prefix: "__device__ i" -->

### Draft attention

Visibility includes the current token. Scalar draft uses direct attention below eight tokens or without scratch, tiled MXFP8 at 8–256, ordered value replay after parallel scores at 257–5,173, and 4,096-position chunks with FP32 carry above that. KV-only repair skips attention.

B200 packets of 1–4 scored rows can use online split at 2,048–262,144 tokens with 24 query heads, four KV heads, 256 channels, covered pages, and aligned buffers. Rows beyond packet capacity can still select online and the long batch covers remaining rows with scratch. Geometry mismatch rejects the batch even after partial submission. The 496,640-byte logits arena aliases ordered carry and scores (419,328 bytes) or split partials (up to 399,360). Invalid combines replay affected heads in FP64. Reduction can change proposals. Compare K4 and K0 with matched target dispatch. [16-row grouping](singleton-gqa-decode.md#grouped-long-draft-reads) [Lane selection](../src/text/gqa/draft_attention.cu#L277)<!-- prefix: "cudaError_t " --> · [Alignment](../src/text/gqa/decode_layout.cu#L26)<!-- prefix: "SGLANG_GQA_H" --> · [Head recovery](../src/text/gqa/target_split.cu#L78)<!-- prefix: "if (!gqa_dec" --> · [Scratch bound](../include/sglang/text/gqa_decode_split.cuh#L28)<!-- prefix: "static_asser" -->

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

Both routes accept empty GETs, ignore the query, and follow [shared redirects](http-server.md#request-targets-and-redirects). A GET with a body or an unknown path returns 404. Unsupported methods return 405 with `Allow: GET` and `{"detail":"Method Not Allowed"}`; HEAD omits the body. The reader rejects malformed HTTP first. Use [readiness](http-server.md#health-and-readiness) for the decoder and [tokenization](native-tokenization.md) for tokenizer access. [Method response](../apps/server/http.cpp#L633)<!-- prefix: "void write_g" --> · [Request line](../apps/server/http.cpp#L128)<!-- prefix: "HttpRequest " -->

### Discovery targets and errors

The list ignores the query and returns 404 on a non-GET or nonempty body. Retrieval ignores query and framed body. The case-sensitive ID is the decoded suffix, including any trailing slash. An interior line feed prevents matching. Empty or unknown IDs return 404 with `error.message: "The model '<id>' does not exist"`, `type: "invalid_request_error"`, `param: "model"`, and `code: "model_not_found"`. Unsupported retrieval methods follow the [model-info 405 rule](#model-info-targets). [ID decoding](../apps/server/http.cpp#L353)<!-- prefix: "std::string " --> · [Lookup and errors](../apps/server/prefix_controls.cpp#L467)<!-- prefix: "void retriev" --> · [Method response](../apps/server/http.cpp#L562)<!-- prefix: "void write_m" -->

### Server-info targets

Both aliases ignore queries and framed GET bodies, return 200 even with `ready: false`, and redirect trailing slashes to the corresponding alias. Unknown paths return 404. Apply the [same 405/HEAD rule](#model-info-targets). Invalid framing fails before the handler. [Query split](../apps/server/http.cpp#L143)<!-- prefix: "const auto s" --> · [Method reply](../apps/server/http.cpp#L562)<!-- prefix: "void write_m" --> · [Worker flag](../apps/server/batch_service.cpp#L719)<!-- prefix: "bool BatchSe" -->

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
