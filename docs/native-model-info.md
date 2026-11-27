# Native model and server information

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP control | Match `GET /model_info` and its deprecated alias `GET /get_model_info`. | [Control route](../apps/server/prefix_controls.cpp#L506)<!-- prefix: "if (!model_i" --> |
| Model discovery | List or retrieve the served model card. | [List dispatch](../apps/server/prefix_controls.cpp#L531)<!-- prefix: "if (path == " --> · [Retrieval dispatch](../apps/server/prefix_controls.cpp#L518)<!-- prefix: "if (model_re" --> · [Shared card writer](../apps/server/prefix_controls.cpp#L453)<!-- prefix: "std::string " --> |
| Server | Report checkpoint and serving properties from the startup profile. | [Metadata writer](../apps/sglang-server.cpp#L219)<!-- prefix: "std::string " --> · [Qwen serving profile](../apps/server/qwen/qwen_serving_profile.cpp#L134)<!-- prefix: "const Servin" --> |
| Checkpoint | Supply the supported model type and architecture. | [Checkpoint identity](../include/sglang/runtime/model/qwen_text_model.h#L8)<!-- prefix: "static const" --> |
| Server information | Match both server-info paths on the HTTP reader and serialize one snapshot per GET. | [Route](../apps/server/prefix_controls.cpp#L690)<!-- prefix: "if (server_i" --> · [Callback](../apps/sglang-server.cpp#L435)<!-- prefix: "return serve" --> · [Response fields](../apps/sglang-server.cpp#L249)<!-- prefix: "std::string " --> |
| Launcher | Record options and supply the executor to the scheduler. | [Command sanitization](../apps/sglang-server.cpp#L103)<!-- prefix: "void append_" --> · [Startup composition](../apps/sglang-server.cpp#L355)<!-- prefix: "auto executo" --> |
| Batch scheduler | Snapshot admitted user jobs, published output tokens, and worker readiness under the scheduler mutex. | [Runtime snapshot](../apps/server/batch_service.cpp#L788)<!-- prefix: "BatchRuntime" --> · [Token accounting](../apps/server/batch_service.cpp#L279)<!-- prefix: "counters.gen" --> |

## Runtime path

### Model information

Both routes bypass the queue and return identical startup JSON with HTTP 200 after decoder construction. Prefix reuse is independent.

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

Send an empty `GET /v1/models` to get HTTP 200 with `object: "list"` and one card in `data`. Send `GET /v1/models/Qwen3.8-27B-FP8` for the card directly as an HTTP 200 JSON object.

The native chat and completion ID `Qwen3.8-27B-FP8` fills `id` and `root`. The card sets `object: "model"`, `owned_by: "sglang"`, and `parent: null`. `created` gives integer Unix seconds. `max_model_len` gives the 262,144-token context shared by prompt and output. GPU memory can lower admission capacity. The `benchmark` alias belongs to the [token-ID API](../benchmarks/streaming/README.md#wire-contract).

Pinned SGLang can list loaded LoRA adapters and retrieves its configured base served name. Native discovery exposes the single fixed checkpoint ID. [Pinned model list](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1998)<!-- prefix: "@app.get(\"/v" --> · [Pinned model retrieval](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L2030)<!-- prefix: "@app.get(\"/v" -->

### Server information

`GET /server_info` and upstream-deprecated `GET /get_server_info` return startup configuration and fresh scheduler state in the same schema. Both call the batch service directly, even with occupied lanes or prefix reuse disabled. [Reader dispatch](../apps/server/prefix_controls.cpp#L692)<!-- prefix: "write_json(c" -->

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

For B200 Tensor Core prefill with positive depth, `mtp_seed_fc` reports `preferred: "tma_tcgen05_bf16_fp32"`, `fallback: "wmma_bf16_fp32"`, and `outside_shape_limits: "simt_bf16_fp32"`. The limits are `min_prompt_tokens: 4096`, `min_rows: 64`, and `max_rows: 128`. Other positive-depth configurations report `preferred: "simt_bf16_fp32"`. Launches still depend on the request, operands, and CUDA preflight. [Policy attachment](../apps/sglang-server.cpp#L317)<!-- prefix: "return body " -->

| Field in `internal_states[0]` | JSON type and meaning |
| --- | --- |
| `dp_rank`, `world_size` | Integers `0` and `1`. |
| `effective_max_running_requests_per_dp` | Integer lane capacity for this rank, equal to `max_running_requests`. |
| `ready` | Boolean. It becomes `false` on permanent decoder worker failure, using the same flag as `/ready`. |
| `num_running_reqs` | Integer count of admitted user generation items still in scheduler slots. It excludes waiting HTTP work and internal health jobs. |
| `generation_tokens_total` | Cumulative integer count of published user output IDs, including additional MTP IDs. It advances at scheduler publication, independently of socket delivery and prefix reuse. Health generation contributes zero. |

The command records switches and validated numeric or enumerated operands. Every `--checkpoint` value, including superseded values, becomes `'<redacted>'`. Public path fields identify the selected checkpoint. Environment, executable paths, and HTTP inputs supply no command text. [Argument recording](../apps/sglang-server.cpp#L122)<!-- prefix: "Arguments pa" --> · [Startup interval](../apps/sglang-server.cpp#L395)<!-- prefix: "const auto s" --> · [Listener timestamp](../apps/sglang-server.cpp#L416)<!-- prefix: "const std::c" -->

Pinned SGLang reports resolved settings, launch command, startup durations, per-DP state, version, frontend, and inline-system capability. Native omits Python-specific options, `kv_events`, `last_gen_throughput`, and `memory_usage`. [Pinned Python routes](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L888)<!-- prefix: "@app.get(\"/g" --> · [Resolved fields](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/server_utils.py#L206)<!-- prefix: "def build_se" --> · [Scheduler measurements](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler_components/internal_state.py#L75)<!-- prefix: "state[\"last_" -->

### INT8 draft head

B200 uses INT8 for unconstrained drafts and BF16 otherwise. The copy adds 1,272,391,680 device bytes per model. Nonfinite weights abort startup with the token ID. K0 skips draft scoring. [Weight preparation](../src/decoder_mtp.cpp#L104)<!-- prefix: "if (capabili" -->

INT8 uses weight-row and activation maxima. Zero rows give zero logits for finite operands. FP64 rescaling rounds to BF16. Lowest token ID wins ties. Invalid scores retry in BF16. Without a finite winner, drafting stops. Verification and repair continue. [Quantization](../src/text_decode_bandwidth.cu#L171)<!-- prefix: "__device__ v" --> · [Retry](../src/decoder_mtp.cpp#L798)<!-- prefix: "bool MtpDraf" -->

Four unconstrained lanes share four vocabulary rows per warp. Signed INT32 sums rescale in FP64 and round to BF16. Weights and activations need 16-byte alignment. Selection follows logit writes. Smaller groups use ordinary INT8. [Wave scoring](../src/decoder_mtp.cpp#L289)<!-- prefix: "void MtpDraf" --> · [Four-lane kernel](../src/text_decode_bandwidth.cu#L299)<!-- prefix: "__global__ _" --> · [Batch dispatch](../src/text_decode_bandwidth.cu#L789)<!-- prefix: "cudaError_t " -->

Callback-free B200 K4 waves of 8, 16, or 32 score active rows with TMA and reuse verification scratch. TP2 needs wide activation reserve. Readiness is cached per allocation. Clean declines use four-lane packets. [Dispatch](../src/decoder_batch.cpp#L631)<!-- prefix: "const bool s" --> · [Setup cache](../src/decoder_mtp.cpp#L543)<!-- prefix: "bool MtpDraf" --> · [TMA setup](../src/text/draft_depth.cu#L246)<!-- prefix: "std::optiona" --> · [Scoring](../src/decoder_mtp.cpp#L636)<!-- prefix: "check(projec" -->

B200 TP2 rank 0 fuses scoring and argmax for full 16-lane proposal depths with grouped argmax and mapped winners. Four-stage TMA feeds all 248,320 INT8 dots per lane. FP64 rescaling rounds to BF16. Keys for 128-token tiles use 124,160 existing scratch bytes after the activation maxima. Final selection preserves lowest-ID ties and BF16 score bits. TP1, partial depths, and clean setup or mapping declines use existing scoring. Nonfinite scores trigger BF16 retry. [Eligibility](../src/decoder_mtp.cpp#L620)<!-- prefix: "if (count ==" --> · [Setup](../src/text/draft_depth.cu#L270)<!-- prefix: "if (tensor_p" --> · [Winner scratch](../include/sglang/text/mtp/draft_depth.h#L13)<!-- prefix: "inline const" --> · [Fused launch](../src/text/draft_depth.cu#L294)<!-- prefix: "cudaError_t " -->

### Grouped draft projections

Full proposal depths pack eight rows per eight-lane wave or 16 per wider wave. Packed passes batch both normalizations, BF16 Tensor Core FC, FP8 projections, and activations. Sparse proposals and attention use four-lane packets. [Full packets](../src/decoder_mtp.cpp#L593)<!-- prefix: "const int pr" --> · [FC dispatch](../include/sglang/text/mtp/draft_projection.h#L52)<!-- prefix: "if (const au" -->

Clean declines use packet projections. Depth setup, projection, and launch errors propagate. Terminal CUDA errors take precedence. [FP8 dispatch](../include/sglang/text/mtp/draft_projection.h#L83)<!-- prefix: "void project" --> · [Setup status](../src/text/draft_depth.cu#L208)<!-- prefix: "std::optiona" --> · [Submission](../include/sglang/text/mtp/draft_projection.h#L16)<!-- prefix: "inline void " -->

With depth scoring ready, single-rank eight-lane rounds commit together. Callback-free wider rounds can commit 16 lanes. Full repairs share FC and K/V. Sparse repairs use four-lane packets. In-context scalar positions group rotary and KV append. Other lanes append separately. [Commit](../src/decoder_batch.cpp#L703)<!-- prefix: "const int co" --> · [Repair](../src/decoder_mtp.cpp#L670)<!-- prefix: "if (packed &" --> · [Rotary](../src/decoder_mtp.cpp#L476)<!-- prefix: "const bool g" -->

B200 proposal and repair use native BF16 FC and joined FP8 K/V for uncaptured single rows or two unpacked rows sharing rank-local weights. Setup preloads four SM100 kernels without local memory. Declines, capture, and other shapes keep ordinary projections. [Setup](../include/sglang/text/mtp/draft_stream.cuh#L142)<!-- prefix: "if (attribut" --> · [Single row](../src/decoder_mtp.cpp#L390)<!-- prefix: "const bool s" --> · [Two rows](../src/decoder_mtp.cpp#L482)<!-- prefix: "const bool s" -->

### Certified Tensor Core target head

B200 reuses draft INT8 weights with `sm_100a` TMA and `tcgen05`. Eager greedy singletons qualify with 1–5 unscored rows. M20 keeps BF16 logits for requested scores. K0, sampling, constraints, raw logits, and scored singletons use BF16. Two-row graphs keep their head. Packets of 2–20 rows share one projection with 9.7 MiB workspace staging. Declines use grouped scoring. Operator errors propagate. [Score kernel](../src/text/tc_vocab.cu#L78)<!-- prefix: "__global__ _" --> · [Dispatch guard](../src/decoder_verify.cpp#L707)<!-- prefix: "if (round.co" --> · [Score selection](../src/decoder_verify.cpp#L612)<!-- prefix: "if (scored_r" --> · [Graph path](../src/decoder_verify.cpp#L909)<!-- prefix: "if (graph &&" --> · [Packed staging](../include/sglang/decoder_verify.h#L66)<!-- prefix: "bf16_head_pa" --> · [Build target](../xmake.lua#L92)<!-- prefix: "target(\"sgla" -->

Eager greedy K4 can score two complete M20 packets together without generated logprobs. Branch, residual, and gain need 16-byte alignment. Final RMSNorm writes identical BF16 values to lane and packed input. Declines use packet scoring. Submitted CUDA errors propagate. [M40 guard](../src/decoder_verify.cpp#L975)<!-- prefix: "if (vocabula" --> · [Operand checks](../src/text/tc_vocab.cu#L410)<!-- prefix: "bool try_tc_" --> · [RMS pack](../src/text_normalization.cu#L178)<!-- prefix: "__global__ v" -->

The head adds 3,973,120 model bytes. Unsupported maps or unaligned weights decline setup. Initial M5/M20 scratch (5,386,120/21,544,480 bytes before rounding) lasts until batch teardown. TP batches with at least two slots prepare M10 scratch. M40 tries one lazy slab (43,088,960 bytes before rounding) after reserve and CUDA checks. Reserve shortage, recognized setup or map rejection, and OOM use M20. A rejected map retains the slab until teardown. Unexpected CUDA errors propagate. [Weight setup](../src/text/tc_vocab.cu#L246)<!-- prefix: "std::shared_" --> · [Scratch setup](../src/text/tc_vocab.cu#L279)<!-- prefix: "std::shared_" --> · [Batch allocation](../include/sglang/runtime/execution/decoder_batch_state.h#L107)<!-- prefix: "storage-\u003etc_" --> · [Lazy slab](../src/text/tc_vocab.cu#L310)<!-- prefix: "bool prepare" -->

Row-max scaling gives exact INT32 dots. FP64 bounds cover FP32 subnormal rounding. Nonfinite or oversized operands get unbounded intervals. Upper bounds reaching the largest lower bound replay the original BF16 FMA order. Empty or overflowing lists replay all tokens. Refinement skips NaNs and favors low IDs on ties. [Quantization](../src/text/tc_vocab.cu#L358)<!-- prefix: "check_direct" --> · [Bounds](../include/sglang/text/mtp/tc_vocab_certificate.cuh#L114)<!-- prefix: "__device__ I" --> · [Refinement](../include/sglang/text/mtp/tc_vocab_certificate.cuh#L244)<!-- prefix: "__global__ v" -->

M10 refines each row’s highest-lower-bound tile with exact one-split BF16 replay to raise the threshold. Inclusive BF16-rounded upper bounds preserve lowest-ID ties. Per-row counts outside 1–16,384 trigger full BF16 replay. The interval includes `2^-9` times the norm product over 320 groups, assuming empirical SM100 shared-exponent alignment and truncation of 16 products plus the FP32 accumulator. [TP2 dispatch](../src/decoder_verify.cpp#L870)<!-- prefix: "score_verifi" --> · [Packed bound](../include/sglang/text/mtp/tc_vocab_short.cuh#L68)<!-- prefix: "__device__ i" --> · [Full replay guard](../include/sglang/text/mtp/tc_vocab_short.cuh#L180)<!-- prefix: "__device__ i" --> · [B200 accumulation study](https://arxiv.org/html/2512.07004v4)

Compare exact IDs and BF16 scores at M1–5, M10, M20, and M40, including ambiguous, scored, raw-logit, sampled, constrained, and graph cases. Check M40 normalized rows, commits, same-batch setup-decline recovery, and unexpected errors.

### Draft attention

Visible length includes the current token. The per-row route uses tiled MXFP8 at 8–256 tokens and parallel scores with ordered value replay at 257–5,173. [Draft dispatch](../include/sglang/text/mtp/gqa_attention.cuh#L67)<!-- prefix: "constexpr in" -->

On B200, packets of one to four scored rows use online split GQA. See [16-row grouping](singleton-gqa-decode.md#grouped-long-draft-reads). Visible length is 2,048–262,144. Scratch holds one chunk plus carry. Each lane needs full 24-query, 4-KV-head, 256-channel geometry, valid KV and page coverage, and aligned buffers. Rows beyond capacity still select online; the long batch takes remaining rows with sufficient scratch. Geometry mismatches reject the whole batch; earlier rows may have already submitted. [Scored wave call](../src/decoder_mtp.cpp#L508)<!-- prefix: "if (score) {" --> · [Lane selection](../include/sglang/text/mtp/gqa_attention.cuh#L94)<!-- prefix: "cudaError_t " --> · [Alignment](../include/sglang/text/gqa_decode_split.cuh#L70)<!-- prefix: "__host__ __d" -->

The online route shares target split GQA and can change draft proposals through its reduction order. An invalid combine replays the affected head in FP64. [Split launch](../include/sglang/text/gqa_target_split.cuh#L124)<!-- prefix: "cudaError_t " --> · [Head recovery](../include/sglang/text/gqa_target_split.cuh#L16)<!-- prefix: "if (!gqa_dec" --> · [Target round](mtp.md#runtime-path)

Other long calls score 4,096 positions at a time and carry the FP32 maximum, denominator, and weighted sum. Ordered paths use aligned V loads or assemble equivalents. Short calls below eight tokens or with insufficient scratch use direct attention. KV-only repair skips attention. [Chunked replay](../include/sglang/text/mtp/gqa_long.cuh#L152)<!-- prefix: "cudaError_t " --> · [Single-row wave](../src/decoder_mtp.cpp#L451)<!-- prefix: "if (count ==" --> · [Repair waves](../src/decoder_mtp.cpp#L650)<!-- prefix: "void MtpDraf" -->

Each lane reserves 496,640 bytes for logits; ordered long attention 419,328 for scores and carry; online split partials at most 399,360. Both reuse logits storage on the lane stream. [Score storage](../include/sglang/text/mtp/draft_state.h#L49)<!-- prefix: "ExactGqaWork" --> · [Scratch bound](../include/sglang/text/gqa_decode_split.cuh#L42)<!-- prefix: "static_asser" -->

Compare B200 K4 with fresh K0 under matched target dispatch.

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

Send an empty GET to either model-info path. Both ignore the query and follow the [shared redirect rules](http-server.md#request-targets-and-redirects). A GET with a body or an unknown path returns HTTP 404. Other valid methods return HTTP 405 with `Allow: GET` and `{"detail":"Method Not Allowed"}`. For HEAD, the server sends only the headers. Malformed HTTP still fails in the reader. [Method response](../apps/server/http.cpp#L627)<!-- prefix: "void write_g" --> · [Request line](../apps/server/http.cpp#L128)<!-- prefix: "HttpRequest " -->

Use [`/ready`](http-server.md) to check the decoder worker. Use a [tokenization request](native-tokenization.md#verification) to check tokenizer access.

### Discovery targets and errors

The list route ignores the query. After HTTP parsing, methods other than GET or a nonempty body return HTTP 404.

Retrieval ignores the query and any successfully parsed body. After shared path decoding, the ID is the entire suffix after `/v1/models/`, including trailing slashes. An interior line feed prevents the retrieval route from matching. IDs match case-sensitively. An empty or unknown ID returns HTTP 404 with an `error` object. It contains `message` (`The model '<id>' does not exist`), `type` (`invalid_request_error`), `param` (`model`), and `code` (`model_not_found`). Other methods return HTTP 405 with `Allow: GET` and `{"detail":"Method Not Allowed"}`. `HEAD` sends only the headers. [ID decoding](../apps/server/http.cpp#L346)<!-- prefix: "std::string " --> · [Lookup and errors](../apps/server/prefix_controls.cpp#L467)<!-- prefix: "void retriev" --> · [Method response](../apps/server/http.cpp#L556)<!-- prefix: "void write_m" -->

### Server-info targets

Both aliases use the shared decoded path. A GET ignores the query and any successfully framed body. It returns HTTP 200 even when `ready` is `false`. A trailing slash redirects to the corresponding alias. An unknown path returns HTTP 404. Other valid methods return HTTP 405 with `Allow: GET` and `{"detail":"Method Not Allowed"}`. HEAD sends only the headers. Invalid HTTP framing fails before the handler. [Query split](../apps/server/http.cpp#L143)<!-- prefix: "const auto s" --> · [Method reply](../apps/server/http.cpp#L556)<!-- prefix: "void write_m" --> · [Worker flag](../apps/server/batch_service.cpp#L719)<!-- prefix: "bool BatchSe" -->

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

Use the [copyable idle and load probes](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/docs/native-model-info.md#server-information-at-idle-and-under-load) with Bash, `curl`, and `jq`. Require identical alias responses at idle, correct effective options and paths, exactly two redacted checkpoint flags, no canary values, positive startup time, and zero running requests and generation tokens. Verify both aliases ignore a GET body, query, and auth header. Check HEAD, POST, and PUT return 405 with `Allow: GET`. HEAD has no body. Require slash redirects to preserve the query, an unknown path to return 404, and `/health_generate` to leave both counters unchanged.

Run the two concurrent 96-token requests from the probe. Require both lanes visible together, a nondecreasing cumulative token count across both aliases, including a sample strictly between 0 and 192, and final ready state with zero running requests and exactly 192 generation tokens. If polling misses overlap, repeat with longer requests and adjust the expected total. Repeat after accounting changes with prefix reuse and MTP output. Omit `--eager` for prefix reuse.

[Full verification appendix](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/docs/native-model-info.md#verification)
