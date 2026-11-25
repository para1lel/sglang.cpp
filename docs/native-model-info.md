# Native model and server information

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP control | Match `GET /model_info` and its deprecated alias `GET /get_model_info`. | [Control route](../apps/server/prefix_controls.cpp#L506)<!-- prefix: "if (!model_i" --> |
| Model discovery | List or retrieve the served model card. | [List dispatch](../apps/server/prefix_controls.cpp#L531)<!-- prefix: "if (path == " --> · [Retrieval dispatch](../apps/server/prefix_controls.cpp#L518)<!-- prefix: "if (model_re" --> · [Shared card writer](../apps/server/prefix_controls.cpp#L453)<!-- prefix: "std::string " --> |
| Server | Report checkpoint and serving properties from the startup profile. | [Metadata writer](../apps/sglang-server.cpp#L248)<!-- prefix: "std::string " --> · [Qwen serving profile](../src/runtime/model/qwen_serving_profile.cpp#L134)<!-- prefix: "const Servin" --> |
| Checkpoint | Supply the supported model type and architecture. | [Checkpoint identity](../include/sglang/runtime/model/qwen_text_model.h#L8)<!-- prefix: "static const" --> |
| Server information | Match both server-info paths on the HTTP reader and serialize one snapshot per GET. | [Route](../apps/server/prefix_controls.cpp#L690)<!-- prefix: "if (server_i" --> · [Callback](../apps/sglang-server.cpp#L461)<!-- prefix: "return serve" --> · [Response fields](../apps/sglang-server.cpp#L278)<!-- prefix: "std::string " --> |
| Launcher | Record options and supply the executor to the scheduler. | [Command sanitization](../apps/sglang-server.cpp#L106)<!-- prefix: "void append_" --> · [Startup composition](../apps/sglang-server.cpp#L384)<!-- prefix: "auto executo" --> |
| Batch scheduler | Snapshot admitted user jobs, published output tokens, and worker readiness under the scheduler mutex. | [Runtime snapshot](../apps/server/batch_service.cpp#L784)<!-- prefix: "BatchRuntime" --> · [Token accounting](../apps/server/batch_service.cpp#L279)<!-- prefix: "counters.gen" --> |

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

Send an empty `GET /v1/models` to get HTTP 200 with `object: "list"` and one card in `data`. Send `GET /v1/models/Qwen3.8-27B-FP8` for the card directly as an HTTP 200 JSON object.

The card uses `Qwen3.8-27B-FP8` for `id` and `root`. Use this ID for native chat and text completions. The card sets `object` to `model`, `owned_by` to `sglang`, and `parent` to `null`. The `created` field gives the creation time in integer Unix seconds. The `max_model_len` field gives the 262,144-token context shared by prompt and output. Available GPU memory can reduce admitted capacity. The `benchmark` alias belongs to the [token-ID API](../benchmarks/streaming/README.md#wire-contract).

Pinned SGLang can list loaded LoRA adapters and retrieves its configured base served name. Native discovery exposes the single fixed checkpoint ID. [Pinned model list](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1998)<!-- prefix: "@app.get(\"/v" --> · [Pinned model retrieval](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L2030)<!-- prefix: "@app.get(\"/v" -->

### Server information

`GET /server_info` and its alias `GET /get_server_info` return the same JSON schema. Upstream deprecates the alias. The reader calls the batch service directly, even with all lanes occupied or prefix reuse disabled. Each call returns the startup configuration and a fresh scheduler snapshot. [Reader dispatch](../apps/server/prefix_controls.cpp#L692)<!-- prefix: "write_json(c" -->

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
| `enable_multimodal` | Boolean. `true` for [image and video inputs](chat-serving.md#image-and-video-inputs); the vision encoder loads on first use. [Capability field](../apps/sglang-server.cpp#L287)<!-- prefix: "body += R\u0022(," --> |
| `mtp_online_gqa` | Boolean. True on B200 with positive server draft depth and capacity for at least two lanes. Each scored draft wave checks its lanes. [Metadata check](../apps/sglang-server.cpp#L309)<!-- prefix: "body += back" --> |
| `mtp_int8_head` | Boolean indicating availability of the [INT8 draft head](#int8-draft-head). |
| `certified_int8_verify_head` | Boolean indicating availability of the [certified Tensor Core head](#certified-tensor-core-target-head). |
| `fp8_prefill` | String `"tensor-core"` on B200 or `"exact"` otherwise. This reports backend policy; individual grants still apply shape and resource guards. [Backend selection](../apps/sglang-server.cpp#L319)<!-- prefix: "const bool t" --> |
| `target_gqa` | String `"grouped-split"` on B200, or `"exact"` otherwise. Per-row shape and numerical guards still apply. [Backend selection](../apps/sglang-server.cpp#L322)<!-- prefix: "sglang::json" --> |
| `paired_prefill` | Boolean. True on B200 with at least two lanes and a step budget of at least 2,048. The [whole-tile selector](native-prefill-workspace.md#projection-pairs-for-mtp) determines eligibility and workspace fallback per step. [Metadata check](../apps/sglang-server.cpp#L324)<!-- prefix: "const bool p" --> |
| `mtp_seed_fc` | Null at K0, otherwise the configured [draft seed FC policy](mtp.md#runtime-path). This field is exclusive to the server information routes. [Metadata builder](../apps/sglang-server.cpp#L487)<!-- prefix: "std::string " --> |
| `sampling_defaults` | String `"model"` or `"openai"`, according to the serving configuration. |
| `reasoning_parser`, `tool_call_parser` | JSON `null` when disabled. Enabled values are `"qwen3"` and `"qwen3_coder"`, respectively. |
| `eager`, `trace_batches`, `batch_cuda_graphs` | Booleans for eager execution, batch tracing, and the effective [batch graph policy](batch-graphs.md). `batch_cuda_graphs` defaults to `true`. It is `false` with `--eager`, even with `--batch-cuda-graphs`. |
| `startup_time` | Object with one numeric field, `server_e2e`: elapsed monotonic seconds from before argument parsing through listener creation, fixed for the process lifetime. |
| `internal_states` | Array with one object for the native scheduler rank. |

For B200 Tensor Core prefill with positive depth, `mtp_seed_fc` reports `preferred: "tma_tcgen05_bf16_fp32"`, `fallback: "wmma_bf16_fp32"`, and `outside_shape_limits: "simt_bf16_fp32"`. The limits are `min_prompt_tokens: 4096`, `min_rows: 64`, and `max_rows: 128`. Other positive-depth configurations report `preferred: "simt_bf16_fp32"`. Launches still depend on the request, operands, and CUDA preflight. [Policy attachment](../apps/sglang-server.cpp#L346)<!-- prefix: "return body " -->

| Field in `internal_states[0]` | JSON type and meaning |
| --- | --- |
| `dp_rank`, `world_size` | Integers `0` and `1`. |
| `effective_max_running_requests_per_dp` | Integer lane capacity for this rank, equal to `max_running_requests`. |
| `ready` | Boolean. It becomes `false` on permanent decoder worker failure, using the same flag as `/ready`. |
| `num_running_reqs` | Integer count of admitted user generation items still in scheduler slots. It excludes waiting HTTP work and internal health jobs. |
| `generation_tokens_total` | Cumulative integer count of published user output IDs, including additional MTP IDs. It advances at scheduler publication, independently of socket delivery and prefix reuse. Health generation contributes zero. |

The command retains switches and validated numeric or enumerated operands. It replaces every `--checkpoint` operand, including superseded paths, with `'<redacted>'`. The two public path fields still identify the selected checkpoint. Environment variables, executable paths, headers, query values, and request bodies do not supply command text. [Argument recording](../apps/sglang-server.cpp#L125)<!-- prefix: "Arguments pa" --> · [Startup interval](../apps/sglang-server.cpp#L421)<!-- prefix: "const auto s" --> · [Listener timestamp](../apps/sglang-server.cpp#L442)<!-- prefix: "const std::c" -->

Pinned SGLang exposes resolved settings and a launch command at the top level, with startup durations, per-DP state, version, frontend, and inline-system capability. The native schema reports its own settings and counters. It omits Python-specific options, `kv_events`, and upstream `last_gen_throughput` and `memory_usage` measurements. [Pinned Python routes](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L888)<!-- prefix: "@app.get(\"/g" --> · [Resolved fields](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/server_utils.py#L206)<!-- prefix: "def build_se" --> · [Scheduler measurements](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler_components/internal_state.py#L75)<!-- prefix: "state[\"last_" -->

### INT8 draft head

B200 drafts use signed INT8 when unconstrained, BF16 when constrained. The INT8 copy adds 1,272,391,680 device bytes per model. A nonfinite weight aborts startup with its token ID. K0 skips draft scoring. [Weight preparation](../src/decoder_mtp.cpp#L121)<!-- prefix: "if (capabili" -->

INT8 quantization uses absolute maxima per weight row and activation. Zero rows yield zero logits for finite operands. FP64 rescaling rounds to BF16. Lowest token ID wins ties. Invalid INT8 scores retry the same activation in BF16. Without a finite retry winner, the lane stops drafting. Target verification and repair continue. [Quantization](../src/text_decode_bandwidth.cu#L176)<!-- prefix: "__device__ v" --> · [Retry](../src/decoder_mtp.cpp#L832)<!-- prefix: "bool MtpDraf" -->

Four unconstrained lanes share four vocabulary rows per warp. Each weight vector serves four activations with independent signed INT32 sums, FP64 rescaling, and BF16 rounding. Weights and quantized activations need 16-byte alignment. Each lane writes its logits before selection. One to three lanes use ordinary INT8 kernels. [Wave scoring](../src/decoder_mtp.cpp#L367)<!-- prefix: "void MtpDraf" --> · [Four-lane kernel](../src/text_decode_bandwidth.cu#L304)<!-- prefix: "__global__ _" --> · [Batch dispatch](../src/text_decode_bandwidth.cu#L798)<!-- prefix: "cudaError_t " -->

Eligible callback-free B200 K4 waves of 8, 16, or 32 lanes reuse verification scratch. TMA scores active rows. Winner selection and clean setup decline use four-lane packets. Setup caches readiness or decline per allocation. [Dispatch](../src/decoder_batch.cpp#L561)<!-- prefix: "const bool s" --> · [Setup cache](../src/decoder_mtp.cpp#L602)<!-- prefix: "bool MtpDraf" --> · [TMA setup](../src/text/draft_depth.cu#L162)<!-- prefix: "std::optiona" --> · [Scoring](../src/decoder_mtp.cpp#L676)<!-- prefix: "check(projec" -->

### Grouped draft projections

Full proposal depths pack eight rows in eight-lane waves and 16 in wider waves. Packed passes batch input normalization, try BF16 Tensor Core FC, and batch FP8 projections, activations, and output normalization. Sparse proposals and attention reads use four-lane packets. [Full packets](../src/decoder_mtp.cpp#L650)<!-- prefix: "const int pr" --> · [FC dispatch](../include/sglang/text/mtp/draft_projection.h#L48)<!-- prefix: "if (const au" -->

Setup decline uses packet projections. Depth setup, projection, and launch errors propagate. Recorded terminal CUDA errors outrank nonterminal results during depth setup and failed grouped projection submissions. [FP8 dispatch](../include/sglang/text/mtp/draft_projection.h#L71)<!-- prefix: "void project" --> · [Setup status](../src/text/draft_depth.cu#L124)<!-- prefix: "std::optiona" --> · [Submission](../include/sglang/text/mtp/draft_projection.h#L15)<!-- prefix: "inline void " -->

Full eight-lane verification commits together when depth scoring is ready. Callback-free wider rounds can commit 16 lanes. Full repairs share FC and packed K/V. Sparse repairs use four-lane packets. In-context scalar positions permit grouped rotary and KV append. Otherwise lanes append separately. [Commit](../src/decoder_batch.cpp#L629)<!-- prefix: "const int co" --> · [Repair](../src/decoder_mtp.cpp#L704)<!-- prefix: "if (packed &" --> · [Rotary](../src/decoder_mtp.cpp#L539)<!-- prefix: "const bool g" -->

### Certified Tensor Core target head

B200 reuses draft INT8 weights with `sm_100a` TMA and `tcgen05`. Eager greedy MTP singletons qualify with one to five unscored rows. Twenty-row packets keep BF16 logits for requested score rows. K0, sampling, constraints, raw logits, and scored singletons use BF16. Two-row graphs keep their existing head. Packets of two to twenty rows share one full-weight head projection. Workspace staging covers the 20-row packet capacity, about 9.7 MiB, from the workspace arena at construction. Declines fall back to the grouped path; operator errors propagate. [Score kernel](../src/text/tc_vocab.cu#L114)<!-- prefix: "__global__ _" --> · [Dispatch guard](../src/decoder_verify.cpp#L761)<!-- prefix: "if (round.co" --> · [Score selection](../src/decoder_verify.cpp#L683)<!-- prefix: "if (scored_r" --> · [Graph path](../src/decoder_verify.cpp#L894)<!-- prefix: "if (graph \u0026\u0026" --> · [Packed staging](../src/decoder_verify.cpp#L73)<!-- prefix: "bf16_head_pa" --> · [Build target](../xmake.lua#L90)<!-- prefix: "target(\"sgla" -->

Packed eager greedy K4 verification can score two complete 20-row packets together when no row needs generated logprobs. Branch, residual, and gain require 16-byte alignment. Final RMSNorm writes the same BF16 values to the lane and packed input. Preflight decline uses packet scoring. CUDA errors after submission propagate. [M40 guard](../src/decoder_verify.cpp#L960)<!-- prefix: "if (vocabula" --> · [Operand checks](../src/text/tc_vocab.cu#L410)<!-- prefix: "bool try_tc_" --> · [RMS pack](../src/text_normalization.cu#L211)<!-- prefix: "__global__ v" -->

The head adds 3,973,120 model bytes. Unsupported maps or unaligned weights decline setup. Initial 5- or 20-row scratch (5,386,120 or 21,544,480 bytes before rounding) lasts until batch teardown. Eligible M40 setup tries once per batch. After resident reserve and CUDA setup checks, it allocates one additional slab (43,088,960 bytes before rounding). Reserve shortage, recognized setup or map rejection, or allocation OOM uses M20. A declined map leaves the slab owned until teardown. Unexpected CUDA errors propagate. [Weight setup](../src/text/tc_vocab.cu#L235)<!-- prefix: "std::shared_" --> · [Scratch setup](../src/text/tc_vocab.cu#L267)<!-- prefix: "std::shared_" --> · [Batch allocation](../include/sglang/runtime/execution/decoder_batch_state.h#L105)<!-- prefix: "storage-\u003etc_" --> · [Lazy slab](../src/text/tc_vocab.cu#L297)<!-- prefix: "bool prepare" -->

Row maxima scale signed INT8 for exact INT32 dots. FP64 bounds cover FP32 subnormal rounding. Nonfinite or oversized operands get unbounded intervals. Candidates whose upper bound reaches the largest lower bound replay the BF16 dot in its original FMA order. Empty or overflowing lists replay all tokens. Refinement skips NaNs and breaks ties by lowest ID. [Quantization](../src/text/tc_vocab.cu#L346)<!-- prefix: "check_direct" --> · [Bounds](../include/sglang/text/mtp/tc_vocab_certificate.cuh#L114)<!-- prefix: "__device__ I" --> · [Refinement](../include/sglang/text/mtp/tc_vocab_certificate.cuh#L243)<!-- prefix: "__global__ v" -->

Compare exact winner IDs and BF16 winning scores with the ordinary head at 1–5, 20, and 40 rows. Exercise ambiguous candidates, scored and raw-logit rows, sampling, constraints, and graph replay. Compare M40 normalized rows and committed state across clean setup decline. Check same-batch recovery after setup decline and propagation of unexpected errors.

### Draft attention

Visible length includes the current token. Draft attention uses tiled MXFP8 at 8–256 tokens and parallel scores with ordered value replay at 257–5,173 with standard scratch. [Draft dispatch](../include/sglang/text/mtp/gqa_attention.cuh#L19)<!-- prefix: "constexpr in" -->

On B200, scored waves of two to four rows use grouped online split GQA for eligible lanes. Visible length must exceed score capacity (5,174 or more with standard scratch). Each lane needs valid KV and page coverage, aligned query and KV buffers, and sufficient aligned scratch. Lanes keep independent positions, page tables, scratch, and routes. [Scored wave call](../src/decoder_mtp.cpp#L567)<!-- prefix: "if (score) {" --> · [Lane selection](../include/sglang/text/mtp/gqa_attention.cuh#L46)<!-- prefix: "cudaError_t " --> · [Alignment](../include/sglang/text/gqa_decode_split.cuh#L70)<!-- prefix: "__host__ __d" -->

The online route shares target split GQA and can change draft proposals through its reduction order. An invalid combine replays the affected head in FP64. [Split launch](../include/sglang/text/gqa_target_split.cuh#L132)<!-- prefix: "cudaError_t " --> · [Head recovery](../include/sglang/text/gqa_target_split.cuh#L18)<!-- prefix: "if (!gqa_dec" --> · [Target round](mtp.md#runtime-path)

Other long calls score 4,096 positions at a time and carry the FP32 maximum, denominator, and weighted sum. Ordered paths use four-byte aligned V loads or assemble equivalent values. Short calls below eight tokens or with insufficient scratch use direct attention. KV-only repair skips attention. [Chunked replay](../include/sglang/text/mtp/gqa_long.cuh#L152)<!-- prefix: "cudaError_t " --> · [Single-row wave](../src/decoder_mtp.cpp#L513)<!-- prefix: "if (count ==" --> · [Repair waves](../src/decoder_mtp.cpp#L690)<!-- prefix: "void MtpDraf" -->

Each lane reserves 496,640 bytes for logits. Ordered long attention uses 419,328 bytes for scores and carry. Online split partials use at most 399,360 bytes. Both reuse logits storage on the lane stream before scoring. [Score storage](../src/decoder_mtp.cpp#L51)<!-- prefix: "ExactGqaWork" --> · [Scratch bound](../include/sglang/text/gqa_decode_split.cuh#L47)<!-- prefix: "static_asser" -->

Compare B200 K4 with fresh K0 under matched target dispatch. Earlier profiles and settings are in Git history at `8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3`.

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

Both aliases use the shared decoded path. A GET ignores the query and any successfully framed body. It returns HTTP 200 even when `ready` is `false`. A trailing slash redirects to the corresponding alias. An unknown path returns HTTP 404. Other valid methods return HTTP 405 with `Allow: GET` and `{"detail":"Method Not Allowed"}`. HEAD sends only the headers. Invalid HTTP framing fails before the handler. [Query split](../apps/server/http.cpp#L143)<!-- prefix: "const auto s" --> · [Method reply](../apps/server/http.cpp#L556)<!-- prefix: "void write_m" --> · [Worker flag](../apps/server/batch_service.cpp#L720)<!-- prefix: "bool BatchSe" -->

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
    (.lmcache_enabled == false) and (.lmcache_available == false) and
    .enable_multimodal and has("mtp_seed_fc") and .mtp_seed_fc == null and
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
