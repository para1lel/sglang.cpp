# SGLang HTTP route coverage

The baseline native dispatch overlaps 21 of 92 default Python SGLang paths for Qwen3.8-27B-FP8 on one B200. With native prefix reuse enabled, the baseline overlap is 23. The baseline tables record method, launch and model limits, and the observations report the HTTP cases exercised. Native has since added [pointwise scoring](../serving/pointwise-score.md), [five Ollama paths](../serving/ollama.md), [Anthropic Messages](../serving/anthropic-messages.md), [three Responses paths](../serving/openai-responses.md), and [text Decisions](../serving/decisions.md), taking the default registered overlap to 32 of 92, or 34 with prefix reuse. Responses retrieval and cancellation require the optional response store.

## Source and scope

Accessed November 22, 2026. The native registry uses baseline `bd55e77edd926e4db9a8e2a115b54b6d773eb879` with the current chat tokenization contract and Anthropic count route. SGLang is `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`, read from the local pinned tree at `cache/research-src/sglang-f04d886bf79aa3d6237cbb12a3311838eb9f9bd8` under the shared user directory. The count covers the default Python autoregressive app. Ray reuses that app. `SGLANG_RUST_SERVER=1` selects the Rust listener, and `sgl-router` has its own registry. [Launch](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/launch_server.py#L17)<!-- prefix: "def run_serv" --> · [Rust selection](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L3275)<!-- prefix: "if envs.SGLA" --> · [Native startup](../../apps/sglang-server.cpp#L328)<!-- prefix: "serve_connec" -->

## Registry counts

| Upstream Python launch | Registered paths | Native prefix reuse off (default), shared / missing | Native prefix reuse on, shared / missing |
| --- | ---: | ---: | ---: |
| Default | 92 | 21 / 71 | 23 / 69 |
| `--enable-metrics` | 93 | 21 / 72 | 24 / 69 |

Count each route template or mount once across methods and each alias separately. The default has 85 paths from the HTTP module, three from its included routers, and four FastAPI documentation paths. `DISABLE_OPENAPI_DOC=1` removes four missing paths, and `DUMPER_SERVER_PORT=reuse` adds one. Ollama route variables and `AIP_PREDICT_ROUTE` replace names. These counts use the default names and include routes whose handlers require a different model, feature or deployment. Native-only `/reset_prefix_cache` is outside the upstream registry. [App and included routers](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L500)<!-- prefix: "app = FastAP" --> · [Metrics gate](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L321)<!-- prefix: "if get_obser" --> · [Dumper gate](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L945)<!-- prefix: "@app.api_rou" -->

Table line numbers refer to upstream `http_server.py`. `G/P/U/D/H/W` denote GET/POST/PUT/DELETE/HEAD/WebSocket. `✓` marks native dispatch subject to the listed limits. `—` marks a missing native path. Native splits the query, percent-decodes the path and redirects recognized trailing-slash targets with 307. Model detail treats the trailing slash as ID data. The loopback HTTP/1.1 listener has 8 KiB header and 2 MiB body limits. Method errors depend on the handler. [Native transport](../http-server.md#request-targets-and-redirects) · [Control dispatch](../../apps/server/prefix_controls.cpp#L512)<!-- prefix: "void Connect" --> · [Work dispatch](../../apps/server/prefix_controls.cpp#L644)<!-- prefix: "void Connect" -->

## Baseline Python service comparison

| Method and path | Native | Contract at the pin |
| --- | --- | --- |
| G `/ready` | ✓ | 758. Upstream empty 200/503 reflects startup, pause, health and shutdown. Native JSON 200/503 reflects the worker-failure flag. |
| G `/health` | ✓ | 765. Upstream empty status, generation enabled by default; native always 200 with configured limits. |
| G `/health_generate` | ✓ | 766. Empty 200/503 on both. Upstream observes detokenizer or scheduler progress and updates health state. Native observes token publication, uses an idle probe or bounded FIFO, and keeps readiness separate. |
| G `/model_info`, `/get_model_info` | ✓ | 846–885. Alias; upstream reports live model identity and runtime fields. Native reports fixed loader/model capabilities and selected parser flags. |
| G `/server_info`, `/get_server_info` | ✓ | 888–929. Native aliases report launch settings and a scheduler snapshot in the native schema. Upstream reports resolved configuration and per-DP state. |
| G `/v1/models` | ✓ | 1998. Native one canonical card; upstream uses served name and can list loaded LoRAs. |
| G `/v1/models/{model:path}` | ✓ | 2030. Lookup by full slash-containing ID and structured `model_not_found` 404. Native serves one fixed ID. |
| P/U `/generate` | ✓ | 963–1025. Native text or IDs, scalar/batch, JSON or cumulative SSE. Upstream also accepts embeds, supported media, LoRA and additional controls. In-stream error termination differs. |
| P `/abort_request` | ✓ | 1727. Empty 200 and unknown-ID no-op. Native abort registry covers `/generate` only; upstream tokenizer/scheduler also cancels OpenAI generation. Native retains dispatched-prefix matching after an exact-ID gate. |
| P/U `/encode` | — | 1028. SGLang embedding API for embedding launches. |
| P/U `/classify` | — | 1041. Reward/classification on compatible models, using embedding-shaped inputs. |
| P `/parse_function_call` | ✓ | 1739. Standalone `normal_text` and `calls` with serialized arguments. Native supports `qwen3_coder`, `step3p5` and `nanbeige` independent of chat flags. Python literals and schema unions have bounded support. |
| P `/separate_reasoning` | ✓ | 1779. Standalone `reasoning_text` and `text` split with optional blocks. Native supports `qwen3`, `qwen3-thinking`, `dots`, `interns1`, `nanbeige` and `minimax`. |
| P `/v1/completions` | ✓ | 1863. Scalar and batched text/IDs, JSON/SSE, echo and logprobs. Upstream supports multiple choices with `n`; native allows only `n=1` and serves a private benchmark token-ID format on this path. |
| P `/v1/chat/completions` | ✓ | 1873. Native Qwen templating, JSON/SSE, optional reasoning/tool parsers, inline image/video data URLs; narrower options, parser and media handling. |
| P `/v1/tokenize`, `/tokenize` | ✓ | 1911. Both accept raw text, text batches, or one chat conversation. The [native chat path](../native-tokenization.md#encode-chat) uses the checkpoint template and encodes media placeholders as token IDs. Native normalizes common tool schema aliases. Its schema validation and template support cover fewer cases. |
| P `/v1/detokenize`, `/detokenize` | ✓ | 1929. IDs to text; native errors and queue limits apply. |
| P `/v1/embeddings` | — | 1885. OpenAI embedding response; generation launch rejects embedding requests. |
| P `/v1/classify` | — | 1899. Labels/scores from a classification-model handler. |
| P/U `/v1/rerank` | — | 2106. Query/document ranking via supported cross-encoder or generative scoring. |
| P `/v1/score` | — | 2055. Candidate logprob or classification-logit scoring. |
| P `/v1/decisions` | — | 2063. Typed choice, score and yes/no via single-token labels on compatible generation models. Native added [text Decisions](../serving/decisions.md) after this baseline. |
| P `/v1/systemone` | — | 2186. System One prompt and decision layout over candidate scoring. Validation returns 422, an exception to the app's usual 400. |
| P `/v1/audio/transcriptions` | — | 1947. Multipart audio; JSON/text/verbose JSON; needs transcription adapter. |
| W `/v1/realtime` | — | 1987. Transcription sessions only; requires a supported audio model. |
| P `/v1/responses` | — | 2071. JSON/SSE create, with optional background processing and tool support. Native added [plain text and reasoning](../serving/openai-responses.md) after this baseline. |
| G `/v1/responses/{response_id}` | — | 2090. Retrieve stored response by ID. Native added opt-in retrieval after this baseline. |
| P `/v1/responses/{response_id}/cancel` | — | 2098. Cancel a background response. All three paths register even if handler initialization fails at 402–424. Native added opt-in cancellation after this baseline. |
| G/H `/` | — | 2121–2133. JSON greeting; `SGLANG_OLLAMA_ROOT_ROUTE` replaces the path and greeting. |
| P `/api/chat` | — | 2136. Ollama chat. |
| P `/api/generate` | — | 2142. Ollama text generation. |
| G `/api/tags` | — | 2150. Ollama model list. |
| P `/api/show` | — | 2156. Ollama model metadata. Each Ollama path has its own `SGLANG_OLLAMA_*_ROUTE` override. |
| P `/v1/messages` | — | 2165. Anthropic Messages via chat adapter. Native added [generation and SSE](../serving/anthropic-messages.md) after this baseline. |
| P `/v1/messages/count_tokens` | ✓ | 2175. Both return `input_tokens`. Native counts the rendered Qwen prompt on the CPU with prefix reuse off or on. Launch with `--reasoning-parser qwen3` for explicit enabled or adaptive thinking. Native HTTP 400 uses the Anthropic error envelope. [Count contract](../native-tokenization.md#count-anthropic-prompt-tokens) |
| G `/ping` | — | 2195. SageMaker empty health status; intentional pause can remain healthy. |
| P `/invocations` | — | 2221. SageMaker dispatch order `messages`, `prompt`, `input`. |
| P `/vertex_generate` | — | 2256. Vertex instances/parameters wrapper for generation; `AIP_PREDICT_ROUTE` overrides path. |

See native [generation](../native-generate.md), [abort](../native-abort.md), [text](../text-serving.md), [chat](../chat-serving.md), [tokenization](../native-tokenization.md), [metadata](../native-model-info.md), [reasoning split](../serving/separate-reasoning.md), and [parser limits](../serving/parse-function-call.md). [Parser dispatch](../../apps/server/prefix_controls.cpp#L675)<!-- prefix: "if (const au" --> · [Parser handler](../../apps/server/parse_function_call.cpp#L543)<!-- prefix: "void parse_f" -->

## Gaps by workload

- **Single-checkpoint generation.** Native dispatches three core generation paths (`/generate`, `/v1/completions`, `/v1/chat/completions`), five tokenization paths, and two standalone parser paths. Pointwise text scoring and text Decisions run on the same causal checkpoint. Generic System One wraps the same Decisions scoring and remains a separate route gap. [Score dispatch](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L2055)<!-- prefix: "@app.post(\"/" --> · [Decision guards](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/openai/serving_decisions.py#L159)<!-- prefix: "if not self." -->
- **Client interfaces.** The native Responses paths cover text and reasoning, with optional retrieval and cancellation. Responses tool workflows and cloud wrappers remain outside that scope. [Responses initialization](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L402)<!-- prefix: "try:" -->
- **Other model roles.** Embedding, classification and transcription entries require their respective workload support. The generation launch rejects embedding-shaped requests. Rerank needs a cross-encoder or a detected decoder reranker backend. Ordinary chat generation establishes no rerank coverage. [Embedding guard](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_manager.py#L1515)<!-- prefix: "if isinstanc" --> · [Reranker selection](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/openai/serving_rerank.py#L319)<!-- prefix: "async def _h" -->
- **Operations.** `/v1/loads` is a separate telemetry gap even on one worker. Native `/metrics` supplies prefix and scheduler counters when reuse is enabled. Distributed, offload, instrumentation, adapter and weight-management entries in the registry depend on their respective deployment or operator workflow. [Load contract](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/v1_loads.py#L96)<!-- prefix: "@router.get(" --> · [Native controls](../prefix-cache.md#native-http-controls)

## Baseline scoring recommendation

Implement pointwise `POST /v1/score` for the decoder with `--enable-mis` off upstream. Start with token-ID replay, using `query: []`, `items: [[...]]` and shared or per-item `label_token_ids`. The pin also accepts a query prefix and text inputs. `item_first` reverses concatenation. Each prompt gets one ordered row for distinct, in-vocabulary labels. The upstream generation request asks for those IDs' next-token logprobs with `max_new_tokens: 0`. Native `/generate` currently accepts `token_ids_logprob` only when null, so this requires selected-label readout through prefill. Defer fused scoring, extraction anchors, embedding overrides and classifier readouts. [Request schema](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/openai/protocol.py#L1619)<!-- prefix: "class Scorin" --> · [Label validation](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_manager_score_mixin.py#L920)<!-- prefix: "is_generatio" --> · [Prefill request](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_manager_score_mixin.py#L1173)<!-- prefix: "batch_reques" --> · [Native field check](../../apps/server/text_request.cpp#L728)<!-- prefix: "void validat" -->

The response has `object: "scoring"`, `model`, `scores` and `usage` with `prompt_tokens` and `total_tokens` equal. `scores` defaults to probabilities over the full vocabulary evaluated at each label. `apply_softmax: true` renormalizes over the supplied labels. A finite positive `temperature` other than 1 requires softmax. `return_token_logprobs` supplies separate full-vocabulary log-probabilities for the labels. Pointwise decoder responses include `pooled_hidden_states: null`. Invalid inputs return 400. Verify ordered rows, empty items, label validation and both probability modes against pinned upstream replay with the same prompt IDs and cache state. [Score conversion](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_manager_score_mixin.py#L1243)<!-- prefix: "def _convert" --> · [Response writer](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/openai/serving_score.py#L108)<!-- prefix: "response = S" --> · [Replay example](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/docs/docs/supported-models/decision_models.mdx#L170)<!-- prefix: "To keep a te" -->

## Python management and optional surfaces

| Method and path | Native | Operation and gate |
| --- | --- | --- |
| G/P `/flush_cache` | ✓ | 1060. Optional admin upstream. Native works with reuse on or off. A zero timeout checks immediate eligibility, and a positive timeout allows a FIFO worker to wait for idle. Native returns 200 on success and 400 on busy refusal, timeout or invalid input. It invalidates published target/draft prefixes and retains reserved storage, counters and tool grammars. Upstream also clears pools, grammar state, metrics and media preprocessing cache. |
| G/P `/start_profile` | ✓ P | 1240. Admin; upstream optional `ProfileReq` controls scheduler profiler. Native requires exact step-capture options and prefix reuse. |
| G/P `/stop_profile` | ✓ P | 1250. Admin; upstream stops profiler. Native returns captured step JSON and requires prefix reuse. |
| ASGI `/metrics` | ✓ G | 321. Upstream mounts a Prometheus ASGI exporter with `--enable-metrics`. Native GET requires prefix reuse, redirects the trailing slash and emits different series and counters. |
| G `/v1/loads` | — | Included at 529. Per-DP JSON or Prometheus load sections, rank and section filters. |
| P/U `/set_internal_state` | — | 934. Admin scheduler state mutation. |
| P `/dumper/{method}` | — | 945. Admin control; registers with `DUMPER_SERVER_PORT=reuse`. |
| P `/add_external_corpus` | — | 1077. Register external corpus. |
| P `/remove_external_corpus` | — | 1101. Remove corpus by ID. |
| G `/list_external_corpora` | — | 1118. List corpus records. All three corpus controls require Ngram speculative decoding. |
| P `/hicache/storage-backend/clear` | — | 1134. Clear backing storage. |
| U `/hicache/storage-backend` | — | 1154. Attach or switch storage; HiCache configuration and idleness apply. |
| D `/hicache/storage-backend` | — | 1184. Detach storage; idleness applies. |
| G `/hicache/storage-backend` | — | 1207. Live backend settings. |
| G `/get_hicache_l3_cache_layout` | — | 1223. Scheduler-derived L3 layout or disabled reason. |
| G/P `/set_trace_level` | — | 1260. Change trace level. |
| G/P `/freeze_gc` | — | 1270. Freeze Python GC. |
| G/P `/start_expert_distribution_record` | — | 1282. Start/reset expert recording. |
| G/P `/stop_expert_distribution_record` | — | 1292. Stop expert recording. |
| G/P `/dump_expert_distribution_record` | — | 1302. Dump expert records; requires appropriate MoE instrumentation. |
| P `/update_weights_from_disk` | — | 1312. Swap weights from disk. |
| P `/init_weights_send_group_for_remote_instance` | — | 1340. Initialize remote weight send group. |
| P `/send_weights_to_remote_instance` | — | 1358. Send weights to remote instance. |
| G `/remote_instance_transfer_engine_info`, `/get_remote_instance_transfer_engine_info` | — | 1375–1408. Rank-specific remote transfer bootstrap query and deprecated alias. |
| P `/init_weights_update_group` | — | 1411. Create update group. |
| P `/destroy_weights_update_group` | — | 1426. Tear down update group. |
| P `/update_weights_from_tensor` | — | 1441. Apply serialized tensor update. |
| P `/begin_weight_update` | — | 1462. Begin update transaction. |
| P `/end_weight_update` | — | 1476. End update transaction. |
| P `/plugin_weight_update` | — | 1490. Dispatch configured plugin update. |
| P `/update_weights_from_distributed` | — | 1507. Update over distributed group. |
| P `/update_weights_from_ipc` | — | 1526. Update via IPC; can mark initial weights loaded. |
| P `/update_weight_version` | — | 1544. Update reported version while idle; optional abort-all. |
| G/P `/get_weights_by_name` | — | 1575. Named parameter inspection. |
| G/P `/release_memory_occupation` | — | 1590. Release selected memory occupation. |
| G/P `/resume_memory_occupation` | — | 1601. Restore selected memory occupation. |
| G/P `/weights_checker` | — | 1612. Admin check action; can return rank information and checksums. |
| G/P `/slow_down` | — | 1633. Inject scheduler slowdown. |
| P `/pd_role_switch` | — | 1646. Switch supported disaggregation role. |
| P `/load_lora_adapter` | — | 1662. Load adapter from path. |
| P `/load_lora_adapter_from_tensors` | — | 1672. Load tensor payload. |
| P `/unload_lora_adapter` | — | 1684. Remove loaded adapter. |
| G/P `/open_session` | — | 1694. Allocate stateful session ID. |
| G/P `/close_session` | — | 1708. Close session by ID. |
| G/P `/configure_logging` | — | 1718. Set request logging options. |
| P `/pause_generation` | — | 1836. Pause generation using scheduler pause policy. |
| P `/continue_generation` | — | 1848. Resume generation. |
| P `/scale_elastic_ep` | — | Included at 545. Asynchronous scale; disabled backend returns 404 after validation. |
| G `/is_scaling_elastic_ep` | — | Included at 545. Inspect scale state; disabled backend returns 404. |
| G/H `/openapi.json`, `/docs`, `/docs/oauth2-redirect`, `/redoc` | — | 500. FastAPI built-ins, controlled by `DISABLE_OPENAPI_DOC`. |

[Management docs](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/docs/docs/basic_usage/native_api.mdx#L136)<!-- prefix: "## Flush Cac" --> · [Cache clearing](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler.py#L6269)<!-- prefix: "def flush_ca" --> · [Media cache](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_control_mixin.py#L330)<!-- prefix: "async def fl" --> · [Loads](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/v1_loads.py#L96)<!-- prefix: "@router.get(" --> · [Elastic EP](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/elastic_ep.py#L14)<!-- prefix: "@router.post" --> · [Metrics](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/server_utils.py#L61)<!-- prefix: "def add_prom" --> · [Corpus gate](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler.py#L5958)<!-- prefix: "def add_exte" --> · [Native controls](../prefix-cache.md#native-http-controls)

In the Python auth map, flush and abort use optional admin policy, profiling requires an admin key, and metadata and standalone parsing use regular policy. Optional admin routes prefer a configured admin key, then a configured regular key. Native loopback dispatch has no API-key middleware. The path overlap measures dispatch independently of credentials. [Auth map](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/utils/auth_levels.json#L3)<!-- prefix: "\"/abort_requ" --> · [Auth decisions](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/utils/auth.py#L117)<!-- prefix: "def decide_r" --> · [Native flush contract](../prefix-cache.md#cache-flush)

## Rust engine and gateway

The Rust engine mounts Python-table methods for `/model_info`, `/get_model_info`, `/server_info`, `/flush_cache`, `/weights_checker`, `/start_profile`, `/stop_profile`, `/abort_request`, both engine health paths, `/v1/loads` and the two HiCache storage paths. With metrics enabled on that listener, it mounts GET `/metrics`, `/metrics/` and `/metrics/{*path}`. Axum GET also handles HEAD. Its OpenAI routes are model list/detail (one path segment for detail), completions, chat and the four tokenization paths. `/generate` is POST only. `/ready`, `/get_server_info` and the Python-only inference families are absent. The typed server-info response identifies the Rust frontend. Flush uses immediate or deferred scheduler control, with 400 for bad queries and 503 for an unavailable core. POST `/internal/peer` forwards binary DP control. Prefill bootstrap optionally mounts G/U `/route`, P `/register_dp_rank` and P `/query_dp_ranks`. All additional internal paths lack native dispatch. [Rust assembly](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/rust/sglang-server/src/api_server/http/app.rs#L157)<!-- prefix: "let mut rout" --> · [Rust controls](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/rust/sglang-server/src/api_server/http/info.rs#L34)<!-- prefix: "pub(super) f" --> · [Rust flush](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/rust/sglang-server/src/api_server/http/info.rs#L195)<!-- prefix: "async fn flu" -->

The Axum gateway routes G/H `/healthz`, `/health`, `/liveness`, `/readyz`, `/readiness`, `/metrics`, `/v1/models`, `/server_info`, `/get_server_info`, `/model_info`, `/get_model_info`, `/health_generate`, and `/internal/kv_snapshot`. It accepts P `/v1/tokenize`, `/v1/detokenize`, `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/embeddings`, `/v1/classify`, `/flush_cache`, and P/U `/generate`, `/v1/rerank`. Gateway health, readiness and metrics describe the gateway. Worker info and inference use backend selection. Flush fans out. KV snapshot supports peer bootstrap. All methods on `/v1/responses`, `/v1/responses/{*rest}`, `/v1/conversations`, `/v1/conversations/{*rest}` and `/v1/loads` reach explicit 501 handlers. Native coverage of shared names is in the tables. Gateway-only names lack native dispatch. [Gateway registration](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/sgl-router/src/server/app.rs#L217)<!-- prefix: "pub fn build" --> · [501 routes](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/sgl-router/src/server/routes/unsupported.rs#L13)<!-- prefix: "pub(crate) c" -->

## Separate conditional listeners

| Mode | Method and path | Operation |
| --- | --- | --- |
| Python PD bootstrap | G/U `/route`, P `/register_dp_rank`, P `/query_dp_ranks`, G/H `/health` | Prefill or enabled role switching. Write and fetch KV transfer registry data, publish and query DP rank mapping, and report bootstrap health. The first three paths are also the optional Rust prefill mount, with Axum HEAD for GET. |
| Transfer-engine seed bootstrap | G/H documentation paths, G `/health`, U `/register_transfer_engine_info`, G `/get_transfer_engine_info` | Separate FastAPI listener for remote weight loading, with per-rank registration and lookup. |
| Encoder-only HTTP | P `/encode`, `/send`, `/scheduler_receive_meta_data`, `/scheduler_receive_url` | Encode media, trigger Mooncake RDMA send, wait for partition metadata, and register decoder destinations. |
| Encoder-only HTTP | G `/health`, `/health_generate`, G/P `/start_profile`, `/stop_profile` and G/H documentation paths | Encoder probes, profiler controls and FastAPI documentation. |

Native runs none of these auxiliary listeners. [Python bootstrap](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/disaggregation/common/conn.py#L3299)<!-- prefix: "def _setup_r" --> · [Rust bootstrap](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/rust/sglang-server/src/api_server/disaggregation/bootstrap.rs#L462)<!-- prefix: ".route(\"/rou" --> · [Weight bootstrap](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/engine_info_bootstrap_server.py#L28)<!-- prefix: "class Engine" --> · [Encoder](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/disaggregation/encoder/http_server.py#L284)<!-- prefix: "@app.post(\"/" -->

## Existing observations

Paths below are relative to repository `artifacts/`. The native B200 release receipt binds the running executable to the `bd55e77` release and checks parser-source digests. Its HTTP log records eight union conversion cases, HTTP 400 for an unsupported parser, text completion, chat SSE and health. A separate supplied-binary run recorded 25 parser checks across the three aliases and 30 smoke assertions over 23 native HTTP requests. That run used a candidate build. Both used the target checkpoint.

- `takeover-serving/{release-identity.log,running-identity.log,checks.log}`
- `takeover-http/run.ljyh1dU1/final-parser.CIxvPN9y/summary.md`

The historical totals below were previously reported. Their receipts are unavailable in this checkout. Earlier native B200 tests reported 26 JSON Schema assertions and 34 prompt-logprobs HTTP responses with 41 frames. The prompt-logprobs report recorded an exact warm/cold match over 9,755 scores and used upstream source evidence.

- `current-native-json-schema/evidence/checkpoint-1/{device.csv,http.log,http.exit,http/command.json}`
- `current-native-prompt-logprobs-accept/evidence/{identity.log,upstream.json,checkpoint-4/http.log}`

Earlier host HTTP fixtures exercised generation batches, abort, all tokenization aliases, text prompt forms and health. The pinned tokenizer comparison reported 300 success and 90 rejection comparisons. The model-info packet matrix reported ten matches and eight differences per configuration. The model-list check exercised the writer.

- `current-batch-generate-accept/tests/evidence/latest-native.log`
- `current-native-abort-accept/logs/test-published-final.log`
- `current-tokenization-accept/proof/{integration-http,reference-probe}.log`
- `current-completions-prompt-forms/host/frozen-wire.log`
- `current-health-generate/header-layout/cpu/health.log`
- `current-model-info-accept/proof/matrix-summary.log`
- `resume-models-api/evidence/models-response-check.log`

The receipts cover their recorded native cases. A live comparison with the pinned Python service across the registry, including launch and model gates, remains unverified.
