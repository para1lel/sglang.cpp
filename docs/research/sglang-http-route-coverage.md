# SGLang HTTP route coverage

Native dispatches 16 of the 92 distinct default Python service paths, including four tokenization paths and both model-info paths. Two of the 16 require native prefix reuse. With upstream metrics enabled the overlap increases to 17 paths, with native `/metrics` gated by prefix reuse. These counts measure path overlap. The target is Qwen3.8-27B-FP8 on one B200.

Implement `GET` and `POST /flush_cache` first. The pinned benchmark flushes after warmup with `--flush-cache` or when the backend name contains `sglang` and `SGLANG_IS_IN_CI` is true. For an SGLang backend, the client sends `POST /flush_cache` with a default `timeout` of `60.0` seconds. It uses `/reset_prefix_cache` for vLLM. The Python endpoint defaults to zero when the request omits `timeout`. Reuse the native reset's idle and exclusion logic and retain the existing reset API. Split query from path for this control and support a nonnegative timeout in seconds. Keep waits off the HTTP reader. Match Python's raw text and 200/400 flush outcomes, and its JSON 400 validation error for invalid query values. Verify immediate success, busy refusal, timeout, wake on idle and rejection of a second pending flush. Confirm which native caches and counters the new control must clear. [Benchmark docs](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/docs/docs/developer_guide/bench_serving.mdx#L89)<!-- prefix: "`--flush-cac" --> · [Benchmark trigger](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/benchmark/serving.py#L1523)<!-- prefix: "should_flush" --> · [Benchmark client](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/benchmark/serving.py#L1017)<!-- prefix: "def flush_se" --> · [Python endpoint](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L1061)<!-- prefix: "async def fl" --> · [Native reset](../../apps/server/prefix_controls.cpp#L215)<!-- prefix: "void Connect" --> · [Deferred flush](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/scheduler_components/flush_wrapper.py#L22)<!-- prefix: "def handle(s" --> · [Validation](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L673)<!-- prefix: "@app.excepti" -->

## Source and scope

Accessed November 22, 2026. Native is `9bf9f9a4205773948f63d10a22dd72de1a9693c2`. SGLang is `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`, read from the local pinned tree at `cache/research-src/sglang-f04d886bf79aa3d6237cbb12a3311838eb9f9bd8` under the shared user directory. `sglang serve` selects the Python autoregressive app by default. `SGLANG_RUST_SERVER=1` replaces that listener. The Rust engine and separate `sgl-router` gateway have different route sets. Ray reuses the Python app. Diffusion is outside this checkpoint's serving mode. [Launch](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/launch_server.py#L17)<!-- prefix: "def run_serv" --> · [Rust selection](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L3275)<!-- prefix: "if envs.SGLA" --> · [Native startup](../../apps/sglang-server.cpp#L291)<!-- prefix: "serve_connec" -->

The numeric references in the tables point to lines in [Python `http_server.py`](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/http_server.py#L500)<!-- prefix: "app = FastAP" -->. `G/P/U/D/H/W` mean GET/POST/PUT/DELETE/HEAD/WebSocket. `✓` means native dispatches the listed path, subject to the stated limits. `—` means missing from native dispatch. The default count includes four framework paths and excludes the optional metrics mount and dumper route. Native uses exact raw targets, a loopback HTTP/1.1 listener, and 8 KiB header and 2 MiB body limits. Query suffixes and trailing slashes generally miss dispatch. Unexpected methods generally return 404. Model info and generation health explicitly return 405. Forced admin routes return 403 without a configured admin key. Optional admin routes prefer the admin key and fall back to the regular API key when configured. Upstream probe and metrics prefixes bypass auth. Native has no API-key middleware. [Target parser](../../apps/server/http.cpp#L116)<!-- prefix: "HttpRequest " --> · [Control dispatch](../../apps/server/prefix_controls.cpp#L299)<!-- prefix: "void Connect" --> · [Work dispatch](../../apps/server/prefix_controls.cpp#L398)<!-- prefix: "void Connect" --> · [Auth map](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/utils/auth_levels.json#L2)<!-- prefix: "\"/\": \"normal" --> · [Auth decisions](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/utils/auth.py#L117)<!-- prefix: "def decide_r" -->

## Python service

| Method and path | Native | Contract at the pin |
| --- | --- | --- |
| G `/ready` | ✓ | 758. Upstream empty 200/503 reflects startup, pause, health and shutdown. Native JSON 200/503 reflects the worker-failure flag. |
| G `/health` | ✓ | 765. Upstream empty status, generation enabled by default; native always 200 with configured limits. |
| G `/health_generate` | ✓ | 766. Empty 200/503 on both. Upstream observes detokenizer or scheduler progress and updates health state. Native observes token publication, uses an idle probe or bounded FIFO, and keeps readiness separate. |
| G `/model_info`, `/get_model_info` | ✓ | 846–885. Alias; upstream reports live model identity and runtime fields. Native reports fixed loader/model capabilities and selected parser flags. |
| G `/server_info`, `/get_server_info` | — | 888–929. Alias; resolved startup config and launch command, per-DP live state, version and frontend. `/health` supplies two limits only. |
| G `/v1/models` | ✓ | 1998. Native one canonical card; upstream uses served name and can list loaded LoRAs. |
| G `/v1/models/{model:path}` | — | 2030. Base card lookup, including slash-containing IDs; structured `model_not_found` 404 for other IDs. |
| P/U `/generate` | ✓ P | 963–1025. Native text or IDs, scalar/batch, JSON or cumulative SSE. Upstream also accepts embeds and supported media, LoRA and additional controls. PUT and in-stream error termination differ. |
| P `/abort_request` | ✓ | 1727. Empty 200 and unknown-ID no-op. Native abort registry covers `/generate` only; upstream tokenizer/scheduler also cancels OpenAI generation. Native retains dispatched-prefix matching after an exact-ID gate. |
| P/U `/encode` | — | 1028. Native embedding inputs and outputs on embedding models. |
| P/U `/classify` | — | 1041. Reward/classification on compatible models, using embedding-shaped inputs. |
| P `/parse_function_call` | — | 1739. Standalone text/tool parser returns `normal_text` and calls. |
| P `/separate_reasoning` | — | 1779. Standalone reasoning/text split, with optional blocks. |
| P `/v1/completions` | ✓ | 1863. Scalar and batched text/IDs, JSON/SSE, echo and logprobs. Upstream supports multiple choices with `n`; native allows only `n=1` and serves a private benchmark token-ID format on this path. |
| P `/v1/chat/completions` | ✓ | 1873. Native Qwen templating, JSON/SSE, optional reasoning/tool parsers, inline image/video data URLs; narrower options, parser and media handling. |
| P `/v1/tokenize`, `/tokenize` | ✓ | 1911. Text or chat to token IDs/counts; native errors, option and queue limits apply. |
| P `/v1/detokenize`, `/detokenize` | ✓ | 1929. IDs to text; native errors and queue limits apply. |
| P `/v1/embeddings` | — | 1885. OpenAI embedding response; generation launch rejects embedding requests. |
| P `/v1/classify` | — | 1899. Labels/scores from a classification-model handler. |
| P/U `/v1/rerank` | — | 2106. Query/document ranking via supported cross-encoder or generative scoring. |
| P `/v1/score` | — | 2055. Candidate logprob or classification-logit scoring. |
| P `/v1/decisions` | — | 2063. Typed choice, score and yes/no via single-token labels on compatible generation models. |
| P `/v1/systemone` | — | 2186. System One prompt and decision layout over candidate scoring. Validation returns 422, an exception to the app's usual 400. |
| P `/v1/audio/transcriptions` | — | 1947. Multipart audio; JSON/text/verbose JSON; needs transcription adapter. |
| W `/v1/realtime` | — | 1987. Transcription sessions only; requires a supported audio model. |
| P `/v1/responses` | — | 2071. JSON/SSE create, with optional background processing and tool support. |
| G `/v1/responses/{response_id}` | — | 2090. Retrieve stored response by ID. |
| P `/v1/responses/{response_id}/cancel` | — | 2098. Cancel a background response. All three paths register even if handler initialization fails at 402–424. |
| G/H `/` | — | 2121–2133. JSON greeting; `SGLANG_OLLAMA_ROOT_ROUTE` replaces the path and greeting. |
| P `/api/chat` | — | 2136. Ollama chat. |
| P `/api/generate` | — | 2142. Ollama text generation. |
| G `/api/tags` | — | 2150. Ollama model list. |
| P `/api/show` | — | 2156. Ollama model metadata. Each Ollama path has its own `SGLANG_OLLAMA_*_ROUTE` override. |
| P `/v1/messages` | — | 2165. Anthropic Messages via chat adapter. |
| P `/v1/messages/count_tokens` | — | 2175. Anthropic prompt token count. |
| G `/ping` | — | 2195. SageMaker empty health status; intentional pause can remain healthy. |
| P `/invocations` | — | 2221. SageMaker dispatch order `messages`, `prompt`, `input`. |
| P `/vertex_generate` | — | 2256. Vertex instances/parameters wrapper for generation; `AIP_PREDICT_ROUTE` overrides path. |

See native [generation](../native-generate.md), [abort](../native-abort.md), [text](../text-serving.md), [chat](../chat-serving.md), [tokenization](../native-tokenization.md), [metadata](../native-model-info.md), and [health](../http-server.md) for field and wire limits. The upstream embedding guard applies to embedding-shaped requests on a generation launch. Support for the scoring, decision and rerank variants on this checkpoint remains unmeasured. [Model guard](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_manager.py#L1515)<!-- prefix: "if isinstanc" --> · [OpenAI docs](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/docs/docs/basic_usage/openai_api_completions.mdx#L32)<!-- prefix: "## Chat Comp" -->

## Python management and optional surfaces

| Method and path | Native | Operation and gate |
| --- | --- | --- |
| G/P `/flush_cache` | — | 1060. Optional admin. Clears cache pools, grammar state, metrics and media preprocessing cache when eligible. Positive `timeout` permits one pending flush. Native P `/reset_prefix_cache` drops published prefixes, preserves counters and reserved storage, responds in JSON, and refuses busy requests with 409. |
| G/P `/start_profile` | ✓ P | 1240. Admin; upstream optional `ProfileReq` controls scheduler profiler. Native requires exact step-capture options and prefix reuse. |
| G/P `/stop_profile` | ✓ P | 1250. Admin; upstream stops profiler. Native returns captured step JSON and requires prefix reuse. |
| ASGI `/metrics` | ✓ G | 321. Upstream enables a Prometheus ASGI mount for the `/metrics` prefix, including slash and suffix forms. The mount has no method filter. The locally cached Prometheus client 0.26.0 exporter likewise ignores method, while upstream leaves the dependency version open. Native accepts the exact GET path with prefix reuse and emits different series and counters. |
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

The auth map marks abort, flush, corpus listing, HiCache clear, trace level, GC freeze, remote transfer-info aliases and elastic status as optional admin. It requires admin auth for both profiler routes, internal-state edits, dumper, corpus writes, HiCache storage status and mutation, expert recording, weight and memory controls, slowdown, PD role switch, LoRA management, request logging, pause/resume and elastic scale. Server metadata, HiCache layout, loads and session controls use the regular key policy. [Auth map](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/utils/auth_levels.json#L3)<!-- prefix: "\"/abort_requ" -->

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

Paths below are relative to repository `artifacts/`. Earlier native B200 runs used the target checkpoint. The JSON Schema receipt records 26 successful HTTP assertions, including eight expected pre-admission errors with no scheduler rows. The prompt-logprobs acceptance receipt for selected revision `7eea6d91e4e1e1d0d221c2c7c1fb61f552ba40c1` records 34 HTTP responses and 41 frames in the candidate case, plus an exact warm/cold match across 9,755 compared scores. Its `upstream.json` is source-based evidence.

- `current-native-json-schema/evidence/checkpoint-1/{device.csv,http.log,http.exit,http/command.json}`
- `current-native-prompt-logprobs-accept/evidence/{identity.log,upstream.json,checkpoint-4/http.log}`

Earlier host HTTP fixtures exercised generation batches, abort, all tokenization aliases, text prompt forms and health. The pinned tokenizer comparison records 300 success and 90 rejection comparisons. The model-info packet matrix records ten matches and eight differences per configuration. The model-list receipt tests the writer.

- `current-batch-generate-accept/tests/evidence/latest-native.log`
- `current-native-abort-accept/logs/test-published-final.log`
- `current-tokenization-accept/proof/{integration-http,reference-probe}.log`
- `current-completions-prompt-forms/host/frozen-wire.log`
- `current-health-generate/header-layout/cpu/health.log`
- `current-model-info-accept/proof/matrix-summary.log`
- `resume-models-api/evidence/models-response-check.log`

Exact-base socket parity, pinned upstream live behavior, model and feature gates on the target checkpoint, and unlisted request variants remain unverified. The missing routes are static dispatch findings. Main can assign the new flush route and its focused HTTP verification as one change.
