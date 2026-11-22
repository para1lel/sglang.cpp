# Ollama discovery and generation

Ollama discovery supports `GET /`, `HEAD /`, `GET /api/tags`, and `POST /api/show`. The greeting is a JSON string. `HEAD /` returns the greeting headers and an empty body. Tags lists the served checkpoint, `Qwen3.8-27B-FP8`. The size, digest, and quantization fields follow the pinned SGLang placeholders.

For show, send `Content-Type: application/json` and a JSON object with a `model` string. The response uses the supplied name for metadata and reports the served checkpoint's context limit of 262,144 tokens. Different names, including an empty string, produce metadata for that same checkpoint. The `name` and `verbose` keys have no effect.

`SGLANG_OLLAMA_ROOT_ROUTE`, `SGLANG_OLLAMA_TAGS_ROUTE`, `SGLANG_OLLAMA_SHOW_ROUTE`, and `SGLANG_OLLAMA_GENERATE_ROUTE` set replacement paths at startup. An explicit mapped path matches before a slash redirect. At a collision, an exact method from an earlier native route takes priority, including `POST /v1/score`. [Ollama dispatch](../../apps/server/ollama_discovery.cpp#L240)<!-- prefix: "OllamaDiscov" -->

`POST /api/generate` accepts `model` and `prompt` strings. A nonempty `system` string precedes the prompt and two newlines separate them. Generation uses the served checkpoint and the native decoder queue. Streaming defaults to true and returns `application/x-ndjson`. Set `stream` to `false` for one JSON response with counts and total duration when the decoder runs. A blank combined prompt returns an empty completion. Streams end with an empty `response`, `done: true`, and `done_reason: "stop"`.

Sampling options are `temperature`, `top_p`, `top_k`, `num_predict`, `stop`, `presence_penalty`, and `frequency_penalty`. The pinned handler rejects a supplied `options.seed` for a nonblank prompt. It validates fields such as `images`, `format`, and `context` but does not apply them during generation. This endpoint handles plain-text generation. Ollama chat, vision inputs, and structured-output enforcement are outside its contract.
