# Ollama discovery

Ollama discovery supports `GET /`, `HEAD /`, `GET /api/tags`, and `POST /api/show`. The greeting is a JSON string. `HEAD /` returns the greeting headers and an empty body. Tags lists the served checkpoint, `Qwen3.8-27B-FP8`. The size, digest, and quantization fields follow the pinned SGLang placeholders.

For show, send `Content-Type: application/json` and a JSON object with a `model` string. The response uses the supplied name for metadata and reports the served checkpoint's context limit of 262,144 tokens. Different names, including an empty string, produce metadata for that same checkpoint. The `name` and `verbose` keys have no effect.

`SGLANG_OLLAMA_ROOT_ROUTE`, `SGLANG_OLLAMA_TAGS_ROUTE`, and `SGLANG_OLLAMA_SHOW_ROUTE` set replacement paths at startup. An explicit mapped path matches before a slash redirect. At a collision, an exact method from an earlier native route takes priority, including `POST /v1/score`. [Discovery dispatch](../../apps/server/ollama_discovery.cpp#L226)<!-- prefix: "bool OllamaD" -->
