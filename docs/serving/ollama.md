# Ollama discovery, generation, and chat

Ollama discovery supports `GET /`, `HEAD /`, `GET /api/tags`, and `POST /api/show`. The greeting is a JSON string. `HEAD /` returns the greeting headers and an empty body. Tags lists the served checkpoint, `Qwen3.8-27B-FP8`. The size, digest, and quantization fields follow the pinned SGLang placeholders.

For show, send `Content-Type: application/json` and a JSON object with a `model` string. The response uses the supplied name for metadata and reports the served checkpoint's context limit of 262,144 tokens. Different names, including an empty string, produce metadata for that same checkpoint. The `name` and `verbose` keys have no effect.

`SGLANG_OLLAMA_ROOT_ROUTE`, `SGLANG_OLLAMA_TAGS_ROUTE`, `SGLANG_OLLAMA_SHOW_ROUTE`, `SGLANG_OLLAMA_GENERATE_ROUTE`, and `SGLANG_OLLAMA_CHAT_ROUTE` set replacement paths at startup. An explicit mapped path matches before a slash redirect. An exact method from an earlier native route takes priority, including `POST /v1/score`. Within Ollama, chat takes precedence over generate and show for a shared POST path. [Ollama dispatch](../../apps/server/ollama_discovery.cpp#L248)<!-- prefix: "OllamaDiscov" -->

Overrides provide path dispatch. Specialized upstream validators can give different
statuses or error bodies. For example, a wrong-typed `model` under `/v1/responses/ollama-test`
gives HTTP 400 with different error envelopes, while `/v1/systemone` gives upstream HTTP 422
and native HTTP 400.

`POST /api/generate` accepts `model` and `prompt` strings. A nonempty `system` string precedes the prompt and two newlines separate them. Generation uses the served checkpoint and the native decoder queue. Streaming defaults to true and returns `application/x-ndjson`. Set `stream` to `false` for one JSON response with counts and total duration when the decoder runs. A blank combined prompt returns an empty completion. Streams end with an empty `response`, `done: true`, and `done_reason: "stop"`.

Both endpoints apply `temperature`, `top_p`, `top_k`, `num_predict`, `stop`, `presence_penalty`, and `frequency_penalty`. `num_predict` defaults to 2,048. The pinned sampling path rejects a supplied `options.seed` when it runs. Generate validates `images`, `format`, and `context` and ignores their values.

## Chat

`POST /api/chat` accepts a `model` string and an ordered `messages` array. Send a JSON content type such as `application/json`. Each message needs string `role` and `content` fields. The Qwen checkpoint template consumes the conversation, including prior assistant and tool text, and appends the assistant generation prompt. Include a user query and place any system message first. The response reports the served checkpoint's name.

The pinned handler accepts `images` as null or an array of strings and validates `format`, `keep_alive`, and `think`. Their values are ignored. Extra `tools` and `tool_calls` fields are ignored. Chat consumes text only and applies the checkpoint template's default thinking settings.

Set `stream` to `false` for a single JSON response with the assistant message, prompt and completion token counts, and total duration. By default the endpoint returns NDJSON. Each nonterminal record carries an assistant text delta. The final record has empty content, `done: true`, and `done_reason: "stop"`. It carries no counts and discards any text first published on the terminal decoder step. Schema failures return 400. Template failures return 500 before streaming starts. A sampling failure can leave a started stream without a terminal record. [Chat dispatch](../../apps/server/native_generate.cpp#L449)<!-- prefix: "std::string " -->

With default MTP depth 4, a terminal step can publish several tokens. The streamed text
can omit a multi-token suffix that appears in the single JSON response.
