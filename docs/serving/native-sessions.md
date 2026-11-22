# Native generation sessions

Use `GET` or `POST /open_session` with a JSON body. `capacity_of_str_len` is a required integer. The server returns a JSON string with the session ID. Supply `session_id` to choose the ID, or omit it to have the server generate one.

```json
{"capacity_of_str_len": 2048, "session_id": "geo"}
```

Send one [native generate](../native-generate.md) request with `session_params.id`. Use a request ID to refer to that turn later.

```json
{"text":"The capital of France is","rid":"france","session_params":{"id":"geo"},"sampling_params":{"max_new_tokens":2,"temperature":0}}
```

For a continuation, set `session_params.rid` to the ID of a completed turn. The server appends the new text or flat prompt IDs to that turn's prompt and output IDs. The response counts the full assembled prompt. Omit the parent ID to start a fresh turn in the same session.

```json
{"text":". The capital of Italy is","session_params":{"id":"geo","rid":"france"},"sampling_params":{"max_new_tokens":2,"temperature":0}}
```

Set `drop_previous_output: true` inside `session_params` to take only the parent's prompt. A positive `offset` keeps at most that many preceding IDs. A negative offset removes that many IDs from the end. Zero appends to the selected history. Branches leave the parent available for further turns.

Open a regular session and set `stream: true` on `/generate` for cumulative SSE. A successful turn becomes a usable parent when generation completes. Sessions support single text prompts or flat prompt IDs with `n: 1`. The server rejects `replace: true`, `streaming: true` session opens, media, and batched turns.

Send `GET` or `POST /close_session` with `{"session_id":"geo"}`. The server returns an empty HTTP 200, including for unknown IDs. Closing discards stored turns. An active request can finish on its current connection. A cancelled or aborted turn leaves no history. An optional `timeout` in seconds expires the session based on its most recent admission.

The server supports 256 open sessions, 256 turns per session, and a shared budget of 4,194,304 retained token slots. The budget includes allocated output capacity. `capacity_of_str_len` is accepted for protocol compatibility. Session history is held in host memory. [Prefix reuse](../prefix-cache.md) can reuse a matching checkpoint when enabled.
