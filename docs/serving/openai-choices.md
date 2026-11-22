# OpenAI completion choices

`POST /v1/completions` and `POST /v1/chat/completions` accept integer `n` from 1 to 4. The default is 1. Larger values repeat one prompt through the same decoder scheduler. Text accepts one string or flat token-ID sequence. Chat accepts one text conversation and requires an explicit `max_tokens` or `max_completion_tokens` budget. Prompt arrays, decoded chat media and advertised chat tools require `n: 1`.

The repeated prompt lengths and output budgets together must fit 262,144 tokens. Each choice has its own output, constraint, stop and score state. Sampling uses independent random streams. A supplied seed gives reproducible sampling per index and keeps the scalar seed for choice zero. The existing [text options](../text-serving.md#send-a-request) and [chat options](../chat-serving.md#request-options) apply.

JSON choices arrive in index order. SSE choices can interleave. Join text or chat deltas within each zero-based index and read its terminal finish. Chat sends an assistant role for each index. Each index keeps its own response ID. Text choice chunks carry `usage: null`. With `stream_options.include_usage: true`, the last choice's ID appears on the final usage chunk with `choices: []`. A successful stream ends with `[DONE]`.

Usage counts the input and its cached tokens once. Completion usage sums committed output IDs across choices. The underlying [stream handling](../http-server.md#completion-streams) cancels unfinished choices when the connection closes.
