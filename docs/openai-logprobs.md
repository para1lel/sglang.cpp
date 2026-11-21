# OpenAI generated-token logprobs

Both `/v1/completions` and `/v1/chat/completions` can return the score of each committed output token and a bounded list of alternatives. Scored calls use the same full-vocabulary target log-softmax as [native `/generate`](native-generate.md#generated-token-logprobs).

## Request requirements

Use greedy decoding and disable server MTP. Set `temperature: 0` for greedy decoding. `top_k: 1` also stays greedy at positive temperature. Active sampling, output constraints, decoded media, and active chat reasoning or tool output parsers return HTTP 400 before SSE. The completion route accepts `echo: false` and rejects `echo: true`. Scores cover generated tokens only. [Score options](../apps/server/text_request.cpp#L318)<!-- prefix: "int openai_l" -->

Generated logprobs use the scores after [output penalties](output-penalties.md) and logit bias.

## Logit bias

In OpenAI completions and native `/generate`, `logit_bias` maps decimal token ID strings in `[0, 248320)` to finite offsets in `[-1e30, 1e30]`. When different spellings identify the same token, the last entry supplies its offset. Omission, null, an empty object, and zero offsets leave scores unchanged. Active offsets apply to FP32 scores before greedy or sampled selection and respect output constraints. Active bias uses target-only decoding for that request, including on an MTP server. Generated logprobs use the adjusted scores and follow the request requirements above. [Bias parser](../apps/server/text_request.cpp#L133)<!-- prefix: "void set_log" -->

## Text completions

Set integer `logprobs` from `0` through `20` on any of the [four prompt forms](text-serving.md#send-a-request). `0` returns selected token scores and an empty `top_logprobs` array. Values `1`–`20` add a token-string-to-score map for each selected token. When ranked IDs map to the same string, the last ranked entry supplies that key's score. Omission or null leaves the response field absent. The [frozen benchmark request](../benchmarks/streaming/README.md#wire-contract) rejects `logprobs`.

JSON includes `logprobs` inside each choice, with `tokens`, `token_logprobs`, `top_logprobs`, and `text_offset`. Offsets are `-1`. Score token strings preserve valid UTF-8 and map invalid byte fragments through Latin-1. A zero-budget choice has empty arrays. In SSE, collect new scores by choice index. [Completion score writer](../apps/server/text_completion.cpp#L60)<!-- prefix: "void append_" -->

## Chat completions

Set `logprobs: true` and optionally an integer `top_logprobs` from `0` through `20`. Omit the top count or set it to null for zero alternatives. Omission or `false` for `logprobs` leaves scores absent even with a valid top count. An explicit `logprobs: null` returns HTTP 400.

The choice's `logprobs.content` contains one object per committed token. Each object has `token`, `bytes`, `logprob`, and a `top_logprobs` array. The `token` string uses replacement characters for invalid UTF-8. The `bytes` array contains the original unsigned bytes. A zero-budget JSON choice has empty `content`. The SSE role choice carries `logprobs: null`. [Chat score writer](../apps/server/text_completion.cpp#L123)<!-- prefix: "void append_" -->

## Streaming and stops

SSE delivers each score once. A scored event can have empty visible text while the writer holds incomplete UTF-8 or a possible stop prefix. In a scored stream, choices with no new scores, including terminal choices, carry `logprobs: null`. Usage events carry no choices. A committed token that matches a text stop or stop ID retains its score and contributes to usage. The visible text follows the ordinary stop trimming rules. [Delta projection](../apps/server/text_completion.cpp#L433)<!-- prefix: "bool TextRes" --> · [Score alignment](../apps/server/text_completion.cpp#L531)<!-- prefix: "bool consume" -->

## Verification

With server MTP disabled, compare JSON with the concatenated SSE scores for the same greedy requests. Check selected-only and top-20 responses, batch choices by index, zero-budget responses, stop IDs, cross-token text stops, and split UTF-8. Repeat on eager and graph decode. Require HTTP 400 before SSE for active sampling, MTP, and output constraints. Run the [native CUDA and CPU score checks](native-generate.md#verification) for the shared reducer and native score delivery.

Build the [CPU fixture](../tests/text/README.md#native-generate-on-cpu), then run the OpenAI wire suite from the repository root in the managed Pixi environment.

```bash
out="$PWD/artifacts/native-generate-cpu"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
CUDA_VISIBLE_DEVICES='' python3 -B tests/text/test_openai_generated_logprobs_cpu.py \
    --server "$out/server" --checkpoint "$checkpoint" --output "$out/openai-logprob-evidence"
```

Require exit status zero. The suite compares JSON with SSE across all four completion prompt forms and chat, checks score arrays, byte projection, and stop behavior, and exercises opt-out, frozen benchmark, and MTP guards.
