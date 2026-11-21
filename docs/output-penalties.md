# Output penalties

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Request options | Validate the wire values and populate `DecoderOptions`. | [HTTP validation](../apps/server/text_request.cpp#L175)<!-- prefix: "double sampl" --> · [Native defaults](../apps/server/text_request.cpp#L754)<!-- prefix: "const auto* " --> · [C++ options](../include/sglang/decoder.h#L49)<!-- prefix: "double frequ" --> |
| Decoder | Keep each item's history and choose its execution path. | [Batch depth](../src/decoder_batch.cpp#L508)<!-- prefix: "if (options." --> · [Direct execution](../src/decoder.cpp#L561)<!-- prefix: "const Reques" --> |
| CUDA score adjustment | Update history and rank the adjusted scores. | [History update](../src/decoder_sampling.cu#L414)<!-- prefix: "__device__ f" --> · [Repetition selection](../src/decoder_sampling.cu#L871)<!-- prefix: "cudaError_t " --> |

## Runtime path

The first output uses an empty history. Before selecting each subsequent output, CUDA adds one frequency increment for the previous committed ID and marks it as seen. Only the current item's generated IDs contribute. For each vocabulary ID, CUDA subtracts its accumulated frequency penalty and, if seen, one presence penalty from its FP32 score. Positive values discourage reuse. Negative values encourage it.

For a seen ID, CUDA then multiplies a negative score by `repetition_penalty` or divides a nonnegative score by it. It applies this scaling once per score regardless of how often the ID appeared. Values above one discourage reuse, and values below one encourage it. Prompt IDs, including reused prefixes, do not seed this history.

Repetition scaling uses FP64 and preserves a zero score. Ordinary rows rank the adjusted FP32 scores and use FP64 score differences for normalization. Extreme rows keep a separate exponent, a low-order arithmetic residual, and small bias differences alongside large values. CUDA ranks these scores before subtracting the row maximum for probability calculations. Greedy selection breaks equal represented scores by token ID.

CUDA applies any output constraint mask and then adds [logit bias](openai-logprobs.md#logit-bias). For extreme rows, ranking and normalization subtract the repetition-adjusted bases before applying the bias difference. This preserves a small bias between equal large scores. Direct C++ `top_logits` reports ranked scores before constraint masking. For extreme rows with a finite maximum, the readback subtracts that maximum and limits the reported difference to the finite FP32 range. Reported logprobs also use a finite FP32 lower bound. Selection and normalization use the scores before either reporting limit. [Direct score readback](../src/decoder.cpp#L314)<!-- prefix: "if (adjusted" -->

An item with an active penalty uses target-only decoding. Eligible peers in the same batch can use MTP. Batch adjustment runs after eager scoring or eligible K0 graph replay. Direct C++ generation with an active penalty uses eager decoding. A neutral repetition value leaves the ordinary path eligible. [Batch selection](../src/decoder_batch.cpp#L610)<!-- prefix: "auto* adjust" -->

## Constraints and failure behavior

`frequency_penalty` and `presence_penalty` each accept a finite number in `[-2, 2]` and default to zero. Both OpenAI routes return HTTP 400 for explicit null. A null field inside native `/generate` `sampling_params` uses zero. C++ callers set the corresponding `DecoderOptions` fields.

`repetition_penalty` accepts a finite number in `(0, 2]` and defaults to `1`. Native `/generate` accepts it inside broadcast or per-item `sampling_params`. Native and `/v1/chat/completions` treat explicit null as the default. `/v1/completions` returns HTTP 400 for explicit null. C++ callers set `DecoderOptions::repetition_penalty`.

## Minimum output

Use `sampling_params.min_new_tokens` on `/generate` or `min_tokens` on either OpenAI completion route. Supply a nonnegative integer up to the effective output budget. Omission uses zero. Native `null` uses zero. OpenAI routes reject `null`.

The minimum counts generated IDs. The sampler masks model EOS and explicit stop IDs before selection until the count reaches the minimum, including when `ignore_eos` disables token-ID termination. Text stops can still finish earlier. If the minimum equals the maximum, token stops remain masked throughout the output budget. A positive minimum uses target-only generation on a server configured for MTP. [Stop mask upload](../src/decoder.cpp#L154)<!-- prefix: "void Request" --> · [Generated count](../src/decoder.cpp#L224)<!-- prefix: "const auto* " --> · [CUDA mask](../src/decoder_sampling.cu#L407)<!-- prefix: "__device__ b" -->

## Verification

Run the [native and sampling route fixtures](../tests/text/README.md) against a CPU-backed server. Compare omitted and neutral values. Check null, the smallest positive FP64 factor, and out-of-range values on each route. Check native broadcast and per-item forwarding.

Use the maintained [score-check runner](../tests/text/run_logit_bias_check.sh#L4)<!-- prefix: "if (( $# != " --> on a B200 for eager and graph scoring. Compare the empty and repeated output histories with the full-vocabulary oracle, including its tie at the requested top-logprob cutoff. The [synthetic row checks](../tests/text/logit_bias_check.cpp#L443)<!-- prefix: "void verify_" --> cover zero, masking, ranking past FP32 range, a 75/25 split from `log(3)` bias on equal repeated logits, and bias just below and above a quotient residual. Run greedy and sampled generation with tiny factors. For mixed requests, check that only generated IDs from the same item affect its scores. With MTP configured, verify that an active penalty keeps its lane at K0 while an eligible peer drafts.
