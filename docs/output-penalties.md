# Output penalties

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Request options | Validate the wire values and populate `DecoderOptions`. | [HTTP validation](../apps/server/text_request.cpp#L168)<!-- prefix: "double sampl" --> · [Native defaults](../apps/server/text_request.cpp#L726)<!-- prefix: "const auto* " --> · [C++ options](../include/sglang/decoder.h#L49)<!-- prefix: "double frequ" --> |
| Decoder | Keep each item's history and choose its execution path. | [Batch depth](../src/decoder_batch.cpp#L506)<!-- prefix: "if (options." --> · [Direct execution](../src/decoder.cpp#L487)<!-- prefix: "const Reques" --> |
| CUDA score adjustment | Update history and prepare the FP32 selection row. | [History update](../src/decoder_sampling.cu#L402)<!-- prefix: "__global__ v" --> · [Adjustment order](../src/decoder_sampling.cu#L510)<!-- prefix: "cudaError_t " --> |

## Runtime path

The first output uses an empty history. Before selecting each subsequent output, CUDA adds one frequency increment for the previous committed ID and marks it as seen. Only the current item's generated IDs contribute. For each vocabulary ID, CUDA subtracts its accumulated frequency penalty and, if seen, one presence penalty from its FP32 score. Positive values discourage reuse. Negative values encourage it.

CUDA applies any output constraint mask and then adds [logit bias](openai-logprobs.md#logit-bias). Greedy selection, sampling, and supported generated logprobs use the resulting scores. Direct C++ `top_logits` reports the adjusted full-vocabulary row before constraint masking. [Direct score readback](../src/decoder.cpp#L243)<!-- prefix: "if (adjusted" -->

An item with either penalty nonzero uses target-only decoding. Eligible peers in the same batch can use MTP. Batch adjustment runs after eager scoring or eligible K0 graph replay. Direct C++ generation with an active penalty uses eager decoding. [Batch selection](../src/decoder_batch.cpp#L608)<!-- prefix: "auto* adjust" -->

## Constraints and failure behavior

`frequency_penalty` and `presence_penalty` each accept a finite number in `[-2, 2]` and default to zero. Both OpenAI routes return HTTP 400 for explicit null. A null field inside native `/generate` `sampling_params` uses zero. C++ callers set the corresponding `DecoderOptions` fields.

## Verification

Run the [native and sampling route fixtures](../tests/text/README.md) against a CPU-backed server. Compare omitted and zero values. Check null and out-of-range values on each route.

Use the maintained [score-check runner](../tests/text/run_logit_bias_check.sh#L4)<!-- prefix: "if (( $# != " --> on a B200 for eager and graph scoring. Compare the empty and repeated output histories with the full-vocabulary oracle, including its tie at the requested top-logprob cutoff. For mixed requests, check that only generated IDs from the same item affect its scores. With MTP configured, verify that an active penalty keeps its lane at K0 while an eligible peer drafts.
