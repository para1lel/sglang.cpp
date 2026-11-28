# Native text completions

`POST /v1/completions` serves `Qwen3.8-27B-FP8` with text or token-ID prompts and the frozen `benchmark` model with [token-ID SSE](../benchmarks/streaming/README.md#wire-contract). See [native chat](chat-serving.md), [`POST /generate`](native-generate.md), and [CPU tokenization](native-tokenization.md).

## Send a request

Follow the [server launch procedure](../README.md#serve-token-ids-over-http).

```bash
curl --fail-with-body http://127.0.0.1:17384/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen3.8-27B-FP8","prompt":"一、北京\n二、上海\n三、","max_tokens":8}'
```

| `prompt` example | Input |
| --- | --- |
| `"Hello"` | One UTF-8 string |
| `[220, 16]` | One sequence of token IDs |
| `["Hello", "World"]` | Two UTF-8 strings |
| `[[220, 16], [220, 17]]` | Two sequences of token IDs |

Each string must contain 1 to 1,048,576 UTF-8 bytes. Each ID sequence must be nonempty and contain integers in `[0, 248320)`. IDs go directly to the decoder. Each prompt and its output budget must fit 262,144 tokens. A batch contains 1 to 32 prompts of one form and shares request options across all items. The sum of output budgets must be at most 262,144 tokens. Invalid prompts and options return HTTP 400 before decoder admission. [Prompt parsing](../apps/server/text_request.cpp#L464)<!-- prefix: "TextCompleti" --> · [Resident memory admission](batch-graphs.md#resident-memory-admission)

The [HTTP reader](http-server.md#http-admission-and-shutdown) limits headers and bodies. Text and chat reject malformed JSON, duplicate decoded keys, more than 32 nested containers, and floating-point range errors. Oversized integer `top_k` is validated before clamping. The frozen `benchmark` parser retains signed-zero underflow and fixed fields. [Numeric dispatch](../apps/server/completions.cpp#L175)<!-- prefix: "json::Value " --> · [Benchmark parser](../apps/server/completions.cpp#L28)<!-- prefix: "CompletionRe" -->

| Field | Behavior |
| --- | --- |
| `max_tokens` | Nonnegative integer output budget. Omit it to use 16. Null is invalid. For an unconstrained request, zero returns no output IDs. |
| `min_tokens` | [Minimum output](output-penalties.md#minimum-output), default zero. |
| `n` | [Multiple choices](serving/openai-choices.md), integer `1`–`4` for one prompt. |
| `echo` | Boolean, default `false`. Set it to `true` to prepend the prompt to each choice's text. |
| `stream` | Boolean, default `false`. Set it to `true` for SSE. |
| `stream_options` | For SSE, set `{"include_usage":true}` to receive a separate usage event. |
| `ignore_eos` | Boolean, default `false`. Set it to `true` to disable model stop IDs `248044` and `248046` and all explicit stop IDs. |
| `stop_token_ids` | Array of explicit stop IDs in `[0, 248320)`. The server validates them even when `ignore_eos` is `true`. |
| `stop` | One nonempty string or an array of nonempty strings. Omission, null, and an empty array disable text stops. |
| `return_token_ids` | Boolean, default `false`. Set it to `true` to include diagnostic `prompt_token_ids` and `token_ids` for each choice. |
| `logprobs` | Integer `0`–`20`. Requests [token scores](serving/openai-logprobs.md#text-completions). |
| `logit_bias` | Optional [per-token offsets](serving/openai-logprobs.md#logit-bias). |
| `ebnf` | Optional string that restricts output to [one literal choice](#finite-choice-constraints). |
| `regex` | Optional [whole-output regular expression](output-constraints.md#regular-expressions). |
| `response_format` | Optional [bounded JSON Schema](output-constraints.md#json-schema). |
| `cache_salt` | Optional [prefix cache key](prefix-cache.md#runtime-path) component. |

Echo copies string prompts and decodes ID prompts with special tokens hidden. Each SSE index echoes once, in its first choice. Stops apply to generated output. With `max_tokens: 0`, echo returns the prompt with zero completion tokens.

The text route accepts [output penalties](output-penalties.md), including `repetition_penalty`, and rejects `max_completion_tokens` and unknown options. [Request options](../apps/server/text_request.cpp#L453)<!-- prefix: "bool is_text" -->

Text, chat, and `/generate` share the immutable tokenizer. The server verifies checkpoint `tokenizer.json` size and SHA-256 on first use. Missing, unreadable, or mismatched assets return 500 before output. Later requests retry. Benchmark IDs need no tokenizer.

## Serial prefill

At K0, the generator and `Decoder::generate` use serial prefill. With MTP they use [batch scheduling](batch-graphs.md#runtime-path), as HTTP does at every capacity. Ordinal 0 is the default. `--device` or the C++ device argument overrides it. The serial row cap is 1,024 on B200 and 128 on the fallback backend, including media.

Tiles and any shorter tail run in order. Capacity fits the prompt, step cap, and workspace. On allocation OOM or an unsupported pool, retry large scratch at 128 rows, then process rows singly if normal scratch is unavailable. Other errors propagate. Actual tile size determines [MXFP8 flash eligibility](flash-prefill.md#constraints-and-failure-behavior). [Tile loop](../src/decoder.cpp#L658)<!-- prefix: "bool Decoder" -->

### Serial prefill verification

Reserve a B200. For serial K0, run the generator with `--mtp-steps 0`. Check workspace capacity, full BF16 scores, and output IDs at the tile boundary and on a ragged tail. Inject recoverable allocation failures for large and normal scratch. Compare each retry with the same tile schedule and check a later request on the same decoder. Follow the [native prefill runbook](../tests/prefill/README.md) for operator checks.

## Request progress

Serial generation and batch lanes track the next target position and the number of emitted output IDs separately. The position advances with restored prefixes, prompt inputs, and consumed decode rows. Scoring the final prompt row emits the first output without processing another input. Each emitted ID uses one unit of the output budget. For a positive output budget, the logical target KV limit is `prompt length + output budget - 1` positions. [Progress state](../include/sglang/decoder_internal.h#L108)<!-- prefix: "struct Reque" --> · [First score](../src/decoder.cpp#L690)<!-- prefix: "if (progress" -->

If `is_cancelled` returns true at entry, direct C++ generation finishes with `cancelled` before prefill. To deliver a selected ID, it appends the ID, calls `on_token`, then records requested top logits. If `on_token` returns false, the result retains that ID and its requested logits and finishes with `cancelled`. Otherwise, a constraint finish or stop match takes precedence over length. In direct MTP, the callback runs before the verifier commits the [consumed target state](mtp.md#runtime-path). [Result update](../src/decoder.cpp#L468)<!-- prefix: "bool SerialO" --> · [Finish decision](../src/decoder.cpp#L329)<!-- prefix: "std::optiona" -->

## Sampling

Native HTTP uses these controls for JSON and SSE. C++ and the generator use the same controls and default to greedy selection. [HTTP validation](../apps/server/text_request.cpp#L199)<!-- prefix: "void set_sam" --> · [C++ options](../include/sglang/decoder.h#L58)<!-- prefix: "struct Decod" --> · [C++ validation](../src/decoder.cpp#L62)<!-- prefix: "void validat" -->

| Field | Valid values |
| --- | --- |
| `temperature` | Finite nonnegative number. Values below `1e-6` select greedy decoding. |
| `top_p` | Number in `(0, 1]`. A value of `1` disables the nucleus cutoff. |
| `top_k` | Integer `-1` for the full vocabulary, or positive. Larger values clamp to 248,320. A value of `1` selects greedy decoding. |
| `min_p` | Number in `[0, 1]`. The minimum weight is `min_p` times the maximum weight in the selection row. |
| `seed` | Signed 64-bit integer. Omission or null leaves the seed unset. |

Raw text defaults to `temperature: 1`, `top_p: 1`, `top_k: -1`, and `min_p: 0`. Null numeric controls except `seed` return 400. Chat selects [server defaults](chat-serving.md#sampling-defaults-and-output-budget). [Route defaults](../apps/server/text_request.cpp#L339)<!-- prefix: "TextRequest " -->

CUDA orders scores descending, breaking ties by lowest token ID. It exponentiates temperature-scaled differences. The nucleus includes the first token whose cumulative weight exceeds `top_p` times total row weight. Draw from the intersection of nucleus, top-k and weights meeting `min_p`. The resolved seed and zero-based output index determine each draw. Identical rows and options preserve draws across slots. Options are validated even for greedy decoding.

Unseeded sampling resolves one seed with `std::random_device`. Text can use [sampled MTP](mtp.md#sampled-mtp), including constraints. K0 uses eager or [eligible graph logits](batch-graphs.md#runtime-path). The frozen benchmark client stays greedy with `temperature: 0`, `seed: 0`, and `ignore_eos: true`.

Negative infinity masks a token. NaN, positive infinity, or no finite logits yields `DecoderSamplingError` in direct unconstrained generation and `sampling_failed` in batch mode. The failed lane retires and peers continue. See [HTTP failure mapping](http-server.md#completion-streams). [Seed resolution](../src/decoder.cpp#L120)<!-- prefix: "SamplingPara" --> · [Cutoff and draw](../src/decoder_sampling.cu#L195)<!-- prefix: "__global__ v" --> · [Batch dispatch](../apps/server/batch_service.cpp#L861)<!-- prefix: "DecoderResul" --> · [Sample readback](../src/decoder.cpp#L376)<!-- prefix: "int read_tok" --> · [Batch retirement](../src/decoder_batch.cpp#L813)<!-- prefix: "if (result.s" --> · [Benchmark request](../benchmarks/streaming/client.py#L40)<!-- prefix: "def request_" -->

## Finite-choice constraints

Use top-level `ebnf` with [one rule](output-constraints.md#native-ebnf), such as `root ::= "yes" | "no"`, on text or chat. Allow up to 32 nonempty valid UTF-8 literals with JSON escapes, at most 256 decoded bytes each and 2,048 bytes total. Grammar text after request decoding is limited to 16,384 bytes. Each state allows 4,094 ordinary choices plus two reserved EOS entries. HTTP 400 covers malformed or non-string grammar, extra rules, concatenation, repetition, no tokenizable completion within budget, another output constraint, explicit `stop_token_ids` including `[]`, or `ignore_eos: true`. The benchmark route accepts only its fixed fields. [Runnable literal request](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/text-serving.md#finite-choice-constraints).

The request cursor tracks checkpoint token bytes across grammar nodes and split UTF-8 scalars. [Shared selection and finish](output-constraints.md#selection-and-finish) apply to JSON and SSE, including acceptance at the cap and a possible later EOS after a shared prefix. Each winner readback is four bytes. Serial `top_logits` expose full-vocabulary scores. [constrained MTP](mtp.md#constrained-mtp) · [BF16 selection](../src/decoder_constraint.cpp#L28)<!-- prefix: "void select_" --> · [FP32 selection](../src/decoder.cpp#L210)<!-- prefix: "void Adjuste" --> · [Cursor update](../src/tokenizer/constraint.cpp#L162)<!-- prefix: "bool Constra" --> · [HTTP finish](../apps/server/text_completion.cpp#L210)<!-- prefix: "const char* " -->

In C++, include `sglang/decoder_constraint.h`. Link `sglang_tokenizer` to compile a `TokenConstraint` with the checkpoint tokenizer. Assign the shared constraint to `DecoderOptions::constraint`. To use MTP, construct the decoder with draft weights. Set `DecoderOptions::mtp_steps` to a value from 1 through 4. Use only the two model EOS IDs if you supply stop IDs. Successful C++ generation reports `DecoderFinishReason::constraint`. [C++ validation](../src/decoder.cpp#L32)<!-- prefix: "void validat" -->

An invalid constrained winner causes serial `DecoderConstraintError`. The batch marks `constraint_failed` and retires that lane. HTTP uses the same failure handling as for sampling.

## Read the response

JSON choices follow prompt order and include `index`, `text`, and `finish_reason`. SSE sends one choice per data event and can interleave indices. Join text deltas by zero-based `index`. Each index has one terminal choice. Requested usage follows all terminals with `choices: []`, then the [stream end marker](http-server.md#completion-streams).

Usage counts prompt and emitted IDs. `prompt_tokens_details.cached_tokens` sums [restored prompt IDs](prefix-cache.md#runtime-path). A stop ID counts in usage and diagnostics, hides its text, and takes precedence over `length`. See [finite-choice](#finite-choice-constraints) and [other constraint finishes](output-constraints.md).

Choice text omits special and unmapped IDs. The writer holds incomplete UTF-8 and replaces malformed or flushed incomplete bytes with U+FFFD. Diagnostics can carry an ID with empty text or final text with zero IDs. The first diagnostic SSE choice includes `prompt_token_ids`, including terminal zero-budget choices.

Text stops match generated output [before commit](native-generate.md#text-stops), also with MTP or `ignore_eos: true`. They conflict with constraints. The writer holds possible stop prefixes, hides the match and suffix, and releases unmatched prefixes at finish. Count through the matching token and report `stop`. Other choices continue. [Batch response](../apps/server/text_completion.cpp#L816)<!-- prefix: "void complet" --> · [Response writer](../apps/server/text_completion.cpp#L579)<!-- prefix: "bool TextRes" --> · [Stop preview](../apps/server/text_completion.cpp#L301)<!-- prefix: "std::functio" --> · [Batch token delivery](../apps/server/text_completion.cpp#L677)<!-- prefix: "bool consume" -->

## Run with bundled libraries

The Linux x86-64 Release test bundles `libcudart.so.13`, `libstdc++.so.6`, and `libgcc_s.so.1` with server and generator. Supply host glibc, a compatible NVIDIA driver, and the checkpoint. Use the host loader to bypass build `DT_RPATH`; both programs must resolve and show help without LLVM or Pixi paths. [Loader procedure](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/text-serving.md#run-with-bundled-libraries).

## Tokenizer assets

The native tokenizer uses the Qwen checkpoint's literal added tokens, vocabulary, and byte-pair encoding merge ranks. It matches literal added tokens first. For ordinary spans, it applies NFC, the checkpoint's pretokenizer split, and ByteLevel BPE. It adds zero BOS, EOS, and template tokens.

| Asset | Pinned source |
| --- | --- |
| `tokenizer.json` | Qwen `Qwen/Qwen3.8-27B-FP8` revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`, 12,809,320 bytes, SHA-256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`. [Checkpoint authentication](research/checkpoint.md#checkpoint-evidence) |
| Tokenizer license | Apache-2.0 from the pinned checkpoint. [License and asset links](../tools/unicode/README.md) |
| NFC tables | Unicode Character Database 9.0.0, canonical decomposition, combining classes, and composition exclusions. |
| Pretokenizer tables | Unicode Character Database 16.0.0, general categories, `White_Space`, and simple case folds. |

The [Unicode regeneration procedure](../tools/unicode/README.md) pins both archives and identifies the applicable Unicode notice. The build compiles the checked-in tables.

## Verification

Run the CPU tokenizer check and HTTP fixture under the managed Pixi environment. Build `text-tokenizer-check` with `--text-tokenizer-check=y` and run it with the pinned `tokenizer.json` and `tests/text/cases.json`. Require exit zero and exact Hugging Face Tokenizers 0.23.3 fixture IDs, including budget, normalization, pretokenization and UTF-8 boundaries. [Commands](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/text-serving.md#verification).

On a reserved B200, run `python -B tests/text/test_native_text.py --port 17384 --width 1 --tokenizer /data0/shared/Qwen3.8-27B-FP8/tokenizer.json`. Repeat at widths 2 and 4 against matching `--max-active`, saving results with `--evidence FILE`. Require exit zero, ordered JSON and SSE parity with diagnostic IDs and an independent UTF-8 decoder. Cover all prompt forms, token-ID equivalents, zero budget, no usage, individual stops, queued/active resets, peer and replacement requests, and 400 for invalid prompts or `n`.

Require matched seeded IDs, finishes, and usage across text/chat and JSON/SSE at identical logits, including eager, replay and slot reuse. Use the [CUDA oracle](../tests/text/README.md#sampling) for tie, cutoff, mask and invalid rows. Check greedy and seed boundaries, omitted seed, `top_k` clamping, invalid controls, peer survival and one four-byte readback per sampled row without diagnostics. Apply the [positive depth gates](mtp.md#sampled-verification).

Use a separate checkpoint directory sharing immutable weights. Remove or invalidate its tokenizer and run the fixture with `--expect-tokenizer-error`. Require exit zero, text 500 and unchanged benchmark IDs. Restore the tokenizer and require the next text request to succeed in the same process. Follow the [benchmark checks](development.md#streaming-verification).

### Finite-choice verification

Run the [fixture](../tests/text/README.md#finite-choice-constraints) and [literal request](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/text-serving.md#finite-choice-constraints) on text/chat JSON and SSE. Require legal literals, `stop`, exact committed-ID usage and diagnostic parity for the boundary cases in the fixture. Reject invalid syntax, unsupported fields, insufficient budget, and stop overrides. Require eager/replay byte and score parity, a mixed K0 cohort, peer survival after an invalid row, slot reuse, and separate dispatch and [scratch](batch-graphs.md#constraints-and-failure-behavior) traces. Follow [positive depth checks](mtp.md#constrained-verification).
