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

HTTP caps headers at 8 KiB (431) and bodies at 2 MiB (413). It rejects malformed JSON, duplicate decoded keys, nesting beyond 32 containers, and floating-point range errors. Oversized integer `top_k` values are clamped after validation. The frozen `benchmark` parser retains signed-zero underflow and fixed fields. [Numeric dispatch](../apps/server/completions.cpp#L175)<!-- prefix: "json::Value " --> · [Benchmark parser](../apps/server/completions.cpp#L28)<!-- prefix: "CompletionRe" -->

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

Text, chat, and `/generate` share the immutable tokenizer. On first use, the server checks the size and SHA-256 of the checkpoint's `tokenizer.json`. A missing, unreadable, or mismatched asset returns HTTP 500 before output starts. Later requests retry the load. The benchmark ID route requires no tokenizer.

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

Raw text defaults to `temperature: 1`, `top_p: 1`, `top_k: -1`, and `min_p: 0`. Explicit values override them. Null numeric controls other than `seed` return HTTP 400. Chat uses its [server-selected defaults](chat-serving.md#sampling-defaults-and-output-budget). [Route defaults](../apps/server/text_request.cpp#L339)<!-- prefix: "TextRequest " -->

The decoder validates options even on the greedy path. CUDA ranks scores in descending order and breaks ties by ascending token ID. It divides score differences by temperature before exponentiation. For `top_p < 1`, the nucleus includes the token where cumulative weight first exceeds `top_p` times the selection row's total weight. The unconstrained row covers the vocabulary. Selection draws from the nucleus intersected with top-k and weights at or above the `min_p` threshold. Each draw depends on the resolved seed and zero-based output index. Identical rows and options preserve draws across slot changes. [Seed resolution](../src/decoder.cpp#L120)<!-- prefix: "SamplingPara" --> · [Cutoff and draw](../src/decoder_sampling.cu#L195)<!-- prefix: "__global__ v" -->

Native text requests can use configured [sampled MTP](mtp.md#sampled-mtp), including requests with an output constraint. The frozen benchmark ID route stays greedy. K0 sampling runs on the decoder stream after eager scoring or [eligible graph replay](batch-graphs.md#runtime-path). [Batch dispatch](../apps/server/batch_service.cpp#L849)<!-- prefix: "DecoderResul" -->

Negative infinity masks a token. NaN, positive infinity, or a row without any finite logits fails the sampled request. In batch mode, the decoder retires the failed lane and preserves its peers. For unconstrained requests, serial C++ generation throws `DecoderSamplingError`, and the batch step sets `sampling_failed`. If generation fails before the response starts, HTTP returns 500. Once SSE starts, the server attempts an error event and closes the stream. [Sample readback](../src/decoder.cpp#L376)<!-- prefix: "int read_tok" --> · [Batch retirement](../src/decoder_batch.cpp#L751)<!-- prefix: "if (result.s" -->

An unseeded sampled request resolves one seed with `std::random_device`. The frozen benchmark client explicitly sends `temperature: 0`, `seed: 0`, and `ignore_eos: true`. [Benchmark request](../benchmarks/streaming/client.py#L40)<!-- prefix: "def request_" -->

## Finite-choice constraints

Use top-level `ebnf` with [one rule](output-constraints.md#native-ebnf), such as `root ::= "yes" | "no"`, on text or chat. Allow up to 32 nonempty valid UTF-8 literals with JSON escapes, at most 256 decoded bytes each and 2,048 bytes total. Grammar text after request decoding is limited to 16,384 bytes. Each state allows 4,094 ordinary choices plus two reserved EOS entries. HTTP 400 covers malformed or non-string grammar, extra rules, concatenation, repetition, no tokenizable completion within budget, another output constraint, explicit `stop_token_ids` including `[]`, or `ignore_eos: true`. The benchmark route accepts only its fixed fields. [Runnable literal request](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/text-serving.md#finite-choice-constraints).

Each request has a cursor matching checkpoint token bytes, including tokens spanning prefix nodes or partial UTF-8 scalars. It excludes paths that cannot finish within the remaining budget.

CUDA gathers allowed BF16 scores in token ID order or masks the adjusted FP32 row. Both run after eager scoring or K0 graph replay. Greedy breaks ties by lowest token ID. Sampling uses allowed-token weights. The selector returns a vocabulary ID for the four-byte readback. Serial C++ `top_logits` reports full-vocabulary scores. [BF16 selection](../src/decoder_constraint.cpp#L28)<!-- prefix: "void select_" --> · [FP32 selection](../src/decoder.cpp#L210)<!-- prefix: "void Adjuste" -->

A request stops when it reaches a leaf or completes a literal at the budget limit. This adds no EOS token. At a literal that prefixes a longer choice, the next step can select model EOS if budget remains. A selected EOS counts in usage and diagnostic IDs. The writer omits it from text. Every successful EBNF completion reports HTTP `finish_reason: "stop"`. [Cursor update](../src/tokenizer/constraint.cpp#L162)<!-- prefix: "bool Constra" --> · [HTTP finish](../apps/server/text_completion.cpp#L212)<!-- prefix: "const char* " -->

Constrained text can use the configured [MTP depth](mtp.md#runtime-path) and share a batch with other text requests. K0 constraint selection can follow [eligible graph replay](batch-graphs.md#runtime-path).

In C++, include `sglang/decoder_constraint.h`. Link `sglang_tokenizer` to compile a `TokenConstraint` with the checkpoint tokenizer. Assign the shared constraint to `DecoderOptions::constraint`. To use MTP, construct the decoder with draft weights. Set `DecoderOptions::mtp_steps` to a value from 1 through 4. Use only the two model EOS IDs if you supply stop IDs. Successful C++ generation reports `DecoderFinishReason::constraint`. [C++ validation](../src/decoder.cpp#L32)<!-- prefix: "void validat" -->

An invalid constrained winner causes serial `DecoderConstraintError`. The batch marks `constraint_failed` and retires that lane. HTTP uses the same failure handling as for sampling.

## Read the response

JSON returns `choices` in prompt order with `index`, `text`, and `finish_reason` in each choice. SSE sends one choice per data event. Choices can interleave. Join text deltas by the zero-based `index` inside each choice. Each index has one terminal choice. When `stream_options.include_usage` is `true`, one usage event with `choices: []` follows all terminal choices. A successful stream ends with one `[DONE]`. [Batch response](../apps/server/text_completion.cpp#L818)<!-- prefix: "void complet" -->

Usage totals prompt and output IDs. `prompt_tokens_details.cached_tokens` sums [restored prompt IDs](prefix-cache.md#runtime-path). A matching stop ID counts as output and appears in diagnostic IDs. The writer omits its text and reports `stop`, even at the budget. Unconstrained budget exhaustion reports `length`. See [finite-choice](#finite-choice-constraints) and [other constrained finishes](output-constraints.md#selection-and-finish).

Each choice skips special and unmapped IDs in text, holds incomplete UTF-8 across tokens, and replaces malformed bytes or an incomplete final character with U+FFFD. Diagnostic SSE can carry an ID with empty text or final text with zero IDs. Each index's first diagnostic choice includes `prompt_token_ids`, even if that choice is terminal at zero budget. [Response writer](../apps/server/text_completion.cpp#L581)<!-- prefix: "bool TextRes" -->

Both formats omit a matched text stop and subsequent text in that choice. The writer holds a possible stop prefix across tokens and flushes it if generation ends without a match. Nonempty text stops apply with `ignore_eos: true` and conflict with output constraints. The decoder matches before commit, including during [batched MTP](mtp.md#runtime-path). It yields `stop` and counts through the matching token. Other choices continue. A failed write or TCP reset cancels unfinished choices and discards pending bytes. [Stop preview](../apps/server/text_completion.cpp#L303)<!-- prefix: "std::functio" --> · [Batch token delivery](../apps/server/text_completion.cpp#L679)<!-- prefix: "bool consume" --> · [Completion streams](http-server.md#completion-streams)

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

For active sampling, match IDs, finish and usage in both routes and formats at a fixed seed. Hold the checkpoint, prefill, prompt and budget fixed. Compare eager, replay and reassigned slots at identical logits. Cover the greedy boundary, signed seed limits, omitted seed and `top_k` clamping. Reject invalid controls. Use the [CUDA oracle](../tests/text/README.md#sampling) for ties, exact and combined cutoffs, mask and invalid rows. Require peer output, slot reuse and one four-byte ID readback per sampled row without diagnostics. Follow the [positive-depth checks](mtp.md#sampled-verification).

With a separate checkpoint directory sharing the immutable weights, make `tokenizer.json` missing or invalid and run the HTTP fixture with `--expect-tokenizer-error`. Require exit zero, text HTTP 500 and identical benchmark IDs across the fault. Restore the pinned tokenizer and require the next text request to succeed in the same process. Run the [benchmark streaming checks](development.md#streaming-verification) separately.

### Finite-choice verification

Use the [finite-choice fixture](../tests/text/README.md#finite-choice-constraints) and [literal request](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/text-serving.md#finite-choice-constraints) on text and chat in JSON and SSE with diagnostics. Require legal literals, `stop`, and usage matching committed IDs for shared prefixes, escapes, split UTF-8, alternate tokenizations, leaf completion, EOS and acceptance at the budget. Reject invalid syntax, unsupported fields, insufficient budget and stop overrides. Check eager/replay against byte and score oracles, mixed K0 lanes, peer survival after an invalid allowed row and safe slot reuse. Require separate trace evidence for dispatch and [batch scratch](batch-graphs.md#runtime-path). Run the [positive-depth checks](mtp.md#constrained-verification).
