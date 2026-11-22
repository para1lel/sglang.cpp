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

Each string must contain 1 to 1,048,576 UTF-8 bytes. Each ID sequence must be nonempty and contain integers in `[0, 248320)`. IDs go directly to the decoder. Each prompt and its output budget must fit 262,144 tokens. A batch contains 1 to 32 prompts of one form and shares request options across all items. The sum of output budgets must be at most 262,144 tokens. Invalid prompts and options return HTTP 400 before decoder admission. [Prompt parsing](../apps/server/text_request.cpp#L461)<!-- prefix: "TextCompleti" --> · [Resident memory admission](batch-graphs.md#resident-memory-admission)

HTTP caps headers at 8 KiB (431) and bodies at 2 MiB (413). It rejects malformed JSON, duplicate decoded keys, nesting beyond 32 containers, and floating-point range errors. Oversized integer `top_k` values are clamped after validation. The frozen `benchmark` parser retains signed-zero underflow and fixed fields. [Numeric dispatch](../apps/server/completions.cpp#L175)<!-- prefix: "json::Value " --> · [Benchmark parser](../apps/server/completions.cpp#L28)<!-- prefix: "CompletionRe" -->

| Field | Behavior |
| --- | --- |
| `max_tokens` | Nonnegative integer output budget. Omit it to use 16. Null is invalid. For an unconstrained request, zero returns no output IDs. |
| `min_tokens` | [Minimum output](output-penalties.md#minimum-output), default zero. |
| `n` | Omit it or send the integer `1`. Each prompt yields one choice. |
| `echo` | Boolean, default `false`. Set it to `true` to prepend the prompt to each choice's text. |
| `stream` | Boolean, default `false`. Set it to `true` for SSE. |
| `stream_options` | For SSE, set `{"include_usage":true}` to receive a separate usage event. |
| `ignore_eos` | Boolean, default `false`. Set it to `true` to disable model stop IDs `248044` and `248046` and all explicit stop IDs. |
| `stop_token_ids` | Array of explicit stop IDs in `[0, 248320)`. The server validates them even when `ignore_eos` is `true`. |
| `stop` | One nonempty string or an array of nonempty strings. Omission, null, and an empty array disable text stops. |
| `return_token_ids` | Boolean, default `false`. Set it to `true` to include diagnostic prompt and output IDs for each choice. |
| `logprobs` | Integer `0`–`20`. Requests [token scores](serving/openai-logprobs.md#text-completions). |
| `logit_bias` | Optional [per-token offsets](serving/openai-logprobs.md#logit-bias). |
| `ebnf` | Optional string that restricts output to [one literal choice](#finite-choice-constraints). |
| `regex` | Optional [whole-output regular expression](output-constraints.md#regular-expressions). |
| `response_format` | Optional [bounded JSON Schema](output-constraints.md#json-schema). |
| `cache_salt` | Optional [prefix cache key](prefix-cache.md#runtime-path) component. |

Echo copies string prompts and decodes ID prompts with special tokens hidden. Each SSE index echoes once, in its first choice. Stops apply to generated output. With `max_tokens: 0`, echo returns the prompt with zero completion tokens.

The text route accepts [output penalties](output-penalties.md), including `repetition_penalty`, and rejects `max_completion_tokens` and unknown options. [Request options](../apps/server/text_request.cpp#L450)<!-- prefix: "bool is_text" -->

Text, chat, and `/generate` share the immutable tokenizer. On first use, the server checks the size and SHA-256 of the checkpoint's `tokenizer.json`. A missing, unreadable, or mismatched asset returns HTTP 500 before output starts. Later requests retry the load. The benchmark ID route requires no tokenizer.

## Serial prefill

At K0, the generator and direct `Decoder::generate` calls use the serial prefill path. The generator and direct C++ calls use [batch scheduling](batch-graphs.md#runtime-path) when MTP is enabled. The HTTP server uses it at every capacity. Larger serial prefill tiles require the `tensor-core` FP8 policy. The default policy is `exact`. Default, eager, and overlapping serial K0 calls share the prefill selector. CUDA graphs cover later decode steps. [K0 dispatch](../src/decoder.cpp#L709)<!-- prefix: "if (proposal" --> · [Prefill and decode](../src/decoder.cpp#L642)<!-- prefix: "const auto r" -->

The binaries and default C++ constructor use visible CUDA ordinal 0. Use `--device` or the constructor's device argument to select another ordinal. With `SGLANG_SERIAL_PREFILL_ROWS` unset, the selected device must report `NVIDIA B200` and the encoded prompt must have at least 512 tokens. The automatic cap is then 1,024 rows. Other device names and shorter prompts use the 128-row cap. [Device check](../src/decoder_weights.cpp#L141)<!-- prefix: "b200_ = acce" --> · [Row selector](../src/decoder.cpp#L24)<!-- prefix: "int serial_p" -->

Set `SGLANG_SERIAL_PREFILL_ROWS=128` to retain 128-row tiles. Values `256`, `512`, `1024`, and `2048` override the cap for serial tensor-core K0 on supported devices. An empty or unrecognized value uses the 128-row cap. Exact prefill, positive-depth MTP, and the batch scheduler ignore this setting.

Prefill processes the aligned prefix and any shorter tail in order. The larger tile fits the prompt and the step cap, both rounded down to multiples of 128. If large scratch is unavailable, prefill retries with a tile bounded by 128 and the step cap. If normal scratch is unavailable, prefill processes one row at a time. Unavailable scratch means an allocation ran out of memory or the pool is unsupported. Other allocation errors propagate. [Tile loop and retry](../src/decoder.cpp#L505)<!-- prefix: "bool Decoder" --> · [Workspace allocation](../src/decoder_prefill.cpp#L694)<!-- prefix: "PrefillWorks" -->

The actual tile size determines eligibility for [MXFP8 flash prefill](flash-prefill.md#constraints-and-failure-behavior).

### Serial prefill verification

Reserve a B200 and use the same release binaries for both settings. Follow the [native prefill runbook](../tests/prefill/README.md) for exact controls. For serial K0, run the generator with `--mtp-steps 0`. With tensor-core prefill, compare the unset setting with `SGLANG_SERIAL_PREFILL_ROWS=128`. Check workspace capacity, full BF16 scores, and output IDs at 511, 512, and a longer prompt with a ragged tail. Inject recoverable allocation failures for large and normal scratch. Compare each retry or fallback with its separately forced control. Retry a later request on the same decoder. Check exact, K1, and batched requests for their original tile bounds. Use the [frozen W1/W2 procedure](performance.md#frozen-w1-and-w2-timing) for HTTP timing.

## Request progress

Serial generation and batch lanes track the next target position and the number of emitted output IDs separately. The position advances with restored prefixes, prompt inputs, and consumed decode rows. Scoring the final prompt row emits the first output without processing another input. Each emitted ID uses one unit of the output budget. For a positive output budget, the logical target KV limit is `prompt length + output budget - 1` positions. [Progress state](../include/sglang/decoder_internal.h#L421)<!-- prefix: "struct Reque" --> · [First score](../src/decoder.cpp#L552)<!-- prefix: "if (progress" -->

If `is_cancelled` returns true at entry, direct C++ generation finishes with `cancelled` before prefill. To deliver a selected ID, it appends the ID, calls `on_token`, then records requested top logits. If `on_token` returns false, the result retains that ID and its requested logits and finishes with `cancelled`. Otherwise, a constraint finish or stop match takes precedence over length. In direct MTP, the callback runs before the verifier commits the [consumed target state](mtp.md#runtime-path). [Result update](../src/decoder.cpp#L435)<!-- prefix: "bool SerialO" --> · [Finish decision](../src/decoder.cpp#L327)<!-- prefix: "std::optiona" -->

## Sampling

Native HTTP uses these controls for JSON and SSE. C++ and the generator use the same controls and default to greedy selection. [HTTP validation](../apps/server/text_request.cpp#L200)<!-- prefix: "void set_sam" --> · [C++ options](../include/sglang/decoder.h#L32)<!-- prefix: "struct Decod" --> · [C++ validation](../src/decoder.cpp#L75)<!-- prefix: "void validat" -->

| Field | Valid values |
| --- | --- |
| `temperature` | Finite nonnegative number. Values below `1e-6` select greedy decoding. |
| `top_p` | Number in `(0, 1]`. A value of `1` disables the nucleus cutoff. |
| `top_k` | Integer `-1` for the full vocabulary, or positive. Larger values clamp to 248,320. A value of `1` selects greedy decoding. |
| `min_p` | Number in `[0, 1]`. The minimum weight is `min_p` times the maximum weight in the selection row. |
| `seed` | Signed 64-bit integer. Omission or null leaves the seed unset. |

Raw text defaults to `temperature: 1`, `top_p: 1`, `top_k: -1`, and `min_p: 0`. Explicit values override them. Null numeric controls other than `seed` return HTTP 400. Chat uses its [server-selected defaults](chat-serving.md#sampling-defaults-and-output-budget). [Route defaults](../apps/server/text_request.cpp#L340)<!-- prefix: "TextRequest " -->

The decoder validates options even on the greedy path. CUDA ranks scores in descending order and breaks ties by ascending token ID. It divides score differences by temperature before exponentiation. For `top_p < 1`, the nucleus includes the token where cumulative weight first exceeds `top_p` times the selection row's total weight. The unconstrained row covers the vocabulary. Selection draws from the nucleus intersected with top-k and weights at or above the `min_p` threshold. Each draw depends on the resolved seed and zero-based output index. Identical rows and options preserve draws across slot changes. [Seed resolution](../src/decoder.cpp#L133)<!-- prefix: "SamplingPara" --> · [Cutoff and draw](../src/decoder_sampling.cu#L195)<!-- prefix: "__global__ v" -->

Native text requests can use configured [sampled MTP](mtp.md#sampled-mtp), including requests with an output constraint. The frozen benchmark ID route stays greedy. K0 sampling runs on the decoder stream after eager scoring or [eligible graph replay](batch-graphs.md#runtime-path). [Batch dispatch](../apps/server/batch_service.cpp#L883)<!-- prefix: "DecoderResul" -->

Negative infinity masks a token. NaN, positive infinity, or a row without any finite logits fails the sampled request. In batch mode, the decoder retires the failed lane and preserves its peers. For unconstrained requests, serial C++ generation throws `DecoderSamplingError`, and the batch step sets `sampling_failed`. If generation fails before the response starts, HTTP returns 500. Once SSE starts, the server attempts an error event and closes the stream. [Sample readback](../src/decoder.cpp#L362)<!-- prefix: "int read_tok" --> · [Batch retirement](../src/decoder_batch.cpp#L903)<!-- prefix: "if (result.s" -->

An unseeded sampled request resolves one seed with `std::random_device`. The frozen benchmark client explicitly sends `temperature: 0`, `seed: 0`, and `ignore_eos: true`. [Benchmark request](../benchmarks/streaming/client.py#L40)<!-- prefix: "def request_" -->

## Finite-choice constraints

Set top-level `ebnf` on a `/v1/completions` or `/v1/chat/completions` request to restrict output to one literal in JSON or SSE. Use one rule of the form `root ::= "yes" | "no"`. Each alternative is a nonempty double-quoted string with JSON escapes. The decoded bytes must be valid UTF-8. A single literal is valid. [/generate field](output-constraints.md#native-ebnf) · [HTTP validation](../apps/server/text_request.cpp#L287)<!-- prefix: "void set_con" --> · [Compiler](../src/tokenizer/constraint.cpp#L82)<!-- prefix: "std::shared_" -->

```bash
curl --fail-with-body http://127.0.0.1:17384/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen3.8-27B-FP8","prompt":"Choose an answer.","max_tokens":8,"ebnf":"root ::= \"yes\" | \"no\"","return_token_ids":true}'
```

The grammar limit is 16,384 bytes after JSON request decoding. Supply up to 32 literals, with at most 256 decoded bytes per literal and 2,048 decoded bytes in total. Each state allows up to 4,094 ordinary token choices and reserves two EOS entries. HTTP returns 400 for an invalid grammar or a non-string `ebnf`. It also rejects additional EBNF rules, concatenation, repetition, and requests with no tokenizable choice within budget. Send only one of `ebnf`, [`regex`](output-constraints.md#regular-expressions), and [`response_format`](output-constraints.md#json-schema). The `benchmark` model accepts only its fixed fields. With `ebnf`, omit `stop_token_ids`, including an empty array. Leave `ignore_eos` false.

The compiler matches checkpoint token bytes against literal prefixes. A token may span several prefix nodes or parts of UTF-8 characters. Alternate tokenizations can compete for the same literal. Each request owns its cursor. At each step it excludes paths that cannot complete within the remaining token budget. [Candidate filtering](../include/sglang/decoder_constraint.h#L89)<!-- prefix: "void candida" -->

CUDA gathers allowed BF16 scores in token ID order or masks the adjusted FP32 row. Both run after eager scoring or K0 graph replay. Greedy breaks ties by lowest token ID. Sampling uses allowed-token weights. The selector returns a vocabulary ID for the four-byte readback. Serial C++ `top_logits` reports full-vocabulary scores. [BF16 selection](../src/decoder_constraint.cpp#L28)<!-- prefix: "void select_" --> · [FP32 selection](../src/decoder.cpp#L234)<!-- prefix: "void Adjuste" -->

A request stops when it reaches a leaf or completes a literal at the budget limit. This adds no EOS token. At a literal that prefixes a longer choice, the next step can select model EOS if budget remains. A selected EOS counts in usage and diagnostic IDs. The writer omits it from text. Every successful EBNF completion reports HTTP `finish_reason: "stop"`. [Cursor update](../include/sglang/decoder_constraint.h#L134)<!-- prefix: "bool commit(" --> · [HTTP finish](../apps/server/text_completion.cpp#L214)<!-- prefix: "const char* " -->

Constrained text can use the configured [MTP depth](mtp.md#runtime-path) and share a batch with other text requests. K0 constraint selection can follow [eligible graph replay](batch-graphs.md#runtime-path).

In C++, include `sglang/decoder_constraint.h`. Link `sglang_tokenizer` to compile a `TokenConstraint` with the checkpoint tokenizer. Assign the shared constraint to `DecoderOptions::constraint`. To use MTP, construct the decoder with draft weights. Set `DecoderOptions::mtp_steps` to a value from 1 through 4. Use only the two model EOS IDs if you supply stop IDs. Successful C++ generation reports `DecoderFinishReason::constraint`. [C++ validation](../src/decoder.cpp#L45)<!-- prefix: "void validat" -->

An invalid constrained winner causes serial `DecoderConstraintError`. The batch marks `constraint_failed` and retires that lane. HTTP uses the same failure handling as for sampling.

## Read the response

JSON returns `choices` in prompt order with `index`, `text`, and `finish_reason` in each choice. SSE sends one choice per data event. Choices can interleave. Join text deltas by the zero-based `index` inside each choice. Each index has one terminal choice. When `stream_options.include_usage` is `true`, one usage event with `choices: []` follows all terminal choices. A successful stream ends with one `[DONE]`. [Batch response](../apps/server/text_completion.cpp#L773)<!-- prefix: "void complet" -->

Usage totals prompt and output IDs. `prompt_tokens_details.cached_tokens` sums [restored prompt IDs](prefix-cache.md#runtime-path). A matching stop ID counts as output and appears in diagnostic IDs. The writer omits its text and reports `stop`, even at the budget. Unconstrained budget exhaustion reports `length`. See [finite-choice](#finite-choice-constraints) and [other constrained finishes](output-constraints.md#selection-and-finish).

Each choice skips special and unmapped IDs in text, holds incomplete UTF-8 across tokens, and replaces malformed bytes or an incomplete final character with U+FFFD. Diagnostic SSE can carry an ID with empty text or final text with zero IDs. Each index's first diagnostic choice includes `prompt_token_ids`, even if that choice is terminal at zero budget. [Response writer](../apps/server/text_completion.cpp#L558)<!-- prefix: "bool TextRes" -->

Both formats omit a matched text stop and subsequent text in that choice. The writer holds a possible stop prefix across tokens and flushes it if generation ends without a match. Nonempty text stops apply with `ignore_eos: true` and conflict with output constraints. A text stop yields `stop` and counts through the matching token, even when [batched MTP](mtp.md#runtime-path) produces further IDs in that step. Other choices continue. A failed write or TCP reset cancels unfinished choices and discards pending bytes. [Stop matching](../apps/server/text_completion.cpp#L483)<!-- prefix: "std::string " --> · [Finish choice](../apps/server/text_completion.cpp#L606)<!-- prefix: "std::optiona" --> · [Batch token delivery](../apps/server/text_completion.cpp#L655)<!-- prefix: "bool consume" --> · [Completion streams](http-server.md#completion-streams)

## Run with bundled libraries

Build the Linux Release server and generator with XMake and the managed Pixi toolchain. The tested x86-64 runtime uses bundled `libcudart.so.13`, `libstdc++.so.6`, and `libgcc_s.so.1`. The host provides glibc and compatible NVIDIA driver libraries. Provide the checkpoint separately.

Set `BUNDLE` to the directory with the executables in `bin/` and libraries in `lib/`. Set `CUDA_COMPAT` to the compatibility library directory if the host requires it. The binaries retain build `DT_RPATH`. Use the host loader to bypass it for the isolation check.

```bash
loader=/lib64/ld-linux-x86-64.so.2
libraries="${BUNDLE:?set the bundle directory}/lib${CUDA_COMPAT:+:$CUDA_COMPAT}"
for program in sglang-server sglang-generate; do
    readelf -d "$BUNDLE/bin/$program"
    "$loader" --inhibit-rpath '' --library-path "$libraries" --list "$BUNDLE/bin/$program"
    "$loader" --inhibit-rpath '' --library-path "$libraries" "$BUNDLE/bin/$program" --help
done
```

Check that both programs resolve without LLVM or Pixi paths. Use the same loader options with the usual server or generator arguments.

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

Send all four forms in JSON and SSE with `temperature: 0` and `return_token_ids: true`. Compare equivalent text and IDs. Check JSON ordering, SSE indices, finishes, usage, and `[DONE]`. Repeat at zero budget and without usage. Stop one choice while peers finish. Reset with queued and active items and verify reuse. Require HTTP 400 for empty or mixed arrays, invalid IDs, and unsupported `n` values.

In the default Pixi environment, build and run the CPU tokenizer check. Use the immutable checkpoint's `tokenizer.json`.

```bash
xmake f -P . -o build -p linux -a x86_64 -m release --text-tokenizer-check=y
xmake build -P . text-tokenizer-check
./build/bin/text-tokenizer-check /data0/shared/Qwen3.8-27B-FP8/tokenizer.json \
  tests/text/cases.json
```

Require exit status zero. The check compares exact prompt IDs with [pinned fixtures](../tests/text/cases.json#L2)<!-- prefix: "\"tokenizer_s" --> from Hugging Face Tokenizers 0.23.3. It also checks token budgets, UTF-8 carry and flush, normalization, and pretokenizer splits.

Start the release server on a reserved B200 with `--max-active 1`. Run the HTTP check from the project root. Use `--evidence FILE` to save the output IDs and cancellation result.

```bash
/data1/shared/dongwu.chen/.pixi/envs/default/bin/python -B \
  tests/text/test_native_text.py --port 17384 --width 1 \
  --tokenizer /data0/shared/Qwen3.8-27B-FP8/tokenizer.json
```

Repeat with a server at `--max-active 2` and `4`. Match `--width` to the server. Require exit status zero at each width. The [HTTP check](../tests/text/test_native_text.py#L1)<!-- prefix: "\"\"\"Run again" --> compares diagnostic output IDs, finish reasons, and usage with token-ID requests. It checks SSE and JSON text against an independent UTF-8 decoder. It exercises empty deltas, split characters, terminal flush, stops, zero budget, and cancellation with pending bytes. The concurrent cases check peer and replacement responses.

Run both native routes with `return_token_ids: true`, a fixed seed, and active sampling. Match output IDs, finish reasons, and usage across JSON and SSE. Fix the checkpoint, prefill policy, prompts, and budgets. Compare eager, replay, and reassigned slots with identical logits. Follow the [sampled MTP checks](mtp.md#sampled-verification) for positive depth. Check the greedy threshold, both signed seed limits, omitted seed, and top-k clamping. Require HTTP 400 for invalid values, including `min_p` outside `[0, 1]`, zero top-p, zero top-k, and out-of-range seeds. Run the frozen benchmark check separately.

At the operator boundary, compare CUDA winners with a CPU oracle. Cover ties, exact nucleus and min-p cutoffs, combined filters, masked logits, and invalid rows. Inject an invalid sampled row in a batch and check peer output and slot reuse. With diagnostics disabled, verify one four-byte selected ID transfer per sampled row.

Verify asset errors with a separate checkpoint directory that shares the immutable weights. Restart the server with that `--checkpoint` and with `tokenizer.json` missing or altered.

```bash
/data1/shared/dongwu.chen/.pixi/envs/default/bin/python -B \
  tests/text/test_native_text.py --port 17384 --expect-tokenizer-error
```

Require exit status zero. This check requires HTTP 500 for a valid text request and identical benchmark ID output before and after the error. Restore the pinned tokenizer in that directory. Require the next valid text request to succeed on the same process. Run the [benchmark streaming checks](development.md#streaming-verification) against the same server build.

### Finite-choice verification

Run the [example request](#finite-choice-constraints) on the release server and pinned checkpoint. Repeat with SSE, diagnostic IDs, and usage enabled. Run the same options on chat with a `messages` array in place of `prompt`. Require one literal, `stop`, and usage that matches the committed IDs. Include literals with a shared prefix, escaped characters, split UTF-8, and alternate tokenizations. Check leaf completion, EOS with budget left, and acceptance on the last allowed token. Require HTTP 400 for invalid syntax, unsupported fields, insufficient budget, and stop overrides.

On a reserved B200, compare eager and replay selection on identical rows. Use a tokenizer-byte oracle for legal continuations and a score oracle for greedy ties and fixed-seed sampling. Check mixed constrained and unconstrained K0 lanes. Inject an invalid allowed row. Require peer completion and safe slot reuse. Use a separate traced run to confirm dispatch and [batch scratch](batch-graphs.md#runtime-path). Follow the [constrained MTP checks](mtp.md#constrained-verification) for positive depth.
