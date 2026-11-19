# Native text completions

Send `POST /v1/completions` with `model` set to `Qwen3.8-27B-FP8` and `prompt` set to a UTF-8 string. The same endpoint serves the frozen `benchmark` model with a [prompt ID array and token-ID SSE](../benchmarks/streaming/README.md#wire-contract). Use [native chat](chat-serving.md) to render conversation history.

## Send a request

Follow the [server launch procedure](../README.md#serve-token-ids-over-http).

```bash
curl --fail-with-body http://127.0.0.1:17384/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen3.8-27B-FP8","prompt":"一、北京\n二、上海\n三、","max_tokens":8}'
```

The server generates one completion with greedy decoding by default. The prompt must contain 1 to 1,048,576 UTF-8 bytes. The encoded prompt length plus the output budget must fit 262,144 tokens. HTTP accepts up to 8 KiB of headers and 2 MiB of body. Invalid request options, malformed UTF-8, and invalid token limits return HTTP 400. Oversize headers return HTTP 431. Oversize bodies return HTTP 413. [Request parser](../apps/server/text_request.cpp#L176)<!-- prefix: "TextRequest " -->

The server rejects malformed JSON, duplicate decoded keys, and nesting beyond 32 containers. Native text rejects floating-point range errors. The parser preserves oversized integer `top_k` values for validation and clamping. The frozen `benchmark` parser retains its signed-zero underflow handling and fixed request fields. [Numeric dispatch](../apps/server/completions.cpp#L183)<!-- prefix: "json::Value " --> · [Benchmark parser](../apps/server/completions.cpp#L26)<!-- prefix: "CompletionRe" -->

| Field | Behavior |
| --- | --- |
| `max_tokens` | Nonnegative integer output budget, default 16. For an unconstrained request, zero encodes the prompt without generation. |
| `stream` | Boolean, default `false`. Set it to `true` for SSE. |
| `stream_options` | For SSE, set `{"include_usage":true}` to receive a separate usage event. |
| `ignore_eos` | Boolean, default `false`. Set it to `true` to disable the implicit stops `248044` and `248046`. |
| `stop_token_ids` | Array of explicit stop IDs in `[0, 248320)`. These stops also apply with `ignore_eos: true`. |
| `return_token_ids` | Boolean, default `false`. Set it to `true` to include diagnostic output IDs. Nonstreaming responses also include the encoded prompt IDs. |
| `ebnf` | Optional string that restricts output to [one literal choice](#finite-choice-constraints). |
| `cache_salt` | Optional [prefix cache key](prefix-cache.md#runtime-path) component. |

Tokenizer load occurs on the first text or chat request that reaches encoding. The server reads `tokenizer.json` from the selected checkpoint directory and verifies its size and SHA-256. A missing, unreadable, or mismatched asset returns HTTP 500 before output starts. A subsequent request retries a failed load. Successful loads share an immutable tokenizer across text and chat requests. The benchmark ID route uses the IDs supplied by the client.

## Sampling

Native text, native chat, and C++ `DecoderOptions` use the same sampling controls. These fields apply to both JSON and SSE responses. [HTTP validation](../apps/server/text_request.cpp#L66)<!-- prefix: "void set_sam" --> · [C++ options](../include/sglang/decoder.h#L21)<!-- prefix: "struct Decod" --> · [C++ validation](../src/decoder.cpp#L39)<!-- prefix: "void validat" -->

| Field | Valid values and default |
| --- | --- |
| `temperature` | Finite nonnegative number, default `0`. Values below `1e-6` use greedy selection. |
| `top_p` | Number in `(0, 1]`, default `1`. A value of `1` disables the nucleus cutoff. |
| `top_k` | Integer `-1` or positive, default `-1` for the full vocabulary. Values above 248,320 clamp to 248,320. A value of `1` uses greedy selection. |
| `seed` | Signed 64-bit integer. With active sampling and no seed, generation resolves one from system entropy. |

The decoder validates options even on the greedy path. CUDA ranks BF16 logits by descending score, then ascending token ID. It divides score differences by temperature before exponentiation. For `top_p < 1`, the nucleus ends where cumulative weight first exceeds `top_p` times the selection row's total weight. The unconstrained row covers the full vocabulary. Top-k caps the retained prefix. The decoder draws from the retained weights. Each draw depends on the resolved seed and zero-based output index. With identical logits and options, moving a request to another slot preserves its draws. [Seed resolution](../src/decoder.cpp#L77)<!-- prefix: "SamplingPara" --> · [Cutoff and draw](../src/decoder_sampling.cu#L167)<!-- prefix: "__global__ v" -->

Native HTTP uses K0 for sampled requests. Unconstrained greedy requests on the same server can still use configured MTP. Set `mtp_steps = 0` explicitly for sampled C++ requests. Graph eligibility follows the [usual dispatch rules](batch-graphs.md#runtime-path). The sampler runs on the decoder stream after model replay or eager scoring. [Serial dispatch](../apps/server/completions.cpp#L237)<!-- prefix: "void complet" --> · [Batch dispatch](../apps/server/batch_service.cpp#L540)<!-- prefix: "if (!job-\u003eop" -->

Negative infinity masks a token. NaN, positive infinity, or a row without any finite logits fails the sampled request. In batch mode, the decoder retires the failed lane and preserves its peers. For unconstrained requests, serial C++ generation throws `DecoderSamplingError`, and the batch step sets `sampling_failed`. If generation fails before the response starts, HTTP returns 500. Once SSE starts, the server attempts an error event and closes the stream. [Sample readback](../src/decoder.cpp#L97)<!-- prefix: "int read_tok" --> · [Batch retirement](../src/decoder_batch.cpp#L631)<!-- prefix: "if (result.s" -->

## Finite-choice constraints

Set `ebnf` on a native text or chat request to restrict the generated text to one literal, in JSON or SSE. Use one rule of the form `root ::= "yes" | "no"`. Each alternative is a nonempty double-quoted string with JSON escapes. The decoded bytes must be valid UTF-8. A single literal is valid. [HTTP validation](../apps/server/text_request.cpp#L114)<!-- prefix: "void set_con" --> · [Compiler](../src/tokenizer/constraint.cpp#L82)<!-- prefix: "std::shared_" -->

```bash
curl --fail-with-body http://127.0.0.1:17384/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen3.8-27B-FP8","prompt":"Choose an answer.","max_tokens":8,"ebnf":"root ::= \"yes\" | \"no\"","return_token_ids":true}'
```

The grammar limit is 16,384 bytes after JSON request decoding. Supply up to 32 literals, with at most 256 decoded bytes per literal and 2,048 decoded bytes in total. Each state allows up to 4,094 ordinary token choices and reserves two EOS entries. HTTP returns 400 for an invalid grammar or a non-string `ebnf`. It also rejects requests with no tokenizable choice within the budget. HTTP rejects additional rules, concatenation, repetition, regex, JSON Schema, and `response_format`. The `benchmark` model accepts only its fixed fields. With `ebnf`, omit `stop_token_ids`, including an empty array. Leave `ignore_eos` false.

The compiler matches checkpoint token bytes against literal prefixes. A token may span several prefix nodes or parts of UTF-8 characters. Alternate tokenizations can compete for the same literal. Each request owns its cursor. At each step it excludes paths that cannot complete within the remaining token budget. [Candidate filtering](../include/sglang/decoder_constraint.h#L44)<!-- prefix: "void candida" -->

CUDA gathers the allowed BF16 scores in ascending token ID order after eager scoring or graph replay. Greedy selects the highest score and breaks ties by lowest token ID. With active sampling, the [sampling rules](#sampling) apply to the allowed row. Top-k clamps to its size, and top-p uses its total weight. The selector maps the winning index back to a vocabulary ID for the four-byte readback. C++ serial `top_logits` still reports the original model row. [Selection](../src/decoder_constraint.cpp#L19)<!-- prefix: "void select_" -->

A request stops when it reaches a leaf or completes a literal at the budget limit. This adds no EOS token. At a literal that prefixes a longer choice, the next step can select model EOS if budget remains. A selected EOS counts in usage and diagnostic IDs. The writer omits it from text. Every successful constrained completion reports HTTP `finish_reason: "stop"`. [Cursor update](../include/sglang/decoder_constraint.h#L62)<!-- prefix: "bool commit(" --> · [HTTP finish](../apps/server/text_completion.cpp#L52)<!-- prefix: "const char* " -->

HTTP uses K0 for constrained requests. They can share a batch with unconstrained greedy MTP requests and can use [eligible graph replays](batch-graphs.md#runtime-path). In C++, include `sglang/decoder_constraint.h`. Link `sglang_tokenizer` to compile a `TokenConstraint` with the checkpoint tokenizer. Assign the shared constraint to `DecoderOptions::constraint`. Set `mtp_steps = 0`. Use only the two model EOS IDs if you supply stop IDs. Successful C++ generation reports `DecoderFinishReason::constraint`. [C++ validation](../src/decoder.cpp#L20)<!-- prefix: "void validat" -->

An invalid constrained winner causes serial `DecoderConstraintError`. The batch marks `constraint_failed` and retires that lane. HTTP uses the same failure handling as for sampling.

## Read the response

Nonstreaming responses return `choices[0].text`, `finish_reason`, and usage. SSE sends `choices[0].text` deltas, a terminal choice, optional usage, and `[DONE]`. Join the text deltas in order. Usage counts encoded prompt IDs and committed output IDs. `prompt_tokens_details.cached_tokens` counts [restored prompt IDs](prefix-cache.md#runtime-path). A matching stop ID counts as output and appears in diagnostic IDs. The writer omits its text and reports `stop`, including when it matches on the last allowed token. For unconstrained requests, exhaustion without a stop reports `length`. See [constrained finishes](#finite-choice-constraints).

The response decodes generated IDs into bytes. It skips special and unmapped IDs. The UTF-8 state belongs to the request and holds incomplete characters across tokens. Malformed output bytes produce U+FFFD. Normal completion flushes an incomplete tail as U+FFFD. Diagnostic SSE can carry an ID and empty text, or final UTF-8 text with zero IDs. A connection reset or failed write cancels that request and discards pending bytes. [Response writer](../apps/server/text_completion.cpp#L150)<!-- prefix: "bool TextRes" -->

[Batched MTP](mtp.md#runtime-path) can commit several IDs per step. The worker passes them to the same response writer in order. UTF-8 state stays with the request across tokens and rounds.

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

Run both native routes with `return_token_ids: true`, a fixed seed, and active sampling. Match output IDs, finish reasons, and usage across JSON and SSE. Use the same checkpoint, prefill policy, prompts, and budgets. Compare eager, replay, and reassigned slots with identical logits. Include a mixed batch with a sampled K0 lane and a greedy MTP lane. Check that a temperature below `1e-6` or `top_k: 1` takes the greedy path. Check both signed seed limits, omitted seed, and top-k clamping. Require HTTP 400 for invalid values, including zero top-p, zero top-k, and out-of-range seeds. Run the frozen benchmark check separately.

At the operator boundary, compare CUDA winners with a CPU oracle. Cover ties, the exact nucleus cutoff, combined top-k and top-p, masked logits, and invalid rows. Inject an invalid sampled row in a batch and check peer output and slot reuse. With diagnostics disabled, verify one four-byte selected ID transfer per sampled row.

Verify asset errors with a separate checkpoint directory that shares the immutable weights. Restart the server with that `--checkpoint` and with `tokenizer.json` missing or altered.

```bash
/data1/shared/dongwu.chen/.pixi/envs/default/bin/python -B \
  tests/text/test_native_text.py --port 17384 --expect-tokenizer-error
```

Require exit status zero. This check requires HTTP 500 for a valid text request and identical benchmark ID output before and after the error. Restore the pinned tokenizer in that directory. Require the next valid text request to succeed on the same process. Run the [benchmark streaming checks](development.md#streaming-verification) against the same server build.

### Finite-choice verification

Run the [example request](#finite-choice-constraints) on the release server and pinned checkpoint. Repeat with SSE, diagnostic IDs, and usage enabled. Run the same options on chat with a `messages` array in place of `prompt`. Require one literal, `stop`, and usage that matches the committed IDs. Include literals with a shared prefix, escaped characters, split UTF-8, and alternate tokenizations. Check leaf completion, EOS with budget left, and acceptance on the last allowed token. Require HTTP 400 for invalid syntax, unsupported fields, insufficient budget, and stop overrides.

On a reserved B200, compare eager and replay selection on identical rows. Use a tokenizer-byte oracle for legal continuations and a score oracle for greedy ties and fixed-seed sampling. Check mixed constrained and unconstrained K0 lanes. Check a separate eager step with a constrained lane and a greedy MTP lane. Inject an invalid allowed row. Require peer completion and safe slot reuse. Use a separate traced run to confirm dispatch and [batch scratch](batch-graphs.md#runtime-path).
