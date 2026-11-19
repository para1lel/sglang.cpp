# Native text completions

Send `POST /v1/completions` with `model` set to `Qwen3.8-27B-FP8` and `prompt` set to a UTF-8 string. The same endpoint serves the frozen `benchmark` model with a [prompt ID array and token-ID SSE](../benchmarks/streaming/README.md#wire-contract). Use [native chat](chat-serving.md) to render conversation history.

## Send a request

Follow the [server launch procedure](../README.md#serve-token-ids-over-http).

```bash
curl --fail-with-body http://127.0.0.1:17384/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen3.8-27B-FP8","prompt":"一、北京\n二、上海\n三、","max_tokens":8}'
```

The server generates one completion with greedy decoding by default. The prompt must contain 1 to 1,048,576 UTF-8 bytes. The encoded prompt length plus the output budget must fit 262,144 tokens. HTTP accepts up to 8 KiB of headers and 2 MiB of body. Invalid request options, malformed UTF-8, and invalid token limits return HTTP 400. Oversize headers return HTTP 431. Oversize bodies return HTTP 413. [Request parser](../apps/server/text_request.cpp#L154)<!-- prefix: "TextRequest " -->

The server rejects malformed JSON, duplicate decoded keys, and nesting beyond 32 containers. Native text rejects floating-point range errors. The parser preserves oversized integer `top_k` values for validation and clamping. The frozen `benchmark` parser retains its signed-zero underflow handling and fixed request fields. [Numeric dispatch](../apps/server/completions.cpp#L178)<!-- prefix: "json::Value " --> · [Benchmark parser](../apps/server/completions.cpp#L26)<!-- prefix: "CompletionRe" -->

| Field | Behavior |
| --- | --- |
| `max_tokens` | Nonnegative integer output budget, default 16. Zero loads the tokenizer and encodes the prompt without running generation. |
| `stream` | Boolean, default `false`. Set it to `true` for SSE. |
| `stream_options` | For SSE, set `{"include_usage":true}` to receive a separate usage event. |
| `ignore_eos` | Boolean, default `false`. Set it to `true` to disable the implicit stops `248044` and `248046`. |
| `stop_token_ids` | Array of explicit stop IDs in `[0, 248320)`. These stops also apply with `ignore_eos: true`. |
| `return_token_ids` | Boolean, default `false`. Set it to `true` to include diagnostic output IDs. Nonstreaming responses also include the encoded prompt IDs. |

Tokenizer load occurs on the first text or chat request that reaches encoding. The server reads `tokenizer.json` from the selected checkpoint directory and verifies its size and SHA-256. A missing, unreadable, or mismatched asset returns HTTP 500 before output starts. A subsequent request retries a failed load. Successful loads share an immutable tokenizer across text and chat requests. The benchmark ID route uses the IDs supplied by the client.

## Sampling

Native text, native chat, and C++ `DecoderOptions` use the same sampling controls. These fields apply to both JSON and SSE responses. [HTTP validation](../apps/server/text_request.cpp#L65)<!-- prefix: "void set_sam" --> · [C++ options](../include/sglang/decoder.h#L17)<!-- prefix: "struct Decod" --> · [C++ validation](../src/decoder.cpp#L22)<!-- prefix: "void validat" -->

| Field | Valid values and default |
| --- | --- |
| `temperature` | Finite nonnegative number, default `0`. Values below `1e-6` use greedy selection. |
| `top_p` | Number in `(0, 1]`, default `1`. A value of `1` disables the nucleus cutoff. |
| `top_k` | Integer `-1` or positive, default `-1` for the full vocabulary. Values above 248,320 clamp to 248,320. A value of `1` uses greedy selection. |
| `seed` | Signed 64-bit integer. With active sampling and no seed, generation resolves one from system entropy. |

The decoder validates options even on the greedy path. CUDA ranks BF16 logits by descending score, then ascending token ID. It divides score differences by temperature before exponentiation. For `top_p < 1`, the nucleus ends at the first token that raises cumulative weight strictly above `top_p` times full vocabulary weight. Top-k caps the retained prefix. The decoder draws from the retained weights. Each draw depends on the resolved seed and zero-based output index. With identical logits and options, moving a request to another slot preserves its draws. [Seed resolution](../src/decoder.cpp#L59)<!-- prefix: "SamplingPara" --> · [Cutoff and draw](../src/decoder_sampling.cu#L167)<!-- prefix: "__global__ v" -->

Native HTTP uses K0 for sampled requests. Greedy requests on the same server can still use configured MTP. Set `mtp_steps = 0` explicitly for sampled C++ requests. Graph eligibility follows the [usual dispatch rules](batch-graphs.md#runtime-path). The sampler runs on the decoder stream after model replay or eager scoring. [Serial dispatch](../apps/server/completions.cpp#L232)<!-- prefix: "void complet" --> · [Batch dispatch](../apps/server/batch_service.cpp#L337)<!-- prefix: "if (!job->op" -->

Negative infinity masks a token. NaN, positive infinity, or a row without any finite logits fails the sampled request. In batch mode, the decoder retires the failed lane and preserves its peers. C++ serial generation throws `DecoderSamplingError`. The C++ batch step sets `sampling_failed`. If generation fails before the response starts, HTTP returns 500. Once SSE starts, the server attempts an error event and closes the stream. [Sample readback](../src/decoder.cpp#L79)<!-- prefix: "int read_tok" --> · [Batch retirement](../src/decoder_batch.cpp#L552)<!-- prefix: "if (result.s" -->

## Read the response

Nonstreaming responses return `choices[0].text`, `finish_reason`, and usage. SSE sends `choices[0].text` deltas, a terminal choice, optional usage, and `[DONE]`. Join the text deltas in order. Usage counts encoded prompt IDs and committed output IDs. A matching stop ID counts as output and appears in diagnostic IDs. The writer omits its text and reports `stop`, including when it matches on the last allowed token. The writer reports `length` when the output budget is exhausted without a stop.

The response decodes generated IDs into bytes. It skips special and unmapped IDs. The UTF-8 state belongs to the request and holds incomplete characters across tokens. Malformed output bytes produce U+FFFD. Normal completion flushes an incomplete tail as U+FFFD. Diagnostic SSE can carry an ID and empty text, or final UTF-8 text with zero IDs. A connection reset or failed write cancels that request and discards pending bytes. [Response writer](../apps/server/text_completion.cpp#L143)<!-- prefix: "bool TextRes" -->

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
