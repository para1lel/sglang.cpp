# sglang.cpp

A C++ and CUDA C++ inference engine in development for **Qwen3.8-27B-FP8 on one NVIDIA B200**.
The native decoder loads the official text weights and runs all 64 layers for
token-ID generation. It uses paged MXFP8 KV storage in the 16 full-attention layers
and carries Gated DeltaNet state through the other 48 layers. Prefill and the first
token selection run eagerly. Serial target-only decoding uses a CUDA graph by default.

The reusable runtime API is in
[`include/sglang/decoder.h`](include/sglang/decoder.h#L102)<!-- prefix: "class Decode" -->.
The batch scheduler options are in
[`DecoderBatchOptions`](include/sglang/decoder_batch.h#L47)<!-- prefix: "struct Decod" -->.
The paged KV-cache and causal grouped-query attention API is in
[`include/sglang/mxfp8_gqa.h`](include/sglang/mxfp8_gqa.h#L48)<!-- prefix: "cudaError_t " -->.
The CUDA cache and public attention implementation is in
[`src/mxfp8_gqa.cu`](src/mxfp8_gqa.cu#L833)<!-- prefix: "cudaError_t " -->.
The internal page-parallel path for batched decode is in
[`src/mxfp8_gqa.cu`](src/mxfp8_gqa.cu#L947)<!-- prefix: "cudaError_t " -->.
The internal grouped-query attention prefill path is in
[`src/mxfp8_gqa_prefill.cu`](src/mxfp8_gqa_prefill.cu#L841)<!-- prefix: "cudaError_t " -->.

## Build on the project host

Run these commands from this checkout. XMake uses the CUDA Toolkit and C++ compiler from the
`default` environment in `/data1/shared/dongwu.chen/pixi.toml`.

```bash
bash tools/toolchain-xmake.sh --install
bash tools/build.sh --build-dir build
nvidia-smi --query-gpu=index,name,uuid,memory.free --format=csv
bash tools/build.sh --build-dir build --gpu 4
```

Choose an available B200 index in the last command. The build compiles `sglang-generate`,
its native libraries, and both toolchain probes. It verifies the AOT probe's native cubin
and resolved CUDA runtime.
With `--gpu`, the AOT probe launches one kernel and checks its return value.
Without `--build-dir`, the script uses a temporary build directory.

To run the P1 NVRTC probe, use the physical index of a reserved B200.

```bash
bash tools/build.sh --build-dir build --nvrtc --gpu "${GPU_INDEX:?set the reserved B200 index}"
```

The probe compiles a native `sm_100` kernel and caches the loaded function in its CUDA
context for the second launch. Both launches verify all 257 integers. The output records
the device UUID and properties, allocation size, cold and warm cache results,
NVRTC compilation time, and completed kernel time from CUDA events.

Build the candidate MXFP8 correctness check without using a GPU.

```bash
bash tools/build.sh --build-dir build --mxfp8-check
```

`mxfp8-gqa-check` links `sglang_cuda`. Run the reported executable on a reserved B200
to check cache bytes and attention outputs. This build command only compiles and links the check.

For decoded RGB8 images and video, see the [CPU media preprocessing API](docs/media-preprocessing.md).
The guide covers the standalone C++ library, patch layout, and CPU verification.
To turn those patches into BF16 features on a B200, see the [standalone vision encoder](docs/vision-encoder.md).
See the [native multimodal decoder](docs/multimodal-decoder.md) to generate from those features in C++.

## Generate text token IDs

Use the absolute executable path printed by the build as `Native generator`. Select the
UUID of a reserved B200 from the `nvidia-smi` output.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
  "$GENERATOR" --device 0 --prompt-ids 760,6511,314,9338,369 \
  --max-new-tokens 8 --warmup 2 --repeats 7
```

Set `GENERATOR` to that executable path before running the command. The default checkpoint is
`/data0/shared/Qwen3.8-27B-FP8`. Use `--prompt-file` for IDs separated by commas or ASCII whitespace,
`--stop-token` to finish after emitting that ID, and `--top-logits` to report the highest
raw logits per generated token.

The JSON report includes token IDs, finish reason, model load time, completed request times,
and output tokens per second. Each request resets its inference state and includes prompt
processing and generation. The timing samples exclude warmups and model loading. The
reported peak device allocation bytes count the decoder's CUDA allocations for the model
and the buffers used by that request.

C++ callers link the `sglang_decoder` target and reuse `sglang::Decoder` across requests.
The model retains one set of target-only text request buffers for serial reuse. The first
subsequent decode step captures the graph on demand, and later steps and requests replay
it while capacity suffices. Overlapping calls that cannot acquire the serial slot use
private request buffers and eager decode.

For image or video inputs, C++ callers pass template token IDs and completed BF16 device
features to `Decoder::generate_multimodal`. It runs eager with private state and `mtp_steps = 0`.
See the [multimodal decoder contract](docs/multimodal-decoder.md) for layout and ownership.

Serial single-row target decode uses packed FP8 activation quantization, fused BF16
Gated DeltaNet gate projections, and a paired-row BF16 LM head for eligible shapes.
Graph and eager decode use the same arithmetic for these operations.
[Decode operators](src/text_decode_bandwidth.cu#L194)<!-- prefix: "cudaError_t " -->

The generator and HTTP server accept `--fp8-prefill exact|tensor-core`. The default
`exact` uses the exact FP8 projection path. For target-only requests, `tensor-core`
uses SM100 Tensor Cores for eligible FP8 projections in chunks of at least three rows.
Smaller chunks and ineligible projections use the exact path.
Tensor Core GDN prefill uses chunked recurrence on SM100 for eligible calls with at
least 128 rows. The first GDN bank uses ordered recurrence for the tile at absolute
context position zero. Exact prefill keeps its original recurrence.
With `SGLANG_SERIAL_PREFILL_ROWS` unset, serial K0 text requests on an NVIDIA B200 use a
1,024-row cap for prompts of at least 512 encoded tokens. Other requests use a
128-row cap. See [serial prefill](docs/text-serving.md#serial-prefill) for overrides
and workspace fallback. Tensor Core prefill accumulates FP32 sums in a different
order. Model scores and generated token IDs can differ, including when the highest
scores are far apart.

On the native SM100 build, Tensor Core prefill selects
[paged MXFP8 flash attention](src/mxfp8_gqa_prefill.cu#L751)<!-- prefix: "__global__ v" -->
for 144–1,023 query rows through 32,768 visible tokens and 1,024 query rows
through 131,072 visible tokens. Flash requires 16-byte-aligned K and V payloads
and sufficient dynamic shared memory. It uses BF16 Tensor Core products and FP32
online softmax. Unsafe inputs use ordered repair for affected rows and query-head
pairs. Logits and output token IDs can differ from ordered attention. The `exact`
policy continues to use its original attention path.

The ordered fallback shares decoded K and V across two query heads. It uses
two-row tiles for 64–127 rows and four-row tiles for at least 128 rows, with at
most 32,768 visible tokens. Leftover rows, shorter chunks, and chunks beyond the
bound use protected single-row attention. Ordered attention keeps the FP32 score
reduction tree, causal token order, BF16 output rounding, and selective FP64 recovery.

The selection applies to the model for its lifetime. For C++, pass
`sglang::Fp8PrefillPolicy::tensor_core` as the fourth argument to
`Decoder(checkpoint, device, enable_mtp, policy)`.

Pass `--mtp-steps 1` through `--mtp-steps 4` to enable multi-token prediction
(MTP) for text generation. The generator reports `mtp_drafted_tokens`, `mtp_accepted_tokens`, and
`mtp_rounds` for the first measured request. Accepted tokens count emitted proposals
that match the target. Corrections, bonus tokens, and the first token are
excluded from that count. Zero steps selects the default decoder.

MTP prefill processes up to 128 prompt positions per layer using exact FP8 projections
with either prefill policy. Each final-normalized target row with a following prompt
token seeds the draft KV cache with that token ID, including across chunk boundaries.
A one-token prompt or an unavailable prefill workspace uses the one-row path.

During verification, each target layer processes up to five positions per round.
The verifier compares proposals with the target's greedy selections or seeded draws,
accepts matches in order, and emits the target correction at the first mismatch.
If all proposals match, it can emit a bonus token within the stop rule and budget.

Target KV follows the committed prefix. The verifier retains one additional
recurrent state bank and replays the consumed rows after a partial commit.
Draft KV rewinds and refreshes from the committed target rows.

To enable MTP from C++, construct `Decoder(checkpoint, device, true)` and set
`DecoderOptions::mtp_steps` for the request. The constructor loads the draft
weights once and each MTP request owns its draft cache.

## Serve token IDs over HTTP

From the project root, enter the default Pixi environment and build `sglang-server`.
Set `GPU_UUID` to the UUID of a reserved B200 before starting the server.

```bash
pixi shell --manifest-path /data1/shared/dongwu.chen/pixi.toml --environment default
xmake f -P . -o build -p linux -a x86_64 -m release
xmake build -P . sglang-server
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    ./build/bin/sglang-server --device 0 --port 17384
```

The server loads one model and binds to `127.0.0.1`. `GET /health` returns the effective
`max_active` and `max_num_batched_tokens` after model loading. The defaults are 32 active
requests and 32,768 target rows per scheduler step. Set `--max-active 1` to use the serial
decode graph with prefix caching disabled.
Use `GET /ready` to check whether the inference worker can serve requests.
It returns HTTP 200 with `{"ready":true}` while the worker is available. After a terminal
worker failure, it returns HTTP 503 with `{"ready":false,"reason":"worker_failed"}`.
`GET /health` continues to report liveness and the configured limits.
Extra connections wait in the listener backlog until a socket worker is available.
Both serial and batched target-only requests use the selected FP8 prefill policy.
Set `--max-num-batched-tokens` from `1` through `32768` to lower the step budget. It covers
prompt rows, ordinary decode rows, and MTP target verification rows across all scheduled requests.
One step can contain several backend tiles per request and reuses a single tile workspace.
Ordinary text uses tiles of up to 1,024 rows. Media, MTP, and tensor-core prefix caching
use tiles of up to 128 rows.

The batch service reserves KV cache space for the full accepted output limit before it sends tokens.
A request beyond total resident capacity gets HTTP 413. Temporary memory pressure returns HTTP 503.
If the CUDA stream can be drained after an admission allocation fails, the worker retires that
request and continues serving its peers.
A failed drain ends the affected requests. New completions return HTTP 500 until the server restarts.
[`DecoderCapacityError`](include/sglang/decoder.h#L61)<!-- prefix: "class Decode" -->
reports whether the caller can retry.

With `--batch-cuda-graphs`, the server warms each eligible decode-only width
(one through four) before capture. Singletons can replay through 5,120 visible
tokens. From zero-based position 5,120, widths two through four allocate a
request-owned ordered score workspace sized to KV capacity. Packed paged
attention at 8,192–16,384 visible tokens skips it. Longer singletons,
workspace-dependent steps after allocation failure, other widths, and mixed
prefill and decode steps run eagerly.

With `--fp8-prefill tensor-core`, eager forwards wider than four rows use
[split paged attention](src/mxfp8_gqa.cu#L967)<!-- prefix: "cudaError_t " -->
for K0 text decode at 32,768–262,144 visible tokens, reordering softmax and
weighted value reductions.

The default `--mtp-steps 0` selects target-only decoding.
Use `--mtp-steps 1` through `--mtp-steps 4` for unconstrained greedy and sampled text requests.
MTP supports `--max-active` values from `1` through `32`.
`--mtp-k` is an alias for `--mtp-steps`. Batched requests share target projections
within groups of up to four requests and commit their outputs independently.
With `--batch-cuda-graphs`, a text-only K1 group of one request can replay
two target verification rows when both positions are below 5,120. The first
eligible round runs eagerly. Draft, acceptance, sampling, and commit still
run outside the graph. Other MTP verification shapes run eagerly.
All groups in a step share the same target-row budget. SSE events and usage fields
follow the same contract at every depth.

Set `--eager` to run serial requests without graph capture or replay.
This option requires `--max-active 1` and `--mtp-steps 0`.

Set `--prefix-cache-entries` from `1` through `4` to enable
[prefix reuse](docs/prefix-cache.md). The default is `0`.
Keep `--mtp-steps 0` and omit `--eager`. Exact prefill remains the default.
Use `--fp8-prefill tensor-core` to opt in to [page-aligned prefill](docs/prefix-cache.md#tensor-core-scheduling).
Prefix reuse selects the batch route even at `--max-active 1`.
Tensor-core cache requests keep fixed prompt tile boundaries across scheduler turns.
Their full tiles use 128 rows, or the largest power of two within a smaller step budget.

For model `benchmark`, send a token ID array to `POST /v1/completions` using the
[frozen streaming benchmark contract](benchmarks/streaming/README.md#wire-contract).
Each selected token produces one SSE event. The first choice carries the prompt IDs.
To stop on a specific output ID, include a single ID in `stop_token_ids`.
The server emits the matching token and reports `stop`, even at the output limit.
Otherwise, reaching the output limit reports `length`. Usage and `[DONE]` follow the
finish. A detected connection reset cancels the associated request and releases its slot.

Add `--trace-batches` to log `batch width=… decode=… request_bytes=…` on standard error.
In serial mode the trace describes token-producing steps and reports zero batch request
bytes. In batch mode `request_bytes` counts live allocations owned by the batch executor.
The `batch target_rows=… step_budget=…` line records the charged work for the batch step.
`batch graph_slots=…` gives the bit mask of slots replayed by K0 decode graphs.
`DecoderBatchStepStats::verification_graph_slots` gives the K1 verifier replay mask.
When the batch allocates sampling scratch, `sampling_workspace_bytes=…` reports its size.

`Decoder::generate` accepts optional callbacks. `on_token` receives each selected ID
synchronously after host readback. Returning `false` cancels the request with that ID
in the result. `is_cancelled` is checked before request work and between model steps.

### Submit a text prompt

Send a nonempty UTF-8 `prompt` string with model `Qwen3.8-27B-FP8`.
The prompt limit is 1 MiB. The encoded prompt and output budget share a 262,144-token context.
The first text request loads the [pinned tokenizer](tools/unicode/README.md) from the checkpoint.
The server completes the raw prompt with no chat formatting or automatic BOS or EOS insertion.

```bash
curl -N http://127.0.0.1:17384/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-FP8","prompt":"Explain why the sky is blue.","max_tokens":64,"stream":true}'
```

With `stream: true`, append the valid UTF-8 `choices[0].text` deltas in order until `[DONE]`.
Omit `stream` to receive one JSON response with complete text and usage.
The default `max_tokens` is 16.
See [text completion options and verification](docs/text-serving.md) for prompt and response details.

### Submit a chat conversation

Send `messages` with model `Qwen3.8-27B-FP8` to `POST /v1/chat/completions`.
The server renders the pinned checkpoint template and sends its prompt through the
same serial or batched decoder as raw text and token-ID requests.

```bash
curl -N http://127.0.0.1:17384/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-FP8","messages":[{"role":"user","content":"Explain why the sky is blue."}],"stream":true}'
```

Chat accepts inline decoded RGB8 images and video. Media requests use eager target execution
and bypass prefix reuse.
See [image and video inputs](docs/chat-serving.md#image-and-video-inputs) for the attachment format.
An opt-in build with libpng and FFmpeg also accepts inline PNG images and MP4 video.
See [codec build and verification](docs/chat-serving.md#codec-checks) for setup and checks.

### Sample output tokens

Raw string completions default to `temperature: 1`, `top_p: 1`, `top_k: -1`, and `min_p: 0`.
Chat defaults to `--sampling-defaults model` and reads the checkpoint's `generation_config.json`.
This checkpoint uses `temperature: 1`, `top_p: 0.95`, `top_k: 20`, and `min_p: 0`.
Use `--sampling-defaults openai` to give chat the raw-text sampling defaults.
Explicit sampling controls override the route defaults.
If a chat request omits its output limit, the server uses the final prompt length to set the budget.
Batch admission can lower that implicit budget to fit the worker's resident capacity.
See [chat sampling and output limits](docs/chat-serving.md#sampling-defaults-and-output-budget).
The [chat defaults loader](apps/server/text_request.cpp#L408)<!-- prefix: "const Decode" -->
and [output limit calculation](apps/server/text_request.cpp#L246)<!-- prefix: "void finish_" -->
implement these defaults.

Set a finite, nonnegative temperature.
A temperature below `1e-6` or `top_k: 1` selects the greedy path.
Set `top_p` in `(0, 1]` to limit the nucleus.
Set `top_k` to a positive integer to limit the number of candidates.
The server caps positive values at the vocabulary size.
Use `-1` to let top-p act on the full vocabulary.
Set `min_p` in `[0, 1]` to retain candidates with at least that fraction of the
largest probability. A value of `0` disables this cutoff.

Pass a signed 64-bit integer `seed` to repeat the draws for the same logits.
When you omit `seed`, the decoder selects one seed for the request.
Unconstrained sampled text requests use the configured MTP depth.
The generator defaults to greedy selection. Use `--temperature 0` to request it explicitly.
The generator also accepts `--top-p`, `--top-k`, `--min-p`, and `--seed`.
The token-ID benchmark client explicitly requests greedy selection.

C++ callers set the same fields in
[`DecoderOptions`](include/sglang/decoder.h#L24)<!-- prefix: "struct Decod" -->.
See the [sampling contract](docs/text-serving.md#sampling) for dispatch and seed behavior.

### Constrain generated output

Add one of these fields to a native `/v1/completions` or text-only `/v1/chat/completions`
request. Constrained requests use target-only decoding (K0).

- `{"ebnf":"root ::= \"yes\" | \"no\""}` restricts output to one listed literal.
  See the [finite-choice contract](docs/text-serving.md#finite-choice-constraints).
- `{"regex":"[A-Z]{2}-[0-9]{3}"}` matches a whole output such as `AB-123`.
  A pattern takes 1–1,024 UTF-8 bytes, with explicit repeat bounds up to 64.
  See the [regular expression contract](docs/output-constraints.md#regular-expressions).
- `response_format` with `type: "json_schema"` generates JSON in the response text.
  The supported subset allows up to eight properties per object and eight items per array.

For example, add this field to require a JSON object with a `status` drawn from two values.

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "schema": {
        "type": "object",
        "properties": {"status": {"enum": ["open", "closed"]}},
        "required": ["status"],
        "additionalProperties": false
      }
    }
  }
}
```

See the [JSON Schema contract](docs/output-constraints.md#json-schema) for supported keywords,
token budgets, and finish rules.

## Continue development

The checkpoint lives at `/data0/shared/Qwen3.8-27B-FP8`. Keep its weights and generated
artifacts outside version control. Use SGLang as the architecture reference and the pinned
vLLM installation as the independent performance comparator.

- [Isolated vLLM setup and single-B200 baseline](benchmarks/vllm/README.md) reproduces
  the historical cross-engine token ID diagnostic. Graph token parity uses the published
  native eager decoder as its reference. Baseline vLLM `81a8cb063e62` reached
  **25.89 output tokens/s** and **309.06 ms** median request latency on this serial fixture
  (five prompt tokens, eight generated tokens, two warmups, seven measured requests).
  The W1–W6 serving comparison remains open.
- [Checkpoint and SGLang source map](docs/research/checkpoint.md) identifies the upstream
  revision and records content verification for all 81 files and 1,606 indexed tensors.
- [Development milestones](docs/development.md) define the path from the runtime probe to
  a decoder that matches reference tokens, then to serving and single-card optimization.
- [Benchmark agreement](docs/performance.md) specifies the vLLM commit, the small greedy
  fixture, and the throughput, latency, and correctness evidence for W1–W6.

Start a maintenance session by explicitly invoking `.agents/skills/main-mode/SKILL.md`
with the milestone you want to advance. Main assigns the repository's local research,
coding, test, and documentation roles. See the [repository instructions](AGENTS.md).

For a fresh checkout on this host, install the gate dependencies and enable the local hooks.
Run commits from the managed default environment so the hooks use its Python.

```bash
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890
pixi shell --manifest-path /data1/shared/dongwu.chen/pixi.toml --environment default
python -m pip install -r .agents/checks/requirements.txt
git config --local core.hooksPath .githooks
git config --local http.https://github.com/.proxy http://127.0.0.1:7890
python -IB .agents/checks/check_rules.py check .
python -IB .agents/checks/check_rules.py package .agents/skills --policy-root .
git hook run pre-commit
```

To verify the handoff independently, run the installed hooks in disposable clones and save
the observations. The proof checks staged and unstaged disagreements, an alternate index,
the frozen size limits, test failures, and the XMake compile gate.

```bash
mkdir -p artifacts/handoff
python -B tests/prove_handoff.py --revision HEAD > artifacts/handoff/verification.jsonl
tail -1 artifacts/handoff/verification.jsonl
```

Push each verified commit to the private `origin` and compare `git rev-parse HEAD` with
`git ls-remote origin refs/heads/main`. The hardware run and performance measurements are
separate from the staged build and host checks.
