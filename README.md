# sglang.cpp

A C++ and CUDA C++ inference engine in development for **Qwen3.8-27B-FP8 on one NVIDIA B200**.
The native decoder loads the official text weights and runs all 64 layers for greedy
token-ID generation. It uses paged MXFP8 KV storage in the 16 full-attention layers
and carries Gated DeltaNet state through the other 48 layers. Prefill and the first
greedy selection run eagerly. Subsequent serial decode steps use a CUDA graph.

The reusable runtime API is in
[`include/sglang/decoder.h`](include/sglang/decoder.h#L52)<!-- prefix: "class Decode" -->.
The paged KV-cache and causal grouped-query attention API is in
[`include/sglang/mxfp8_gqa.h`](include/sglang/mxfp8_gqa.h#L48)<!-- prefix: "cudaError_t " -->.
The CUDA implementation is in
[`src/mxfp8_gqa.cu`](src/mxfp8_gqa.cu#L35)<!-- prefix: "__device__ i" -->.

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
The model retains one set of request buffers for serial reuse. The first subsequent
decode step captures the graph on demand, and later steps and requests replay it while
capacity suffices. Overlapping calls that cannot acquire the serial slot use private
request buffers and eager decode.

The generator and HTTP server accept `--fp8-prefill exact|tensor-core`. The default
`exact` uses the exact FP8 projection path. `tensor-core` uses SM100 Tensor Cores for
eligible FP8 projections in chunks of 3 to 128 prompt positions. Smaller chunks and
ineligible projections use the exact path. Tensor Core prefill accumulates FP32 sums in
a different order. Model scores and generated token IDs can differ, including when
the highest scores are far apart.

The selection applies to the model for its lifetime. For C++, pass
`sglang::Fp8PrefillPolicy::tensor_core` as the fourth argument to
`Decoder(checkpoint, device, enable_mtp, policy)`.

Pass `--mtp-steps 1` through `--mtp-steps 4` to enable greedy multi-token prediction
(MTP). The generator reports `mtp_drafted_tokens`, `mtp_accepted_tokens`, and
`mtp_rounds` for the first measured request. Accepted tokens count emitted proposals
that match the target. Corrections, bonus tokens, and the first greedy token are
excluded from that count. Zero steps selects the default decoder.

MTP prompt seeding uses row-wise target forwards with either prefill policy.
The draft uses final-normalized target hidden rows and shifted token IDs. Each
round verifies proposals with ordered eager target forwards, stops at the first
mismatch, and emits the target correction. If all proposals match and output
budget remains, the target emits a bonus token. Target KV and recurrent state
advance along the committed sequence. Draft KV is rewound and extended from the
committed target rows, using the correction or bonus as the shift for the next
round. This implementation adds draft work and per-row
synchronization to ordinary eager target forwards.

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

The server loads one model and binds to `127.0.0.1`. `GET /health` returns HTTP 200 after
model loading. `POST /v1/completions` implements the frozen streaming benchmark contract
for model `benchmark` with token ID input. By default and with `--max-active 1`,
requests run serially and reuse the decode graph. Set `--max-active 2`, `3`, or `4`
to batch concurrent requests on the resident model. Extra connections wait in the
listener backlog until a socket worker is available.
The FP8 prefill policy applies to both serial and batched requests.

Add `--mtp-steps 1` through `--mtp-steps 4` to enable the same eager verifier for
serial HTTP requests with `--max-active 1`. The SSE events and usage fields are
the same as for ordinary serial generation.

Each selected token produces one SSE event. The first choice carries the prompt IDs.
To stop on a specific output ID, include a single ID in `stop_token_ids`.
The server emits the matching token and reports `stop`, even at the output limit.
Otherwise, reaching the output limit reports `length`. Usage and `[DONE]` follow the
finish. A detected connection reset cancels the associated request and releases its slot.

Add `--trace-batches` to log `batch width=… decode=… request_bytes=…` on standard error.
In serial mode the trace describes token-producing steps and reports zero batch request
bytes. In batch mode `request_bytes` counts live allocations owned by the batch executor.

`Decoder::generate` accepts optional callbacks. `on_token` receives each selected ID
synchronously after host readback. Returning `false` cancels the request with that ID
in the result. `is_cancelled` is checked before request work and between model steps.

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
