# Single-B200 development contract

Build a C++ and CUDA C++ inference service for one NVIDIA B200. The local checkpoint at `/data0/shared/Qwen3.8-27B-FP8` matches published revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`. The [checkpoint investigation](research/checkpoint.md) records the content hashes, tensor inventory, and source paths in SGLang.

## Scope at adoption

The first executable check compiled a native `sm_100` cubin with XMake, GCC 15.3.0, and CUDA 13.4.92. On physical B200 4 (`GPU-57c04c1a-9ae4-b231-f31a-5f4367e759a8`), the kernel returned the expected value of 42. Model execution, service compatibility, and throughput each need their own acceptance evidence.

| Part | Responsibility | Source |
| --- | --- | --- |
| Model | Define weights, mathematics, tokenization, media processing, and native multi-token prediction (MTP). | Official checkpoint, configuration, processor, tokenizer, and model code pinned in P0. |
| Service | Study scheduling, Radix prefix reuse, memory ownership, speculative verification, constraints, and streaming behavior. | Architectural and behavioral reference `sgl-project/sglang` at `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. |
| Performance | Reproduce a run independently and compare the candidate on each agreed workload. | The [benchmark agreement](performance.md) owns the vLLM pin, controls, and metrics. |
| Candidate | Build the C++ runtime and CUDA C++ operators in `src/` with XMake and default Pixi. Keep all headers under `include/sglang/`, with server headers in `include/sglang/server/`. Use `sglang/...` includes from `include/`. | [Build options](../tools/build.sh#L5)<!-- prefix: "echo \"Usage:" --> and [XMake project](../xmake.lua#L2)<!-- prefix: "set_project(" -->. |

**Adopted scope.** Target end-to-end inference on one B200 with the local checkpoint. Use project-owned model and GPU operators with the CUDA Toolkit. See the [native decoder](#p3-native-text-decoder) for its checkpoint parser dependency. Eight-B200 expansion remains outside acceptance.

**Operator policy from the project discussion.** Keep cuBLAS, cuBLASLt, cuDNN, CUTLASS, FlashInfer, and equivalent operator libraries or generators outside production execution. Justify each HTTP, JSON, tokenizer, media decoder, and system library. Record its purpose, license, version, and execution path. Reference environments may use their own operators.

**Proposed product scope from the discussion.** Cover text generation, sampling, stopping, image and video inputs, native MTP4, and concurrent scheduling. Use MXFP8 KV for grouped-query attention (GQA). Investigate GDN recurrent state and Radix prefix reuse. Proposed service features include OpenAI-compatible endpoints, streaming, cancellation, and JSON Schema, regex, and grammar constraints. Verify checkpoint semantics and agree on service behavior before acceptance.

## First task and next evidence

From the repository root, run `bash tools/build.sh` to build the generator, libraries, and probes. For a device run, set `GPU_INDEX` to the reserved B200's physical index.

```bash
bash tools/build.sh --gpu "${GPU_INDEX:?set the reserved B200 index}"
```

The build checks the `sm_100` image and CUDA runtime. The ahead-of-time probe must print `GPU returned 42 (expected 42)` and exit successfully. Keep the command, device UUID, toolchain identities, and binary hash.

For P0, use the [checkpoint investigation](research/checkpoint.md) for model semantics and pinned SGLang source mappings. Propose endpoints, API version, parameters, streaming events, and constraint semantics. Record checkpoint conflicts for scope review.

The [baseline smoke procedure](performance.md#first-baseline-run) produced the same eight greedy IDs on seven vLLM requests with BF16 KV and MTP disabled. Keep that continuation as a historical diagnostic. Use the [GQA cache comparison](performance.md#gqa-cache-comparison) for paired runs.

### P1 runtime probe

Use the managed default Pixi toolchain and reserve one B200. From the repository root, set `GPU_INDEX` to its physical index and run:

```bash
bash tools/build.sh --build-dir build --nvrtc --gpu "${GPU_INDEX:?set the reserved B200 index}"
```

The script checks that NVRTC resolves inside Pixi and that `libcuda.so.1` resolves outside Pixi. It exposes only the selected GPU by UUID. Check that the probe reports that UUID, `compute_capability=10.0`, `architecture=sm_100`, and the device allocation size.

The [probe](../tools/nvrtc-probe.cpp#L265)<!-- prefix: "void run_pro" --> uses one CUDA context for both launches. The cold lookup compiles a native cubin and loads its function. The warm lookup reuses that function in the same context. The probe disables NVRTC's persistent cache.

Expect these results in order:

- `phase=cold cache=miss compiled=1` with `compile_ms` for the NVRTC compilation call.
- `phase=warm cache=hit compiled=0` with `compile_ms=0.000000`.
- `NVRTC probe passed` followed by exit status zero.

Both phase lines must report `verified_elements=257 first=42 last=298`. The probe checks every element after each launch. The `kernel_ms` field measures each completed launch with CUDA events, after buffer reset and before the host copy.

Keep the command, full output, allocation size, and device, library, and binary identities. A missing phase, incorrect result, or nonzero exit fails this fixed-kernel and context-scoped cache check.

### P2 GQA cache contract

Store K and V as E4M3 with one E8M0 scale per 32 contiguous channels. Each 256-channel vector carries eight scales. The checkpoint has four KV heads. FP8 weights use BF16 inverse scales per 128×128 weight block. GDN uses separate convolution history and FP32 recurrent state. [Checkpoint investigation](research/checkpoint.md)

Allocate separate device buffers for K and V payloads and scales. Each 128-token physical page needs 131,072 payload bytes and 4,096 scale bytes for K, and the same for V. Map logical pages to distinct physical pages with the device page table. [Cache layout](../include/sglang/mxfp8_gqa.h#L22)<!-- prefix: "/// Separate" --> · [Page table](../include/sglang/mxfp8_gqa.h#L34)<!-- prefix: "struct Mxfp8" -->

Link the CUDA consumer to the static `sglang_cuda` target. Apply RoPE to K and Q before calling the cache writer and attention, respectively. Pass contiguous BF16 K and V tensors shaped `[tokens, 4, 256]` to `mxfp8_kv_append`. Once the required KV prefix is ready on the stream, call `mxfp8_gqa_attention` with the valid prefix length and first query position. Q and output are BF16 with shape `[queries, 24, 256]`. Query row `r` attends through position `first_query_position + r` inclusive. Check each launch status. Observe stream completion before releasing or reusing storage. [Library target](../xmake.lua#L46)<!-- prefix: "target(\"sgla" --> · [Append](../include/sglang/mxfp8_gqa.h#L48)<!-- prefix: "cudaError_t " --> · [Attention](../include/sglang/mxfp8_gqa.h#L55)<!-- prefix: "cudaError_t " -->

For each K or V group, the writer uses maximum magnitude `amax`. It stores `ceil(log2(max(amax, 1e-30) / 448)) + 127`, clamped to `[0, 254]`. The attention kernel decodes packed K and V to FP32 and converts BF16 Q to FP32 for the dot product. [Scale calculation](../src/mxfp8_gqa.cu#L34)<!-- prefix: "__device__ i" --> · [Attention kernel](../src/mxfp8_gqa.cu#L87)<!-- prefix: "__global__ v" -->

#### P2 GQA verification

Build the optional verifier with the managed Pixi toolchain from the repository root.

```bash
bash tools/build.sh --build-dir build --mxfp8-check
```

Set `BUILD_WORK_DIR` to the path printed after `Build directory:`. Set `GPU_UUID` to the reserved B200 UUID. Run the verifier with only that UUID visible.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    "${BUILD_WORK_DIR:?set the printed build directory}/build/bin/mxfp8-gqa-check"
```

Expect `compute_capability=10.0` and checks for prefix append, cross-page append, causal prefill, decode, and page reuse. The [verifier](../tests/mxfp8_gqa_check.cu#L450)<!-- prefix: "void run_che" --> compares K and V payloads and scale bytes with an independent encoder. It compares all BF16 outputs with an independent causal attention reference. Require `MXFP8 GQA check passed` and exit status zero.

Keep the invocation, device UUID, binary identity, and full check output. Record the executed path, effective Q dtype, scale policy, and test shapes. Measure operator timings separately.

### P3 native text decoder

#### Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| `sglang-generate` | Parse one token-ID prompt, run requests serially, and write one JSON result. | [CLI entry](../apps/sglang-generate.cpp#L227)<!-- prefix: "int main(int" --> |
| `sglang_checkpoint` | Validate the text configuration, index, and safetensors metadata. Map immutable shard bytes and expose checked tensor views. | [Loader API](../include/sglang/checkpoint.h#L22)<!-- prefix: "class Checkp" --> |
| `sglang_decoder` | Own device weights and run requests using operators from `sglang_cuda`. | [Decoder API](../include/sglang/decoder.h#L47)<!-- prefix: "class Decode" --> · [Layer dispatch](../src/decoder_forward.cpp#L171)<!-- prefix: "void Decoder" --> |

The loader uses LLVM Support 23.1.5 from Pixi (`llvmdev` and `libllvm23`, Apache-2.0 WITH LLVM-exception) only to parse checkpoint JSON. The loader owns the safetensors validation and mapping. [JSON parser](../src/checkpoint.cpp#L170)<!-- prefix: "json::Object" --> · [Link configuration](../xmake.lua#L60)<!-- prefix: "target(\"sgla" -->

#### Runtime path

Before upload, the decoder checks required tensor names, dtypes, and shapes. Upload retains checkpoint mappings and allocations through stream completion. Requests share device weights and retain references through completion, including deferred cleanup. FP8 projections use E4M3 weights and BF16 inverse scales. They quantize BF16 inputs to E4M3 with one FP32 scale per 128 channels. [Weight upload](../src/decoder_weights.cpp#L97)<!-- prefix: "Decoder::Mod" --> · [Stream cleanup](../src/decoder_resources.cpp#L169)<!-- prefix: "CleanupError" --> · [Activation quantization](../src/text_ops.cu#L121)<!-- prefix: "__global__ v" -->

With MTP off, serial generation reuses a resident stream, workspace, and caches. Larger requests replace undersized storage. Before prefill, the decoder zeros BF16 convolution history and FP32 recurrent state for 48 GDN layers. The 16 GQA layers rebuild the valid prefix in [MXFP8 pages](#p2-gqa-cache-contract) with an identity page table. [Generation loop](../src/decoder.cpp#L272)<!-- prefix: "void Decoder" --> · [State reset](../src/decoder_resources.cpp#L393)<!-- prefix: "void Request" -->

Prefill runs eagerly one prompt token at a time. Eager scoring selects the first output token. Later steps replay a CUDA graph that consumes the previous selection. The decoder captures the graph on the first decode step for that storage. An overlapping call uses private eager storage and closes it before returning. The host synchronizes before reading each selection. [Decode graph](../src/decoder_forward.cpp#L153)<!-- prefix: "void Decoder" -->

A single model position uses FP8 GEMV. The wide kernel uses four warps per block with input widths divisible by 512 and 16-byte-aligned FP8 buffers. With two to four lanes, eligible FP8 projections share weight loads in packed GEMM. Ineligible projections use separate GEMVs. Embedding, quantization, normalization, BF16 projections, attention, state updates, SwiGLU, and scoring run per lane. [Projection dispatch](../src/decoder_forward.cpp#L44)<!-- prefix: "void project" --> · [GEMV](../src/text_ops.cu#L502)<!-- prefix: "cudaError_t " --> · [GEMM](../src/text_ops.cu#L520)<!-- prefix: "cudaError_t " -->

#### Constraints and failure behavior

Pass one nonempty prompt with `--prompt-ids` or `--prompt-file`. Use decimal token IDs separated by commas or ASCII whitespace. Prompt and stop IDs must be in `[0, 248320)`. Prompt length plus output budget must fit 262,144 tokens. The CLI budget defaults to one. A zero budget returns no tokens with finish reason `length`. [CLI options](../apps/sglang-generate.cpp#L31)<!-- prefix: "void usage(s" --> · [Request validation](../src/decoder.cpp#L19)<!-- prefix: "void validat" -->

The decoder chooses the highest BF16 logit, with ties going to the lowest ID. A `--stop-token` match emits the token and returns `stop`, including at the output limit. Otherwise it returns `length`. The CLI applies no implicit EOS rule. Input, load, execution, or explicit cleanup errors produce a diagnostic on stderr and exit status 1. [Selection](../src/text_ops.cu#L384)<!-- prefix: "__device__ T" --> · [Generation loop](../src/decoder.cpp#L272)<!-- prefix: "void Decoder" -->

If cleanup cannot confirm completion, the decoder retains pending CUDA resources and transfer buffers. Model load and request setup retry recoverable cleanup on the same device. Persistent errors block setup. Terminal errors and ambiguous frees quarantine resources until process exit. [Deferred cleanup](../src/decoder_resources.cpp#L103)<!-- prefix: "static void " --> · [Destructor diagnostics](../src/decoder_resources.cpp#L284)<!-- prefix: "Allocations:" -->

Use `--top-logits N` to report up to `N` raw scores per output token, ordered by descending score then ascending ID. The range is 0 to 248,320 and the default is 0. [Logit output](../apps/sglang-generate.cpp#L193)<!-- prefix: "void write_r" -->

#### Verification

Freeze independent reference inputs and precision-aware tolerances before comparison. Check tensor names, shapes, dtypes, and scales. Compare layer outputs and state on model shapes. Require exact metadata and specified packed bytes.

Use the [P2 verifier](#p2-gqa-verification) for MXFP8 encoding, causal attention, and page reuse. Check KV positions and page ownership through append and decode. Check BF16 convolution history and FP32 recurrent state through GDN transitions. Compare supported cache restores and continuations with fresh native execution under the same precision policy. [Independent GDN check](../tests/decoder/gdn_check.cu#L239)<!-- prefix: "void verify_" -->

Follow the [host build procedure](../README.md#build-on-the-project-host). Set `SGLANG_GENERATE` to the printed generator. Reserve a B200 with space for weights and request state. Set `GPU_UUID` to its UUID. The process sees that UUID as CUDA ordinal 0. The default checkpoint is `/data0/shared/Qwen3.8-27B-FP8`. [Generator build](../tools/build-probe.sh#L76)<!-- prefix: "\"$xmake\" bui" -->

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    "${SGLANG_GENERATE:?set the printed generator path}" \
    --device 0 --prompt-ids 760,6511,314,9338,369 --max-new-tokens 8 --repeats 2
```

Check the echoed prompt, eight in-range output IDs, and finish reason `length`. Require identical native IDs and finish reasons across both runs. Set the budget to one and `--stop-token` to the first native ID. Require one output ID and `stop`. With a zero budget, require no output IDs and `length`. To check state reset, repeat the prompt after another request on the same decoder.

Check greedy selection against the BF16 maximum and the lowest token ID at a tie. Compare captured logits with independent calculations. Use `--top-logits 248320` on a short request to check vocabulary shape and ranking. [Request ranking check](../tests/decoder/request_check.cpp#L171)<!-- prefix: "void check_r" -->

The [FP8 fixture](../benchmarks/vllm/small-fp8.json#L2)<!-- prefix: "\"prompt_toke" --> supplies the short prompt. Use the [vLLM IDs](../benchmarks/vllm/small-fp8-result.json#L20)<!-- prefix: "\"output_ids\"" --> and frozen raw logits for historical diagnostics. Capture identity is in `artifacts/native-decoder-verification/final-evidence.sha256`. The [historical verifier](../tests/decoder/run_native.py#L117)<!-- prefix: "def validate" --> applies the former cross-engine checks.

#### Serial greedy MTP

Set `--mtp-steps` to `1`–`4` in either binary to cap proposals per round. The default `0` preserves ordinary generation. Positive steps apply server-wide and require `--max-active 1`. MTP loads draft weights once and shares the target embedding and LM head. Positive-budget requests own eager target state and draft MXFP8 KV. [Weights](../src/decoder.cpp#L237)<!-- prefix: "Decoder::Dec" --> · [Request state](../src/decoder.cpp#L252)<!-- prefix: "DecoderFinis" -->

The target selects the first output in prefill. At draft position `p`, the draft uses the final-normalized target hidden row and the shifted token at `p+1`. Later proposals use the preceding draft's final-normalized row and predicted token. [Prefill](../src/decoder.cpp#L199)<!-- prefix: "DecoderFinis" --> · [Proposals](../src/decoder.cpp#L157)<!-- prefix: "bool propose" -->

The reference verifier checks correctness one eager target position at a time. It commits matches in order and ends the round at the first mismatch with a target correction. If all proposals match, it emits a target bonus when the stop rule and budget allow. The remaining budget caps proposals. Target KV and GDN advance only on committed inputs. The verifier rewinds draft KV and rebuilds its prefix from committed target rows. [Verification loop](../src/decoder.cpp#L177)<!-- prefix: "std::optiona" --> · [Draft rewind](../src/decoder_mtp.cpp#L321)<!-- prefix: "void MtpDraf" -->

SGLang-style parallel target verification, tentative target-state rollback, and speed qualification remain pending. The [MTP measurement contract](performance.md#native-serial-mtp) defines the counters.

Build both release binaries. Run the [MTP acceptance runner](../tests/mtp/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in re" --> on the reserved B200.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    /data1/shared/dongwu.chen/.pixi/envs/default/bin/python tests/mtp/run_native.py \
    --generate build/bin/sglang-generate --server build/bin/sglang-server \
    --evidence artifacts/native-mtp-check
```

Match MTP-off IDs, finish reasons, and top-64 target logits. Compare default and explicit `0`. Check short budgets, stop boundaries, MXFP8 page crossing, and counter bounds. Require exact serial SSE tokens, finish, and usage. Check rejection of batched MTP. Compare committed target state with an eager control. Verify zero, partial, and full acceptance, draft replay, and cancellation. Retain identities, fixtures, and raw results.

### Native streaming server

Follow the [README launch procedure](../README.md#serve-token-ids-over-http). With MTP off, the default `--max-active 1` uses the serial graph path. Set `--max-active 2`, `3`, or `4` for batching. Extra connections wait in the listener backlog. [Serving modes](../apps/sglang-server.cpp#L172)<!-- prefix: "if (argument" --> · [Socket workers](../apps/sglang-server.cpp#L115)<!-- prefix: "void serve_b" -->

Batch lanes share weights and have private MXFP8 KV, GDN state, and workspace. For positive budgets, admission reserves KV for prompt length plus budget minus one position. Each eager step advances one position per ready lane and can interleave prefill and decode. The final prompt position selects the first output token. Readback and retirement follow stream completion. [Admission](../src/decoder_batch.cpp#L95)<!-- prefix: "void Decoder" --> · [Batch step](../src/decoder_batch.cpp#L143)<!-- prefix: "Steps Decode" -->

In batch mode, `--trace-batches` logs `width` for selected lanes, including zero-budget requests. `decode` counts lanes consuming a previous output. Full decode at capacity gives `width=2 decode=2` or `width=4 decode=4`. `request_bytes` counts live batch request device allocations after the step. [Trace](../apps/server/batch_service.cpp#L169)<!-- prefix: "void BatchSe" -->

Follow the [wire contract](../benchmarks/streaming/README.md#wire-contract) for fixed-length requests. Each token gets one event. Optional `stop_token_ids` takes one ID. A match emits that token and finishes with `stop`, including at the budget. Zero budget finishes with `length` without a model step. The [decoder limits](#constraints-and-failure-behavior) apply. HTTP limits are 8 KiB for headers and 2 MiB for the body. [Request parser](../apps/server/completions.cpp#L29)<!-- prefix: "CompletionRe" --> · [HTTP limits](../apps/server/http.cpp#L23)<!-- prefix: "constexpr st" -->

A reset or failed token write cancels the affected request. The batch retires its storage before slot reuse. A write-blocked lane waits while ready peers can advance. Clients can use `shutdown(SHUT_WR)` after sending a complete request. [Cancellation](../apps/server/batch_service.cpp#L120)<!-- prefix: "bool BatchSe" --> · [Disconnect check](../apps/server/http.cpp#L263)<!-- prefix: "bool disconn" -->

#### Streaming verification

Build the release server with the README procedure. Set `GPU_UUID` to the reserved B200 UUID. Put `nsys` on `PATH` for graph checks. Use a new evidence directory.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    /data1/shared/dongwu.chen/.pixi/envs/default/bin/python tests/batch/run_native.py \
    --binary build/bin/sglang-server --evidence artifacts/native-batch-check
```

Require `passed` in `status.json` for each mode. Check serial oracle streams and one graph launch per output after the first. At 2 and 4, check exact streams, full decode width on mixed-stop, and full scheduler width on page crossing. Confirm extra connections wait. After HTTP reset, verify readable peer streams, exact replacements, and zero request bytes. The independent core probe checks post-cancellation GPU advancement of retained lanes. Keep [runner](../tests/batch/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in li" --> identity and raw results. Run the serial [streaming smoke](../tests/test_native_streaming.py#L178)<!-- prefix: "def main():" --> with `--port PORT` for zero budget and half-close.

## Runtime and state ownership

**Proposed cache and MTP checks.** Define page and checkpoint token intervals. Share only compatible committed prefixes. Restore the last valid GDN checkpoint and replay the uncovered suffix. Discard rejected effects before publishing shared state. After cancellation, eviction, or allocation failure, verify GPU completion before storage reuse.

**Proposed checkpoint policy.** Save GDN state every 8,192 processed tokens and at the latest session request end. Measure checkpoint bytes and replay cost before adoption. Verify GDN tensors and MTP semantics against the checkpoint. Verify MXFP8 encoding and scale layout in P2.

## Runnable milestones

The discussion proposes these stages after adoption. Each row gives a test and the evidence needed to close it if approved. Preserve the exact invocation, inputs, expected result, and actual output with each milestone.

| Stage | Run | Accept when |
| --- | --- | --- |
| P0 · Model and baseline | Inspect the pinned tensors and configuration. Run one fixed greedy text request through isolated vLLM on the B200. | The manifest resolves the model identity. The request reproduces. Record the single-card memory budget, reference map, dependency decisions, and smallest experiment for each open risk. |
| P1 · Runtime | Run the [P1 runtime probe](#p1-runtime-probe) on one B200. | The cold miss and warm hit each complete and verify the full output. Retain the device, build, cache, allocation, and timing evidence. |
| P2 · Operators and one layer | Run FP8 GEMM, normalization, RoPE, [MXFP8 GQA](#p2-gqa-cache-contract), GDN, FFN, and sampling on verified model shapes. Exercise each applicable layer type. | Operator and layer outputs meet prespecified independent reference tolerances. Capture state updates and per-shape timings. |
| P3 · Text generation | Run the [native text decoder checks](#verification) with the real checkpoint. | Independent math checks pass under the native precision policy. Greedy selection, deterministic repeats, state, counts, and finish behavior meet the native contract. Record output tokens/s, latency, and peak memory. |
| P4 · Cache and MTP | Compare fresh execution with prefix hits, restore and replay, and native MTP on and off. Exercise zero, partial, and full acceptance, cancellation, eviction, and reuse. | Committed state and continuation agree with fresh target execution. Sampling preserves the target distribution. Rejected tokens never enter a shared prefix. Released storage is safe to reuse. |
| P5 · Serving and media | Use an independent client for text, image, and video requests. Exercise streams, all constraint modes, continuous batching, chunked prefill, cancellation, and a sustained concurrent run. | Verify the agreed API fixtures, media inputs and outputs, stop reasons, valid constrained outputs, and cleanup. Check for leaks and stale state across requests. |
| P6 · End-to-end performance | Profile a named workload. Change the limiting path. Repeat the baseline and candidate runs under the benchmark agreement. | Each required row meets its agreed throughput and latency targets and passes correctness. Retain raw samples, before/after profiles, and the decision to keep or revert the change. |

The proposed P0 time budget is three to five working days. Open questions need an owner and a discriminating experiment. Record raw runs in ignored `artifacts/` storage. Keep accepted decisions and reproducible procedures in the maintained documents.
