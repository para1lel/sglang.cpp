# Single-B200 development contract

Build a C++ and CUDA C++ inference service for one NVIDIA B200. `/data0/shared/Qwen3.8-27B-FP8` matches revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`. The [checkpoint investigation](research/checkpoint.md) records hashes, tensors, and SGLang source paths.

## Scope at adoption

The first `sm_100` cubin built with XMake, GCC 15.3.0, and CUDA 13.4.92. It returned 42 on B200 4 (`GPU-57c04c1a-9ae4-b231-f31a-5f4367e759a8`). Model, service, and throughput require separate evidence.

| Part | Responsibility | Source |
| --- | --- | --- |
| Model | Define weights, mathematics, tokenization, media processing, and native multi-token prediction (MTP). | Official checkpoint, configuration, processor, tokenizer, and model code pinned in P0. |
| Service | Study scheduling, Radix prefix reuse, memory ownership, speculative verification, constraints, and streaming behavior. | Architectural and behavioral reference `sgl-project/sglang` at `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. |
| Performance | Reproduce a run independently and compare the candidate on each agreed workload. | The [benchmark agreement](performance.md) owns the vLLM pin, controls, and metrics. |
| Candidate | Build the C++ runtime and CUDA C++ operators in `src/` with XMake and default Pixi. Keep all headers under `include/sglang/`, with server headers in `include/sglang/server/`. Use `sglang/...` includes from `include/`. | [Build options](../tools/build.sh#L5)<!-- prefix: "echo \"Usage:" --> and [XMake project](../xmake.lua#L2)<!-- prefix: "set_project(" -->. |

**Adopted scope.** Target one B200 with the local checkpoint and project-owned model and CUDA operators. See the [native decoder](#p3-native-text-decoder) for its parser dependency. Eight-B200 expansion remains outside acceptance.

**Operator policy.** Exclude cuBLAS, cuBLASLt, cuDNN, CUTLASS, FlashInfer, and equivalent operator libraries or generators from production execution. Justify HTTP, JSON, tokenizer, media decoder, and system libraries. Record purpose, license, version, and execution path. Reference engines may use their own operators.

**Proposed scope.** Cover text generation, sampling, stopping, image and video inputs, MTP4, and concurrent scheduling. Use MXFP8 KV for grouped-query attention (GQA). Investigate GDN state and Radix prefix reuse. Verify checkpoint and service semantics before accepting OpenAI-compatible endpoints, streaming, cancellation, and JSON Schema, regex, and grammar constraints.

## First task and next evidence

Run `bash tools/build.sh` from the repository root. For a device run, set `GPU_INDEX` to the reserved B200's physical index.

```bash
bash tools/build.sh --gpu "${GPU_INDEX:?set the reserved B200 index}"
```

The build checks the `sm_100` image and CUDA runtime. The ahead-of-time probe must print `GPU returned 42 (expected 42)` and exit successfully. Keep the command, device UUID, toolchain identities, and binary hash.

For P0, use the [checkpoint investigation](research/checkpoint.md) for model semantics and pinned SGLang source mappings. Propose endpoints, API version, parameters, streaming events, and constraint semantics. Record checkpoint conflicts for scope review.

The [baseline smoke procedure](performance.md#first-baseline-run) produced the same eight greedy IDs on seven vLLM requests with BF16 KV and MTP disabled. Keep that continuation as a historical diagnostic. Use the [GQA cache comparison](performance.md#gqa-cache-comparison) for paired runs.

### P1 runtime probe

Use the managed Pixi toolchain. Set `GPU_INDEX` to the reserved B200's physical index.

```bash
bash tools/build.sh --build-dir build --nvrtc --gpu "${GPU_INDEX:?set the reserved B200 index}"
```

The script checks NVRTC inside Pixi and `libcuda.so.1` outside Pixi. It exposes only the selected GPU. Check its UUID, `compute_capability=10.0`, `architecture=sm_100`, and the allocation size.

The [probe](../tools/nvrtc-probe.cpp#L265)<!-- prefix: "void run_pro" --> disables NVRTC's persistent cache. The cold lookup compiles a native cubin and loads its function. The warm lookup reuses the function in the same CUDA context.

Expect these results in order:

- `phase=cold cache=miss compiled=1` with `compile_ms` for the NVRTC compilation call.
- `phase=warm cache=hit compiled=0` with `compile_ms=0.000000`.
- `NVRTC probe passed` followed by exit status zero.

Both phases verify all elements and report `verified_elements=257 first=42 last=298`. CUDA events measure `kernel_ms` after buffer reset and before the host copy.

Keep the command, full output, allocation size, and device, library, and binary identities.

### P2 GQA cache contract

K and V use E4M3 for four 256-channel heads, with one E8M0 scale per 32 contiguous channels. E4M3 weights use BF16 inverse scales per 128×128 block. [Checkpoint investigation](research/checkpoint.md)

Separate K and V buffers use 131,072 payload bytes and 4,096 scale bytes per tensor per 128-token page. The device table maps logical pages to distinct physical pages. [Cache layout](../include/sglang/mxfp8_gqa.h#L22)<!-- prefix: "/// Separate" --> · [Page table](../include/sglang/mxfp8_gqa.h#L34)<!-- prefix: "struct Mxfp8" -->

Link `sglang_cuda`. Apply RoPE to K before appending contiguous BF16 `[tokens, 4, 256]` K and V. Apply RoPE to Q. Pass the ready KV prefix length and first query position to `mxfp8_gqa_attention`. BF16 Q and output are `[queries, 24, 256]`. Row `r` sees through `first_query_position + r` inclusive. Check launch status. Keep storage live through stream completion. [Library](../xmake.lua#L46)<!-- prefix: "target(\"sgla" --> · [Append](../include/sglang/mxfp8_gqa.h#L48)<!-- prefix: "cudaError_t " --> · [Attention](../include/sglang/mxfp8_gqa.h#L55)<!-- prefix: "cudaError_t " -->

For each K or V group's maximum magnitude `amax`, store `ceil(log2(max(amax, 1e-30) / 448)) + 127`, clamped to `[0, 254]`. Attention decodes K and V and converts BF16 Q to FP32. [Scale calculation](../src/mxfp8_gqa.cu#L35)<!-- prefix: "__device__ i" --> · [Direct attention](../src/mxfp8_gqa.cu#L88)<!-- prefix: "__global__ v" -->

Exact scoring preserves the direct FP32 group reductions, ordered sum of eight groups, and `1/16` scale. Replay computes `expf` weights in parallel in 32-token tiles. Running maxima, denominator FMAs, and value FMAs follow token order. FP32 division precedes one BF16 rounding. [Score](../src/mxfp8_gqa.cu#L158)<!-- prefix: "__global__ v" --> · [Replay](../src/mxfp8_gqa.cu#L201)<!-- prefix: "__global__ v" -->

#### P2 GQA verification

Build the verifier from the repository root with the managed Pixi toolchain.

```bash
bash tools/build.sh --build-dir build --mxfp8-check
```

Set `BUILD_WORK_DIR` to the printed build directory and `GPU_UUID` to the reserved B200 UUID.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    "${BUILD_WORK_DIR:?set the printed build directory}/build/bin/mxfp8-gqa-check"
```

Expect `compute_capability=10.0`, `MXFP8 GQA check passed`, and exit zero. The [verifier](../tests/mxfp8_gqa_check.cu#L616)<!-- prefix: "void run_che" --> compares append, prefill, decode, and page reuse against independent packed KV and BF16 references.

The [exact decode check](../tests/mxfp8_gqa_check.cu#L518)<!-- prefix: "void check_e" --> compares BF16 bits with direct attention across tile and page boundaries in eager launches and one reused graph. It checks input preservation, score bounds, and rejection of invalid arguments.

### P3 native text decoder

#### Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| `sglang-generate` | Parse one token-ID prompt, run requests serially, and write one JSON result. | [CLI entry](../apps/sglang-generate.cpp#L227)<!-- prefix: "int main(int" --> |
| `sglang_checkpoint` | Validate the text configuration, index, and safetensors metadata. Map immutable shard bytes and expose checked tensor views. | [Loader API](../include/sglang/checkpoint.h#L22)<!-- prefix: "class Checkp" --> |
| `sglang_decoder` | Own device weights and run requests using `sglang_cuda` operators. | [Decoder API](../include/sglang/decoder.h#L47)<!-- prefix: "class Decode" --> · [Layer dispatch](../src/decoder_forward.cpp#L221)<!-- prefix: "void Decoder" --> |

The loader uses Pixi's LLVM Support 23.1.5 (`llvmdev` and `libllvm23`, Apache-2.0 WITH LLVM-exception) only for JSON. [JSON parser](../src/checkpoint.cpp#L170)<!-- prefix: "json::Object" --> · [Link configuration](../xmake.lua#L60)<!-- prefix: "target(\"sgla" -->

#### Runtime path

The decoder validates tensor names, dtypes, and shapes before upload. Upload retains mappings and allocations until completion. Deferred cleanup retains shared weights. FP8 projections quantize BF16 inputs to E4M3 with one FP32 scale per 128 channels. [Weight upload](../src/decoder_weights.cpp#L97)<!-- prefix: "Decoder::Mod" --> · [Stream cleanup](../src/decoder_resources.cpp#L171)<!-- prefix: "CleanupError" --> · [Activation quantization](../src/text_ops.cu#L141)<!-- prefix: "__global__ v" -->

Serial MTP-off generation reuses its stream and grows storage on demand. Prefill rebuilds [MXFP8 pages](#p2-gqa-cache-contract) with an identity page table. Reset zeros BF16 convolution history and FP32 recurrent state across 48 GDN layers. [Generation loop](../src/decoder.cpp#L272)<!-- prefix: "void Decoder" --> · [State reset](../src/decoder_resources.cpp#L401)<!-- prefix: "void Request" -->

MTP-off prefill processes up to 128 consecutive tokens per layer, preserving row arithmetic, recurrent order, and causality. Single rows use row-wise prefill. The final row supplies initial logits. Serial decode replays a graph. Overlapping calls use private eager storage. [Prefill dispatch](../src/decoder_prefill.cpp#L313)<!-- prefix: "void Decoder" --> · [Decode graph](../src/decoder_forward.cpp#L203)<!-- prefix: "void Decoder" -->

Target prefill and MTP drafts use direct attention. Eager target decode selects exact attention at zero-based positions below score capacity. Graph capture selects exact attention when page-rounded KV slots fit that capacity. Exact graphs read device positions. Other decode uses direct attention. [GQA dispatch](../src/decoder_forward.cpp#L94)<!-- prefix: "cudaError_t " -->

The lane stream orders KV append, scoring, replay, and gating. Score capacity is the lesser of allocated KV slots and 5,120. The logits allocation holds 24 FP32 scores per slot until the LM head writes logits. [Workspace](../src/decoder_resources.cpp#L381)<!-- prefix: "ExactGqaWork" -->

Tuned FP8 projections require a forward group containing only one target decode row. This covers graph and eager decode, including MTP verification. A lone batch decode row can qualify when prefill chunks run separately. GQA combines Q/gate, K, and V in one launch. The MLP fusion rounds gate and up to BF16 before SwiGLU. GDN QKV and Z stay separate. [Lane selection](../src/decoder_forward.cpp#L34)<!-- prefix: "bool single_" --> · [Batch grouping](../src/decoder_batch.cpp#L186)<!-- prefix: "if (rows[slo" -->

Tuned GEMV requires 16-byte-aligned FP8 weights and activations and input widths of 5,120, 6,144, or 17,408. Both fusions require the same alignment and width 5,120. Q/gate, K, and V require 12,288, 1,024, and 1,024 output rows. Gate and up require 17,408 each. Other valid inputs use separate operations. Output spans must be disjoint from inputs and each other. Fused SwiGLU permits identical gate and up spans. [Tuned GEMV](../src/text_decode_fp8.cu#L276)<!-- prefix: "cudaError_t " --> · [Fused QKV](../src/text_decode_fp8.cu#L298)<!-- prefix: "cudaError_t " --> · [Fused SwiGLU](../src/text_decode_fp8.cu#L321)<!-- prefix: "cudaError_t " --> · [Buffers](../include/sglang/decoder_cuda.cuh#L49)<!-- prefix: "// Read-only" -->

Row-wise prefill and MTP drafts use GEMV. Eligible groups of two to four lanes use packed GEMM, with per-lane GEMV fallback. Wide GEMV and packed GEMM require 512-divisible input widths and 16-byte-aligned FP8 payloads. A general FP8 GEMM backend remains pending. [GEMV](../src/text_ops.cu#L651)<!-- prefix: "cudaError_t " --> · [GEMM](../src/text_ops.cu#L669)<!-- prefix: "cudaError_t " -->

Single-row BF16 RMSNorm uses vector loads at width 5,120 with 16-byte-aligned branch, gains, and optional residual. Other calls use generic kernels. [RMSNorm dispatch](../src/text_ops.cu#L822)<!-- prefix: "cudaError_t " -->

#### Constraints and failure behavior

Pass one nonempty prompt with `--prompt-ids` or `--prompt-file`. Separate decimal IDs with commas or ASCII whitespace. Prompt and stop IDs must be in `[0, 248320)`. Prompt length plus budget must fit 262,144 tokens. The default budget is one. Zero budget returns no tokens and `length`. [CLI options](../apps/sglang-generate.cpp#L31)<!-- prefix: "void usage(s" --> · [Request validation](../src/decoder.cpp#L19)<!-- prefix: "void validat" -->

Greedy selection takes the highest BF16 logit, breaking ties by lowest ID. `--stop-token` emits a match and returns `stop`, even at the budget. Budget exhaustion returns `length`. The CLI applies no implicit EOS. Input, load, execution, and explicit cleanup errors report to stderr and exit 1. [Selection](../src/text_ops.cu#L491)<!-- prefix: "__device__ T" --> · [Generation loop](../src/decoder.cpp#L272)<!-- prefix: "void Decoder" -->

The decoder retains CUDA resources and transfer buffers while completion is uncertain. Model and request setup retry cleanup on the same device. Persistent errors block setup. Terminal errors and ambiguous frees quarantine resources until exit. [Deferred cleanup](../src/decoder_resources.cpp#L105)<!-- prefix: "static void " --> · [Destructor diagnostics](../src/decoder_resources.cpp#L286)<!-- prefix: "Allocations:" -->

Multirow scratch uses per-device pools. Out-of-memory or unsupported pooling selects row-wise prefill. Normal release queues a request-stream free and returns the pool for reuse. On enqueue failure, recovery synchronizes the device, checks pool usage, and waits for any per-thread free. Explicit close reports the enqueue error. [Generation](../src/decoder.cpp#L385)<!-- prefix: "reclaim_pref" --> and [batch admission](../src/decoder_batch.cpp#L139)<!-- prefix: "reclaim_pref" --> retry pending frees even before width-one prefill. Multirow allocation also retries. [Allocation](../src/decoder_prefill.cpp#L242)<!-- prefix: "PrefillWorks" --> · [Release](../src/decoder_prefill.cpp#L295)<!-- prefix: "cudaError_t " -->

`--top-logits N` reports up to `N` raw scores per token, sorted by descending score then ascending ID. The range is 0–248,320 and defaults to 0. [Logit output](../apps/sglang-generate.cpp#L193)<!-- prefix: "void write_r" -->

#### Verification

Freeze independent inputs and precision-aware tolerances. Compare layer outputs and state on model shapes. Require exact tensor names, shapes, dtypes, scales, and specified packed bytes.

Use the [P2 verifier](#p2-gqa-verification) for MXFP8 encoding, causal attention, and page reuse. Check KV positions and ownership through append and decode. Check BF16 convolution history and FP32 recurrent state. Compare supported cache restores and continuations with fresh native execution under the same precision policy. [Independent GDN check](../tests/decoder/gdn_check.cu#L239)<!-- prefix: "void verify_" -->

Build the [host generator](../README.md#build-on-the-project-host). Set `SGLANG_GENERATE` to its path and `GPU_UUID` to the reserved B200 UUID. CUDA exposes it as ordinal 0. The default checkpoint is `/data0/shared/Qwen3.8-27B-FP8`. [Generator build](../tools/build-probe.sh#L76)<!-- prefix: "\"$xmake\" bui" -->

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    "${SGLANG_GENERATE:?set the printed generator path}" \
    --device 0 --prompt-ids 760,6511,314,9338,369 --max-new-tokens 8 --repeats 2
```

Check the prompt echo, eight in-range IDs, and `length`. Require identical IDs and finish reasons across runs. At budget one, set `--stop-token` to the first native ID. Require one ID and `stop`. At zero budget, require no IDs and `length`. Check reset by repeating the prompt after another request on the same decoder.

Check BF16 argmax and lowest-ID tie breaking. Compare logits with independent calculations. Use `--top-logits 248320` to check vocabulary shape and ranking on a short request. [Request ranking check](../tests/decoder/request_check.cpp#L171)<!-- prefix: "void check_r" -->

Historical diagnostics use the [FP8 prompt](../benchmarks/vllm/small-fp8.json#L2)<!-- prefix: "\"prompt_toke" -->, [vLLM IDs](../benchmarks/vllm/small-fp8-result.json#L20)<!-- prefix: "\"output_ids\"" -->, and frozen logits. See `artifacts/native-decoder-verification/final-evidence.sha256` and the [cross-engine verifier](../tests/decoder/run_native.py#L117)<!-- prefix: "def validate" -->.

#### Chunked prefill verification

Run the [CPU and B200 prefill checks](../tests/prefill/README.md) against the row-wise oracle. Match greedy IDs, ranked scores, and streams. Check chunk boundaries, mixed steps, reset, and slot reuse. The [measurement contract](performance.md#chunked-prefill) covers full BF16 logits and state.

#### Serial greedy MTP

Set `--mtp-steps` to `1`–`4` in either binary to cap proposals per round. The default `0` preserves ordinary generation. Positive steps apply server-wide and require `--max-active 1`. MTP loads draft weights once and shares the target embedding and LM head. Positive-budget requests own eager target state and draft MXFP8 KV. [Weights](../src/decoder.cpp#L237)<!-- prefix: "Decoder::Dec" --> · [Request state](../src/decoder.cpp#L252)<!-- prefix: "DecoderFinis" -->

MTP target prefill stays row-wise. The draft uses the final-normalized target row at `p` and the shifted token at `p+1`. Later proposals use the previous draft's final-normalized row and prediction. [Prefill](../src/decoder.cpp#L199)<!-- prefix: "DecoderFinis" --> · [Proposals](../src/decoder.cpp#L157)<!-- prefix: "bool propose" -->

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

Batch lanes share weights and own KV, GDN state, and output buffers. Positive budgets reserve KV for prompt length plus budget minus one. Each HTTP step gives ready lanes one position each. Remaining positions go round-robin to prefill lanes, up to 128 total. Decode and single-row prefill run together first. Larger chunks share scratch in slot order. Readback and retirement follow stream completion. C++ `advance` uses one position per selected lane. [Admission](../src/decoder_batch.cpp#L126)<!-- prefix: "void Decoder" --> · [Plan](../src/decoder_batch.cpp#L58)<!-- prefix: "PrefillRows " --> · [Submission](../src/decoder_batch.cpp#L175)<!-- prefix: "std::size_t " -->

In batch mode, `--trace-batches` reports selected `width`, actual `decode` width, and live post-step `request_bytes`. A second line reports `prefill_rows` by slot and transient `workspace_bytes`. Full decode gives `width=2 decode=2` or `width=4 decode=4`. [Trace](../apps/server/batch_service.cpp#L179)<!-- prefix: "void BatchSe" -->

Follow the [wire contract](../benchmarks/streaming/README.md#wire-contract) for fixed-length requests. Each token gets one event. Optional `stop_token_ids` takes one ID. A match emits that token and finishes with `stop`, including at the budget. Zero budget finishes with `length` without a model step. The [decoder limits](#constraints-and-failure-behavior) apply. HTTP limits are 8 KiB for headers and 2 MiB for the body. [Request parser](../apps/server/completions.cpp#L29)<!-- prefix: "CompletionRe" --> · [HTTP limits](../apps/server/http.cpp#L23)<!-- prefix: "constexpr st" -->

A reset or failed token write cancels the affected request. The batch retires its storage before slot reuse. A write-blocked lane waits while ready peers can advance. Clients can use `shutdown(SHUT_WR)` after sending a complete request. [Cancellation](../apps/server/batch_service.cpp#L130)<!-- prefix: "bool BatchSe" --> · [Disconnect check](../apps/server/http.cpp#L263)<!-- prefix: "bool disconn" -->

#### Streaming verification

Build the release server with the README procedure. Set `GPU_UUID` to the reserved B200 UUID. Put `nsys` on `PATH` for graph checks. Use a new evidence directory.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    /data1/shared/dongwu.chen/.pixi/envs/default/bin/python tests/batch/run_native.py \
    --binary build/bin/sglang-server --evidence artifacts/native-batch-check
```

Require `passed` in `status.json` for each mode. Check serial oracle streams and one graph launch per output after the first. At 2 and 4, check exact streams, full decode width on mixed-stop, and full scheduler width on page crossing. Confirm extra connections wait. After HTTP reset, verify readable peer streams, exact replacements, and zero request bytes. The independent core probe checks post-cancellation GPU advancement of retained lanes. Keep [runner](../tests/batch/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in li" --> identity and raw results. Run the serial [streaming smoke](../tests/test_native_streaming.py#L178)<!-- prefix: "def main():" --> with `--port PORT` for zero budget and half-close.

## Runtime and state ownership

**Proposed cache and MTP checks.** Define page and checkpoint intervals. Share compatible committed prefixes. Restore the last valid GDN checkpoint and replay the suffix. Discard rejected effects before publication. Verify GPU completion before storage reuse after cancellation, eviction, or allocation failure.

**Proposed checkpoint policy.** Save GDN state every 8,192 tokens and at the latest session request end. Measure checkpoint bytes and replay cost before adoption. Verify GDN and MTP semantics against the checkpoint and MXFP8 encoding in P2.

## Runnable milestones

For each proposed milestone, retain the exact invocation, inputs, expected result, and actual output.

| Stage | Run | Accept when |
| --- | --- | --- |
| P0 · Model and baseline | Inspect pinned tensors and configuration. Run the fixed greedy request with isolated vLLM on B200. | Reproduce the request and resolve model identity. Record memory budget, reference map, dependency decisions, and experiments for open risks. |
| P1 · Runtime | Run the [P1 runtime probe](#p1-runtime-probe). | Cold and warm launches verify every element and meet the probe's cache checks. |
| P2 · Operators and one layer | Exercise FP8 GEMM, normalization, RoPE, [MXFP8 GQA](#p2-gqa-cache-contract), GDN, FFN, and sampling on model shapes. | Operator and layer outputs meet prespecified independent tolerances. Record state updates and per-shape timings. |
| P3 · Text generation | Run the [native decoder checks](#verification) with the checkpoint. | Math, greedy repeats, state, counts, and finish behavior pass. Record output tokens/s, latency, and peak memory. |
| P4 · Cache and MTP | Compare fresh execution with prefix hits, restore and replay, and MTP on and off. Test zero, partial, and full acceptance, cancellation, eviction, and reuse. | Committed state and continuation match fresh execution. Sampling preserves the target distribution. Shared prefixes exclude rejected tokens. Released storage is safe to reuse. |
| P5 · Serving and media | Use an independent client for text, image, and video. Exercise streaming, all constraints, batching, chunked prefill, cancellation, and sustained concurrency. | API fixtures, media inputs and outputs, stop reasons, constrained outputs, and cleanup pass. Check leaks and stale state. |
| P6 · End-to-end performance | Profile a workload and change its limiting path. Repeat paired runs under the [benchmark agreement](performance.md). | Required rows pass correctness and meet throughput and latency targets. Keep raw samples, before/after profiles, and the adoption decision. |

P0 has a proposed budget of three to five working days. Assign an owner and experiment to each open question. Keep raw runs in ignored `artifacts/` and accepted procedures in maintained docs.
