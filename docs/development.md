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

**Adopted scope.** Target one B200 with the local checkpoint and project-owned model and CUDA operators. The [native decoder](#p3-native-text-decoder) documents its parser dependency.

**Operator policy.** Exclude cuBLAS, cuBLASLt, cuDNN, CUTLASS, FlashInfer, and equivalent operator libraries or generators from production execution. Justify HTTP, JSON, tokenizer, media decoder, and system libraries. Record purpose, license, version, and execution path. Reference engines may use their own operators.

**Proposed scope.** Cover text generation, sampling, stopping, image and video inputs, MTP4, and concurrent scheduling. Use MXFP8 KV for grouped-query attention (GQA). Investigate GDN state and Radix prefix reuse. Verify checkpoint and service semantics before accepting OpenAI-compatible endpoints, streaming, cancellation, and JSON Schema, regex, and grammar constraints.

## First task and next evidence

From the repository root, set `GPU_INDEX` to the reserved B200's physical index.

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

For each K or V group's maximum magnitude `amax`, store `ceil(log2(max(amax, 1e-30) / 448)) + 127`, clamped to `[0, 254]`. Attention decodes K and V and converts BF16 Q to FP32. [Scale calculation](../src/mxfp8_gqa.cu#L35)<!-- prefix: "__device__ i" --> · [Direct attention](../src/mxfp8_gqa.cu#L98)<!-- prefix: "__global__ v" -->

Exact scoring preserves the direct FP32 group reductions, ordered sum of eight groups, and `1/16` scale. Replay computes `expf` weights in parallel in 32-token tiles. Running maxima, denominator FMAs, and value FMAs follow token order. FP32 division precedes one BF16 rounding. [Score](../src/mxfp8_gqa.cu#L168)<!-- prefix: "__global__ v" --> · [Replay](../src/mxfp8_gqa.cu#L213)<!-- prefix: "__global__ v" -->

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
| `sglang-generate` | Parse a token-ID prompt, run requests serially, and write JSON. | [CLI entry](../apps/sglang-generate.cpp#L234)<!-- prefix: "int main(int" --> |
| `sglang_checkpoint` | Validate the text configuration, index, and safetensors metadata. Map immutable shard bytes and expose checked tensor views. | [Loader API](../include/sglang/checkpoint.h#L22)<!-- prefix: "class Checkp" --> |
| `sglang_decoder` | Own device weights and run requests with `sglang_cuda` operators. | [Decoder API](../include/sglang/decoder.h#L59)<!-- prefix: "class Decode" --> · [Layer dispatch](../src/decoder_forward.cpp#L330)<!-- prefix: "void Decoder" --> |

The checkpoint loader uses Pixi's LLVM Support 23.1.5 (`llvmdev` and `libllvm23`, Apache-2.0 WITH LLVM-exception) for JSON. [JSON parser](../src/checkpoint.cpp#L170)<!-- prefix: "json::Object" --> · [Link configuration](../xmake.lua#L86)<!-- prefix: "target(\"sgla" -->

#### Runtime path

The decoder validates tensor names, dtypes, and shapes before upload. Upload retains mappings and allocations until completion. Deferred cleanup retains shared weights. FP8 projections quantize BF16 inputs to E4M3 with one FP32 scale per 128 channels. [Weight upload](../src/decoder_weights.cpp#L97)<!-- prefix: "Decoder::Mod" --> · [Stream cleanup](../src/decoder_resources.cpp#L210)<!-- prefix: "CleanupError" --> · [Activation quantization](../src/text_ops.cu#L189)<!-- prefix: "__global__ v" -->

Serial MTP-off generation reuses its stream and grows storage on demand. Prefill rebuilds [MXFP8 pages](#p2-gqa-cache-contract) with an identity page table. Reset zeros BF16 convolution history and FP32 recurrent state across 48 GDN layers. [Generation loop](../src/decoder.cpp#L342)<!-- prefix: "void Decoder" --> · [State reset](../src/decoder_resources.cpp#L531)<!-- prefix: "void Request" -->

Prefill processes up to 128 positions per layer in causal and recurrent order. The last row supplies initial logits. Serial MTP-off decode replays a graph. Overlapping calls use private eager storage. [Prefill dispatch](../src/decoder_prefill.cpp#L379)<!-- prefix: "void Decoder" --> · [Decode graph](../src/decoder_forward.cpp#L299)<!-- prefix: "void Decoder" -->

Both binaries accept `--fp8-prefill exact|tensor-core`. C++ callers pass `Fp8PrefillPolicy` as the fourth `Decoder` constructor argument. Default `exact` preserves SIMT row arithmetic. Tensor-core arithmetic can change logits and greedy IDs. [Policy](../include/sglang/decoder.h#L54)<!-- prefix: "enum class F" -->

Tensor-core projections require MTP off, 3–128 rows, 128-divisible input width, and contiguous, 16-byte-aligned E4M3 payloads. The `sm_100a` build uses TMA, TMEM, and `tcgen05` PTX. Eligibility also requires capability 10.0, a matching kernel image, opt-in shared memory capacity, and driver support for `cuTensorMapEncodeTiled`. Ineligible calls use SIMT. Invalid arguments and execution errors propagate. [Projection](../src/decoder_prefill.cpp#L276)<!-- prefix: "void Prefill" --> · [Eligibility](../src/prefill_fp8_sm100.cu#L452)<!-- prefix: "std::optiona" --> · [Build](../xmake.lua#L61)<!-- prefix: "target(\"sgla" -->

Target prefill and draft decode use direct attention. Eager target decode, including MTP verification, selects exact attention below score capacity. Serial graph capture selects exact attention when page-rounded KV slots fit that capacity. Exact graphs read device positions. Other decode uses direct attention. [GQA dispatch](../src/decoder_forward.cpp#L156)<!-- prefix: "cudaError_t " -->

The lane stream orders KV append, scoring, replay, and gating. Score capacity is the lesser of allocated KV slots and 5,120. The logits allocation holds 24 FP32 scores per slot until the LM head writes logits. [Workspace](../src/decoder_resources.cpp#L502)<!-- prefix: "ExactGqaWork" -->

Tuned FP8 projections require a forward group containing one target decode row. This includes serial graph decode and single-row eager or MTP verification. A lone batch decode row can qualify when prefill chunks run separately. GQA combines Q/gate, K, and V in one launch. The MLP fusion rounds gate and up to BF16 before SwiGLU. GDN QKV and Z stay separate. [Lane selection](../src/decoder_forward.cpp#L48)<!-- prefix: "bool single_" --> · [Batch grouping](../src/decoder_batch.cpp#L241)<!-- prefix: "if (rows[slo" -->

Tuned GEMV requires 16-byte-aligned FP8 weights and activations and input widths of 5,120, 6,144, or 17,408. Both fusions require the same alignment and width 5,120. Q/gate, K, and V require 12,288, 1,024, and 1,024 output rows. Gate and up require 17,408 each. Other valid inputs use separate operations. Output spans must be disjoint from inputs and each other. Fused SwiGLU permits identical gate and up spans. [Tuned GEMV](../src/text_decode_fp8.cu#L276)<!-- prefix: "cudaError_t " --> · [Fused QKV](../src/text_decode_fp8.cu#L298)<!-- prefix: "cudaError_t " --> · [Fused SwiGLU](../src/text_decode_fp8.cu#L321)<!-- prefix: "cudaError_t " --> · [Buffers](../include/sglang/decoder_cuda.cuh#L68)<!-- prefix: "// Read-only" -->

Row-wise prefill and MTP draft decode use GEMV. Packed GEMM handles eligible groups of two to four batch lanes or two to five MTP verification rows. Other groups use per-row GEMV. Wide GEMV and packed GEMM require 512-divisible input widths and 16-byte-aligned FP8 payloads. [GEMV](../src/text_ops.cu#L745)<!-- prefix: "cudaError_t " --> · [GEMM](../src/text_ops.cu#L763)<!-- prefix: "cudaError_t " -->

Single-row BF16 RMSNorm uses vector loads at width 5,120 with 16-byte-aligned branch, gains, and optional residual. Other calls use generic kernels. [RMSNorm dispatch](../src/text_ops.cu#L919)<!-- prefix: "cudaError_t " -->

Both prefill policies share the decode kernels. Outside MTP verification, a single target decode row can use warp quantization and combined GDN gates. This includes a lone decode row in a batch. Quantization needs whole 128-channel groups, 8-byte-aligned BF16 input, and 4-byte-aligned E4M3 values. It preserves FP32 round-to-nearest scale and reciprocal divisions and saturated E4M3 conversion. Other valid layouts use the generic quantizer. [Dispatch](../src/decoder_forward.cpp#L62)<!-- prefix: "void quantiz" --> · [Quantizer](../src/text_decode_bandwidth.cu#L187)<!-- prefix: "cudaError_t " -->

Combined GDN gates need matching 48-by-5,120 shapes. After the initial score, serial eager and graph decode can pair two vocabulary rows per warp. The paired LM head shares input loads across each pair and uses streaming weight loads. It needs a positive even row count. Both BF16 kernels require width 5,120 and 16-byte-aligned weights and input. They preserve two ordered FP32 FMA chains, their sum, ordered warp reduction, and BF16 rounding. Other valid shapes or alignments use the existing GEMVs. [GDN gates](../src/text_decode_bandwidth.cu#L154)<!-- prefix: "cudaError_t " --> · [LM head](../src/text_decode_bandwidth.cu#L175)<!-- prefix: "cudaError_t " -->

#### Constraints and failure behavior

Pass one nonempty prompt with `--prompt-ids` or `--prompt-file`. Separate decimal IDs with commas or ASCII whitespace. Prompt and stop IDs must be in `[0, 248320)`. Prompt length plus budget must fit 262,144 tokens. The default budget is one. Zero budget returns no tokens and `length`. [CLI options](../apps/sglang-generate.cpp#L34)<!-- prefix: "void usage(s" --> · [Request validation](../src/decoder.cpp#L20)<!-- prefix: "void validat" -->

Greedy selection takes the highest BF16 logit, breaking ties by lowest ID. `--stop-token` emits a match and returns `stop`, even at the budget. Budget exhaustion returns `length`. The CLI applies no implicit EOS. Input, load, execution, and explicit cleanup errors report to stderr and exit 1. [Selection](../src/text_ops.cu#L549)<!-- prefix: "__device__ T" --> · [Generation loop](../src/decoder.cpp#L342)<!-- prefix: "void Decoder" -->

The decoder retains CUDA resources and transfer buffers while completion is uncertain. Model and request setup retry cleanup on the same device. Persistent errors block setup. Terminal errors and ambiguous frees quarantine resources until exit. [Deferred cleanup](../src/decoder_resources.cpp#L115)<!-- prefix: "static void " --> · [Destructor diagnostics](../src/decoder_resources.cpp#L321)<!-- prefix: "Allocations:" -->

Multirow workspaces use private pools on their CUDA device. At most one pool per device sets a 32 MiB release threshold for reserved backing. Checkout prefers it when idle. Other pools use CUDA's default threshold, including when retention is unsupported. [Retention](../src/decoder_prefill.cpp#L40)<!-- prefix: "void configu" -->

Single rows, scratch OOM, and unsupported pool allocation use row-wise prefill. Release queues a stream-ordered free. After enqueue failure, recovery synchronizes the device, checks used bytes, and waits for any retried free. Explicit close reports the original error. [Generation](../src/decoder.cpp#L455)<!-- prefix: "reclaim_pref" -->, [batch admission](../src/decoder_batch.cpp#L163)<!-- prefix: "reclaim_pref" -->, and multirow allocation retry pending frees. [Allocation](../src/decoder_prefill.cpp#L307)<!-- prefix: "PrefillWorks" --> · [Release](../src/decoder_prefill.cpp#L361)<!-- prefix: "cudaError_t " -->

`--top-logits N` reports up to `N` raw scores per token, sorted by descending score then ascending ID. The range is 0–248,320 and defaults to 0. [Logit output](../apps/sglang-generate.cpp#L200)<!-- prefix: "void write_r" -->

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

Run the [CPU and B200 prefill checks](../tests/prefill/README.md) in exact mode. Match the row-wise oracle and check chunk boundaries, mixed steps, reset, and slot reuse. Use the [tensor-core comparison](performance.md#fp8-tensor-core-prefill) for the opt-in mode.

#### Serial greedy MTP

The [native MTP guide](mtp.md) defines exact tiled seeding, the target verifier, and serial request limits. Follow its [verification procedure](mtp.md#verification) before measuring MTP.

### Native streaming server

Follow the [server launch procedure](../README.md#serve-token-ids-over-http). See [native text completions](text-serving.md) for string prompts, tokenizer assets, and verification.

By default, `--max-active 1` uses the serial graph path. `--eager` requires one active request and MTP off. Set `--max-active 2`, `3`, or `4` for eager batching. See [batch CUDA graphs](batch-graphs.md) for opt-in dispatch and verification. Extra connections wait in the listener backlog. [Serving modes](../apps/sglang-server.cpp#L198)<!-- prefix: "if (argument" --> · [Socket workers](../apps/sglang-server.cpp#L136)<!-- prefix: "void serve_b" -->

Lanes share weights and own KV, GDN state, and output buffers. Positive budgets reserve KV for prompt length plus budget minus one. HTTP steps assign one position per ready lane, then distribute remaining positions round-robin to prefill lanes, up to 128 total. Decode and single-row prefill run together first. Larger chunks share scratch in slot order. Readback and retirement follow stream completion. C++ `advance` uses one position per selected lane. [Admission](../src/decoder_batch.cpp#L150)<!-- prefix: "void Decoder" --> · [Plan](../src/decoder_batch.cpp#L73)<!-- prefix: "PrefillRows " --> · [Submission](../src/decoder_batch.cpp#L229)<!-- prefix: "void Decoder" -->

In batch mode, `--trace-batches` reports selected `width`, actual `decode` width, and live post-step `request_bytes`. The second line gives `prefill_rows` by slot and transient `workspace_bytes`. [Trace](../apps/server/batch_service.cpp#L181)<!-- prefix: "void BatchSe" -->

Follow the [benchmark token-ID wire contract](../benchmarks/streaming/README.md#wire-contract). Optional `stop_token_ids` takes one ID. A match emits that ID and finishes with `stop`, including at the budget. Zero budget finishes with `length` without a model step. The [decoder limits](#constraints-and-failure-behavior) apply. [Request parser](../apps/server/completions.cpp#L31)<!-- prefix: "CompletionRe" -->

A reset or failed token write cancels the affected request. The batch retires its storage before slot reuse. A write-blocked lane waits while ready peers can advance. Clients can use `shutdown(SHUT_WR)` after sending a complete request. [Cancellation](../apps/server/batch_service.cpp#L132)<!-- prefix: "bool BatchSe" --> · [Disconnect check](../apps/server/http.cpp#L263)<!-- prefix: "bool disconn" -->

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
| P0 · Model and baseline | Inspect pinned tensors and configuration. Run the fixed greedy request with isolated vLLM on B200. | Reproduce the request. Record model identity, memory budget, reference map, dependencies, and open experiments. |
| P1 · Runtime | Run the [P1 runtime probe](#p1-runtime-probe). | Cold and warm launches verify every element and meet the probe's cache checks. |
| P2 · Operators and one layer | Exercise FP8 GEMM, normalization, RoPE, [MXFP8 GQA](#p2-gqa-cache-contract), GDN, FFN, and sampling on model shapes. | Meet independent operator and layer tolerances. Record state updates and per-shape timings. |
| P3 · Text generation | Run the [native decoder checks](#verification) with the checkpoint. | Math, greedy repeats, state, counts, and finish behavior pass. Record output tokens/s, latency, and peak memory. |
| P4 · Cache and MTP | Compare fresh execution with prefix hits, replay, and MTP on and off. Check zero, partial, and full acceptance, cancellation, eviction, and reuse. | State and continuation match fresh execution. Sampling preserves the target distribution. Share only committed tokens and reuse only retired storage. |
| P5 · Serving and media | Use an independent client for text, image, and video. Exercise streaming, all constraints, batching, chunked prefill, cancellation, and sustained concurrency. | API fixtures, media inputs and outputs, stop reasons, constrained outputs, and cleanup pass. Check leaks and stale state. |
| P6 · End-to-end performance | Profile a workload and change its limiting path. Repeat paired runs under the [benchmark agreement](performance.md). | Required rows meet correctness, throughput, and latency targets. Keep raw samples, before/after profiles, and the adoption decision. |

P0 has a proposed budget of three to five working days. Assign an owner and experiment to each open question. Keep raw runs in ignored `artifacts/` and accepted procedures in maintained docs.
