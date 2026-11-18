# Single-B200 development contract

Build a C++ and CUDA C++ inference service for one NVIDIA B200. The local checkpoint at `/data0/shared/Qwen3.8-27B-FP8` matches published revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`. The [checkpoint investigation](research/checkpoint.md) records the content hashes, tensor inventory, and source paths in SGLang.

## Scope at adoption

The first executable check compiled a native `sm_100` cubin with XMake, GCC 15.3.0, and CUDA 13.4.92. On physical B200 4 (`GPU-57c04c1a-9ae4-b231-f31a-5f4367e759a8`), the kernel returned the expected value of 42. Model execution, service compatibility, and throughput each need their own acceptance evidence.

| Part | Responsibility | Source |
| --- | --- | --- |
| Model | Define weights, mathematics, tokenization, media processing, and native multi-token prediction (MTP). | Official checkpoint, configuration, processor, tokenizer, and model code pinned in P0. |
| Service | Study scheduling, Radix prefix reuse, memory ownership, speculative verification, constraints, and streaming behavior. | Architectural and behavioral reference `sgl-project/sglang` at `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. |
| Performance | Reproduce a run independently and compare the candidate on each agreed workload. | The [benchmark agreement](performance.md) owns the vLLM pin, controls, and metrics. |
| Candidate | Own the C++ model runtime and CUDA C++ GPU operator library in `src/`. Keep public headers in `include/sglang/`. Build with XMake and the managed default Pixi environment. | [Build options](../tools/build.sh#L5)<!-- prefix: "echo \"Usage:" --> and [XMake project](../xmake.lua#L2)<!-- prefix: "set_project(" -->. |

**Adopted scope.** Target one B200, the local checkpoint, and end-to-end inference. The candidate is an independent C++ and CUDA C++ inference framework. SGLang is an external source reference. Isolated vLLM is an external comparator. The candidate runtime depends on its own model and GPU operator code and the CUDA Toolkit. Future expansion to eight B200s is outside current acceptance.

**Operator policy from the project discussion.** Design the production path around project-owned GPU operators and the CUDA Toolkit. The reference design excludes cuBLAS, cuBLASLt, cuDNN, CUTLASS, FlashInfer, and equivalent operator libraries or generators from production execution. Ordinary HTTP, JSON, tokenizer, media decoder, and system libraries are eligible with justification. Record the purpose, license, version, and execution path for each choice. Reference environments may use their own operators.

**Proposed product scope from the discussion.** Cover text generation, sampling, stopping, image and video inputs, native MTP4, and concurrent scheduling. Use MXFP8 KV storage for candidate grouped-query attention (GQA). Investigate GDN recurrent state and Radix prefix reuse. Proposed service features include OpenAI-compatible endpoints, streaming, cancellation, and JSON Schema, regex, and grammar constraints. Verify model semantics against the checkpoint and agree on service behavior before accepting these targets.

## First task and next evidence

From the repository root on the managed host, run `bash tools/build.sh` to build the GPU library and both probes. For a device run, set `GPU_INDEX` to the physical index of the reserved B200.

```bash
bash tools/build.sh --gpu "${GPU_INDEX:?set the reserved B200 index}"
```

The build checks the `sm_100` image and the resolved CUDA runtime. The device run must print `GPU returned 42 (expected 42)` and exit successfully. Keep the command, device UUID, toolchain identities, and binary hash. This probe exercises an ahead-of-time kernel.

For P0, inspect the official configuration, weight index, tensor metadata, tokenizer, chat template, processor, and model implementation together. Record layer order, dimensions, weight formats and scales, state updates, position encoding, MTP weights, and media inputs. Map the applicable behavior to files at the pinned SGLang revision. Propose the API version, endpoints, parameters, streaming events, and constraint semantics for review. If the checkpoint conflicts with a proposed feature, record the evidence and request a scope decision.

The [baseline smoke procedure](performance.md#first-baseline-run) completed on physical B200 1. The seven measured requests produced the same eight greedy IDs with BF16 KV and MTP disabled. Use this independently derived five-token prompt and eight-token continuation to diagnose decoder bring-up. The [GQA cache comparison](performance.md#gqa-cache-comparison) defines the paired acceptance configuration.

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

Keep the command and full output, including device properties, library and binary identities, allocation size, and both phase lines. A missing phase, incorrect result, or nonzero exit fails this check. The runtime check covers one fixed kernel and a function cache scoped to its CUDA context. Model execution and performance acceptance use later milestones.

### P2 GQA cache contract

Store each GQA K and V vector as E4M3 values with one E8M0 scale per contiguous group of 32 channels. The checkpoint has head width 256 and four KV heads, so each K or V vector carries eight scales. The checkpoint FP8 weight matrices use separate BF16 inverse scales per 128×128 weight block. GDN uses separate convolution history and recurrent state, with FP32 configured for the recurrent state. [Checkpoint investigation](research/checkpoint.md)

Allocate separate device buffers for K and V payloads and scales. Each 128-token physical page needs 131,072 payload bytes and 4,096 scale bytes for K, and the same for V. Map logical pages to distinct physical pages with the device page table. [Cache layout](../include/sglang/mxfp8_gqa.h#L22)<!-- prefix: "/// Separate" --> · [Page table](../include/sglang/mxfp8_gqa.h#L34)<!-- prefix: "struct Mxfp8" -->

Link the CUDA consumer to the static `sglang_cuda` target. Apply RoPE to K and Q before calling the cache writer and attention, respectively. Pass contiguous BF16 K and V tensors shaped `[tokens, 4, 256]` to `mxfp8_kv_append`. Once the required KV prefix is ready on the stream, call `mxfp8_gqa_attention` with the valid prefix length and first query position. Q and output are BF16 with shape `[queries, 24, 256]`. Query row `r` attends through position `first_query_position + r` inclusive. Check each launch status. Observe stream completion before releasing or reusing storage. [Library target](../xmake.lua#L46)<!-- prefix: "target(\"sgla" --> · [Append](../include/sglang/mxfp8_gqa.h#L48)<!-- prefix: "cudaError_t " --> · [Attention](../include/sglang/mxfp8_gqa.h#L55)<!-- prefix: "cudaError_t " -->

The writer derives each K or V scale from its group maximum magnitude `amax`. It stores `ceil(log2(max(amax, 1e-30) / 448)) + 127`, clamped to `[0, 254]`. Attention reads the packed cache directly and decodes K and V to FP32 inside the kernel. It converts BF16 Q to FP32 for the dot product. [Scale calculation](../src/mxfp8_gqa.cu#L32)<!-- prefix: "__device__ i" --> · [Attention kernel](../src/mxfp8_gqa.cu#L82)<!-- prefix: "__global__ v" -->

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

## Runtime and state ownership

**Proposed design.** The CPU prepares inputs, applies request policy, and schedules prefill and decode batches. The executor receives batch metadata and reports GPU completion. Keep model semantics independent of kernel dispatch. Keep Radix prefix ownership separate from physical GQA pages and GDN state storage.

**Proposed cache and MTP checks.** Define the token interval covered by each page and recurrent checkpoint. Share compatible committed prefixes. Restore the last valid GDN checkpoint and replay the uncovered suffix. Keep tentative MTP work private until target verification. On rejection, retain state for accepted tokens and discard rejected effects before publishing a shared prefix. After cancellation, eviction, or allocation failure, check that outstanding GPU users finish before storage reuse.

**Proposed checkpoint policy.** Save GDN state after each 8,192 processed tokens and at the latest session request end. Measure checkpoint bytes and replay cost before adopting that interval. Verify the exact GDN state tensors and the MTP head and verification semantics against the checkpoint. Verify the candidate's MXFP8 encoding and scale layout in P2.

## Runnable milestones

The discussion proposes these stages after adoption. Each row gives a test and the evidence needed to close it if approved. Preserve the exact invocation, inputs, expected result, and actual output with each milestone.

| Stage | Run | Accept when |
| --- | --- | --- |
| P0 · Model and baseline | Inspect the pinned tensors and configuration. Run one fixed greedy text request through isolated vLLM on the B200. | The manifest resolves the model identity. The request reproduces. Record the single-card memory budget, reference map, dependency decisions, and smallest experiment for each open risk. |
| P1 · Runtime | Run the [P1 runtime probe](#p1-runtime-probe) on one B200. | The cold miss and warm hit each complete and verify the full output. Retain the device, build, cache, allocation, and timing evidence. |
| P2 · Operators and one layer | Run FP8 GEMM, normalization, RoPE, [MXFP8 GQA](#p2-gqa-cache-contract), GDN, FFN, and sampling on verified model shapes. Exercise each applicable layer type. | Operator and layer outputs meet prespecified independent reference tolerances. Capture state updates and per-shape timings. |
| P3 · Text generation | Load the official weights. Prefill a fixed prompt and decode to a fixed limit. Repeat with a stop condition. | Compare logits and greedy token IDs with the reference. Verify the finish reason and sampling behavior. Record output tokens/s, latency, and peak memory. |
| P4 · Cache and MTP | Compare fresh execution with prefix hits, restore and replay, and native MTP on and off. Exercise zero, partial, and full acceptance, cancellation, eviction, and reuse. | Committed state and continuation agree with fresh target execution. Sampling preserves the target distribution. Rejected tokens never enter a shared prefix. Released storage is safe to reuse. |
| P5 · Serving and media | Use an independent client for text, image, and video requests. Exercise streams, all constraint modes, continuous batching, chunked prefill, cancellation, and a sustained concurrent run. | Verify the agreed API fixtures, media inputs and outputs, stop reasons, valid constrained outputs, and cleanup. Check for leaks and stale state across requests. |
| P6 · End-to-end performance | Profile a named workload. Change the limiting path. Repeat the baseline and candidate runs under the benchmark agreement. | Each required row meets its agreed throughput and latency targets and passes correctness. Retain raw samples, before/after profiles, and the decision to keep or revert the change. |

The proposed P0 time budget is three to five working days. Open questions need an owner and a discriminating experiment. Record raw runs in ignored `artifacts/` storage. Keep accepted decisions and reproducible procedures in the maintained documents.
