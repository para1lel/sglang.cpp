# Single-B200 development contract

Build a C++ and CUDA C++ inference service for one NVIDIA B200. The local checkpoint at `/data0/shared/Qwen3.8-27B-FP8` matches published revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`. The [checkpoint investigation](research/checkpoint.md) records the content hashes, tensor inventory, and source paths in SGLang.

## Scope at adoption

The first executable check compiled a native `sm_100` cubin with XMake, GCC 15.3.0, and CUDA 13.4.92. On physical B200 4 (`GPU-57c04c1a-9ae4-b231-f31a-5f4367e759a8`), the kernel returned the expected value of 42. Model execution, service compatibility, and throughput each need their own acceptance evidence.

| Part | Responsibility | Source |
| --- | --- | --- |
| Model | Define weights, mathematics, tokenization, media processing, and native multi-token prediction (MTP). | Official checkpoint, configuration, processor, tokenizer, and model code pinned in P0. |
| Service | Study scheduling, Radix prefix reuse, memory ownership, speculative verification, constraints, and streaming behavior. | Architectural and behavioral reference `sgl-project/sglang` at `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. |
| Performance | Reproduce a run independently and compare the candidate on each agreed workload. | The [benchmark agreement](performance.md) owns the vLLM pin, controls, and metrics. |
| Candidate build | Build project C++ and CUDA C++ with XMake and the managed default Pixi environment. | [Build options](../tools/build.sh#L5)<!-- prefix: "echo \"Usage:" --> and [XMake project](../xmake.lua#L2)<!-- prefix: "set_project(" -->. |

**Adopted scope.** Target one B200, the local checkpoint, and end-to-end inference. Use SGLang to study architecture and behavior. Use isolated vLLM for independent comparisons. Future expansion to eight B200s is outside current acceptance.

**Operator policy from the project discussion.** Design the production path around project-owned GPU operators and the CUDA Toolkit. The reference design excludes cuBLAS, cuBLASLt, cuDNN, CUTLASS, FlashInfer, and equivalent operator libraries or generators from production execution. Ordinary HTTP, JSON, tokenizer, media decoder, and system libraries are eligible with justification. Record the purpose, license, version, and execution path for each choice. Reference environments may use their own operators.

**Proposed product scope from the discussion.** Cover text generation, sampling, stopping, image and video inputs, native MTP4, and concurrent scheduling. Investigate grouped-query attention (GQA) with MXFP8 KV storage, GDN recurrent state, and Radix prefix reuse. Proposed service features include OpenAI-compatible endpoints, streaming, cancellation, and JSON Schema, regex, and grammar constraints. Verify model semantics against the checkpoint and agree on service behavior before accepting these targets.

## First task and next evidence

From the repository root on the managed host, run `bash tools/build.sh` for a build-only probe. For a device run, set `GPU_INDEX` to the physical index of the reserved B200.

```bash
bash tools/build.sh --gpu "${GPU_INDEX:?set the reserved B200 index}"
```

The build checks the `sm_100` image and the loaded CUDA runtime. The device run must print `GPU returned 42 (expected 42)` and exit successfully. Keep the command, device UUID, toolchain identities, and binary hash. This probe exercises an ahead-of-time kernel. P1 also requires a timed NVRTC launch.

For P0, inspect the official configuration, weight index, tensor metadata, tokenizer, chat template, processor, and model implementation together. Record layer order, dimensions, weight formats and scales, state updates, position encoding, MTP weights, and media inputs. Map the applicable behavior to files at the pinned SGLang revision. Propose the API version, endpoints, parameters, streaming events, and constraint semantics for review. If the checkpoint conflicts with a proposed feature, record the evidence and request a scope decision.

The [baseline smoke procedure](performance.md#first-baseline-run) completed on physical B200 1. The seven measured requests produced the same eight greedy IDs with BF16 KV and MTP disabled. Give the first candidate decoder this five-token prompt and compare all eight IDs on that same UUID. Then schedule the longer streaming workloads separately.

## Runtime and state ownership

**Proposed design.** The CPU prepares inputs, applies request policy, and schedules prefill and decode batches. The executor receives batch metadata and reports GPU completion. Keep model semantics independent of kernel dispatch. Keep Radix prefix ownership separate from physical GQA pages and GDN state storage.

**Proposed cache and MTP checks.** Define the token interval covered by each page and recurrent checkpoint. Share compatible committed prefixes. Restore the last valid GDN checkpoint and replay the uncovered suffix. Keep tentative MTP work private until target verification. On rejection, retain state for accepted tokens and discard rejected effects before publishing a shared prefix. After cancellation, eviction, or allocation failure, check that outstanding GPU users finish before storage reuse.

**Proposed checkpoint policy.** Save GDN state after each 8,192 processed tokens and at the latest session request end. Measure checkpoint bytes and replay cost before adopting that interval. P0 must establish the exact state tensors, MXFP8 encoding and scale layout, and MTP head and verification semantics. The name “MTP4” alone does not establish a head count or draft length.

## Runnable milestones

The discussion proposes these stages after adoption. Each row gives a test and the evidence needed to close it if approved. Preserve the exact invocation, inputs, expected result, and actual output with each milestone.

| Stage | Run | Accept when |
| --- | --- | --- |
| P0 · Model and baseline | Inspect the pinned tensors and configuration. Run one fixed greedy text request through isolated vLLM on the B200. | The manifest resolves the model identity. The request reproduces. Record the single-card memory budget, reference map, dependency decisions, and smallest experiment for each open risk. |
| P1 · Runtime | Allocate device memory. Compile, load, and dispatch a project kernel with NVRTC. Repeat with the JIT cache populated and time execution with CUDA events. | Both runs return the expected value. Record device properties, allocation size, cache miss and hit, compile time, and completed kernel time. |
| P2 · Operators and one layer | Run FP8 GEMM, normalization, RoPE, GQA, GDN, FFN, and sampling on verified model shapes. Exercise each applicable layer type. | Operator and layer outputs meet prespecified independent reference tolerances. Capture state updates and per-shape timings. |
| P3 · Text generation | Load the official weights. Prefill a fixed prompt and decode to a fixed limit. Repeat with a stop condition. | Compare logits and greedy token IDs with the reference. Verify the finish reason and sampling behavior. Record output tokens/s, latency, and peak memory. |
| P4 · Cache and MTP | Compare fresh execution with prefix hits, restore and replay, and native MTP on and off. Exercise zero, partial, and full acceptance, cancellation, eviction, and reuse. | Committed state and continuation agree with fresh target execution. Sampling preserves the target distribution. Rejected tokens never enter a shared prefix. Released storage is safe to reuse. |
| P5 · Serving and media | Use an independent client for text, image, and video requests. Exercise streams, all constraint modes, continuous batching, chunked prefill, cancellation, and a sustained concurrent run. | Verify the agreed API fixtures, media inputs and outputs, stop reasons, valid constrained outputs, and cleanup. Check for leaks and stale state across requests. |
| P6 · End-to-end performance | Profile a named workload. Change the limiting path. Repeat the baseline and candidate runs under the benchmark agreement. | Each required row meets its agreed throughput and latency targets and passes correctness. Retain raw samples, before/after profiles, and the decision to keep or revert the change. |

The proposed P0 time budget is three to five working days. Open questions need an owner and a discriminating experiment. Record raw runs in ignored `artifacts/` storage. Keep accepted decisions and reproducible procedures in the maintained documents.
