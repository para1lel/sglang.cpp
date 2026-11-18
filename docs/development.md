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

**Adopted scope.** Target one B200, the local checkpoint, and end-to-end inference. The candidate is an independent C++ and CUDA C++ inference framework. SGLang is an external source reference. Isolated vLLM is an external comparator. The candidate uses project-owned model and GPU operator code with the CUDA Toolkit. The [native decoder](#p3-native-text-decoder) documents its checkpoint parser dependency. Future expansion to eight B200s is outside current acceptance.

**Operator policy from the project discussion.** Design the production path around project-owned GPU operators and the CUDA Toolkit. The reference design excludes cuBLAS, cuBLASLt, cuDNN, CUTLASS, FlashInfer, and equivalent operator libraries or generators from production execution. Ordinary HTTP, JSON, tokenizer, media decoder, and system libraries are eligible with justification. Record the purpose, license, version, and execution path for each choice. Reference environments may use their own operators.

**Proposed product scope from the discussion.** Cover text generation, sampling, stopping, image and video inputs, native MTP4, and concurrent scheduling. Use MXFP8 KV storage for candidate grouped-query attention (GQA). Investigate GDN recurrent state and Radix prefix reuse. Proposed service features include OpenAI-compatible endpoints, streaming, cancellation, and JSON Schema, regex, and grammar constraints. Verify model semantics against the checkpoint and agree on service behavior before accepting these targets.

## First task and next evidence

From the repository root on the managed host, run `bash tools/build.sh` to build the native generator, its libraries, and both probes. For a device run, set `GPU_INDEX` to the physical index of the reserved B200.

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

### P3 native text decoder

#### Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| `sglang-generate` | Parse one token-ID prompt, run requests serially, and write one JSON result. | [CLI entry](../apps/sglang-generate.cpp#L216)<!-- prefix: "int main(int" --> |
| `sglang_checkpoint` | Validate the text configuration, index, and safetensors metadata. Map immutable shard bytes and expose checked tensor views. | [Loader API](../include/sglang/checkpoint.h#L22)<!-- prefix: "class Checkp" --> |
| `sglang_decoder` | Own device weights and run each request using operators from `sglang_cuda`. | [Decoder API](../include/sglang/decoder.h#L34)<!-- prefix: "class Decode" --> · [Layer dispatch](../src/decoder_forward.cpp#L71)<!-- prefix: "void Decoder" --> |

The loader uses LLVM Support 23.1.5 from Pixi (`llvmdev` and `libllvm23`, Apache-2.0 WITH LLVM-exception) only to parse checkpoint JSON. The loader owns the safetensors validation and mapping. [JSON parser](../src/checkpoint.cpp#L170)<!-- prefix: "json::Object" --> · [Link configuration](../xmake.lua#L60)<!-- prefix: "target(\"sgla" -->

#### Runtime path

The decoder checks the name, dtype, and shape of each required text tensor before upload. The upload state retains the checkpoint mappings and device allocations until CUDA confirms stream completion. The decoder shares its device weights across requests. Each request stream holds a weight reference until CUDA confirms completion, including during deferred cleanup. FP8 projections use the checkpoint's E4M3 weights and BF16 inverse scales. They dynamically quantize BF16 inputs to E4M3 with one FP32 scale per 128 channels. [Weight upload](../src/decoder_weights.cpp#L97)<!-- prefix: "Decoder::Mod" --> · [Stream cleanup](../src/decoder_resources.cpp#L168)<!-- prefix: "CleanupError" --> · [Activation quantization](../src/text_ops.cu#L100)<!-- prefix: "__global__ v" -->

Each request has its own stream, workspace, and state. The 48 GDN layers start with zeroed BF16 convolution history and FP32 recurrent state. The 16 GQA layers append to private [MXFP8 pages](#p2-gqa-cache-contract) with an identity page table. Prefill processes the prompt one token at a time and scores its last position. Each subsequent decode step processes the preceding emitted token. The host waits for the stream before reading each selection. [Request allocation](../src/decoder_resources.cpp#L339)<!-- prefix: "Request::Req" --> · [Generation loop](../src/decoder.cpp#L80)<!-- prefix: "DecoderResul" -->

#### Constraints and failure behavior

Pass one nonempty prompt with `--prompt-ids` or `--prompt-file`. Use decimal token IDs separated by commas or ASCII whitespace. Prompt and stop IDs must be in `[0, 248320)`. Prompt length plus `--max-new-tokens` must fit 262,144 tokens. The output budget defaults to one. A zero budget returns no tokens with finish reason `length` after model load and request validation. [CLI options](../apps/sglang-generate.cpp#L31)<!-- prefix: "void usage(s" --> · [Request validation](../src/decoder.cpp#L15)<!-- prefix: "void validat" -->

The decoder chooses the highest BF16 logit and breaks ties by lowest token ID. Stopping is optional. A generated `--stop-token` match is included in the output and returns `stop`, even on the last allowed step. Otherwise the request returns `length`. The CLI applies no implicit EOS rule. Input, load, execution, or explicit request cleanup errors produce a diagnostic on stderr and exit status 1. The CLI covers token-ID text generation only. [Selection](../src/text_ops.cu#L239)<!-- prefix: "__device__ T" --> · [Generation loop](../src/decoder.cpp#L80)<!-- prefix: "DecoderResul" -->

If cleanup cannot confirm stream completion, the decoder retains the pending stream, device storage, and host transfer buffers. Before a model load or generation with a positive output budget, it retries recoverable cleanup on the same device. Terminal errors and ambiguous stream destruction or memory frees keep the affected resources quarantined until process exit. A persistent cleanup error prevents further model loads and requests that require allocation on that device. [Deferred cleanup](../src/decoder_resources.cpp#L105)<!-- prefix: "static void " --> · [Destructor diagnostics](../src/decoder_resources.cpp#L266)<!-- prefix: "Allocations:" -->

Use `--top-logits N` to report up to `N` raw scores per emitted token, ranked by score and then by ascending token ID. This option accepts values from 0 to 248,320 and defaults to 0. [Logit output](../apps/sglang-generate.cpp#L187)<!-- prefix: "void write_r" -->

#### Verification

Use the [host build procedure](../README.md#build-on-the-project-host) with a retained build directory. Set `SGLANG_GENERATE` to the executable printed after `Native generator:`. Reserve an `sm_100` B200 with space for weights and request state. Set `GPU_UUID` to its UUID. The selected UUID is CUDA ordinal 0 in this process. The checkpoint defaults to `/data0/shared/Qwen3.8-27B-FP8`. [Generator build](../tools/build-probe.sh#L76)<!-- prefix: "\"$xmake\" bui" -->

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    "${SGLANG_GENERATE:?set the printed generator path}" \
    --device 0 --prompt-ids 760,6511,314,9338,369 --max-new-tokens 8
```

The [portable FP8 fixture](../benchmarks/vllm/small-fp8.json#L2)<!-- prefix: "\"prompt_toke" --> defines the prompt and run settings. Its [published vLLM result](../benchmarks/vllm/small-fp8-result.json#L20)<!-- prefix: "\"output_ids\"" --> records greedy IDs, finish reason, offline timing, and runtime settings. Expect `[11751, 13, 198, 760, 6511, 314, 9564, 369]` and `length`. Repeat with `--stop-token 198`. Expect `[11751, 13, 198]` and `stop`. Both requests must exit successfully.

Raw logits come from a separate capture. The frozen input is `artifacts/native-decoder-verification/fp8-oracle/run.4yNLfl/oracle.json`. Set `FP8_ORACLE_SHA256` to its recorded digest in `artifacts/native-decoder-verification/final-evidence.sha256`. To recapture, reserve the B200 selected by `artifacts/native-decoder-verification/fp8-oracle/run.sh`. Run `bash artifacts/native-decoder-verification/fp8-oracle/run.sh` on the project host. The launcher uses the isolated pinned vLLM environment and the `artifacts/decoder-oracle/` inputs pinned in its `plan.json`. Its `collect.py` captures two runs of the top 16 pre-sampler raw logits per step with `fp8_e4m3` GQA KV.

Run the [native verifier](../tests/decoder/run_native.py#L185)<!-- prefix: "def parse_ar" --> on the frozen input. Set `GPU_INDEX` to the physical index for `GPU_UUID`. The verifier requires the per-UUID lock file in the pinned vLLM environment and an idle B200 with at least 65,000 MiB free.

```bash
/data1/shared/dongwu.chen/.pixi/envs/default/bin/python -IB tests/decoder/run_native.py \
    --binary "${SGLANG_GENERATE:?set the printed generator path}" \
    --oracle artifacts/native-decoder-verification/fp8-oracle/run.4yNLfl/oracle.json \
    --oracle-sha256 "${FP8_ORACLE_SHA256:?set the recorded oracle digest}" \
    --gpu-index "${GPU_INDEX:?set the reserved B200 index}" \
    --gpu-uuid "${GPU_UUID:?set the reserved B200 UUID}" \
    --evidence "${NATIVE_EVIDENCE:?set a new evidence directory under artifacts/}"
```

The verifier checks exact token IDs and finish reasons across four cases. For each step, it finds reference token IDs among the candidate's top 64. The absolute tolerance is `0.5 + max(2 × BF16 ULP, 4 × repeat delta)` in raw logit units. It uses the first reference score for the ULP and the largest observed repeat difference for that token's delta. A missing reference score or an absolute difference above tolerance fails the check. [Score comparison](../tests/decoder/run_native.py#L95)<!-- prefix: "budget = 0.5" -->

For timing, omit `--top-logits` to avoid the host logit copy and ranking. Match `--warmup` and `--repeats` to the reference workload. The defaults are zero warmups and one measured request. `load_ms` covers model initialization. `request_ms` measures each completed request from allocation through generation and cleanup. Allocation metrics count decoder-owned device allocations, including preallocated request state. Keep the JSON, invocation, binary hash, and device UUID. These fixtures cover eager baseline correctness. W1 and W2 acceptance requires separate exact-output comparisons under the [benchmark agreement](performance.md). [Measurement and JSON fields](../apps/sglang-generate.cpp#L164)<!-- prefix: "Measurements" -->

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
| P3 · Text generation | Run the [native text decoder checks](#p3-native-text-decoder). | Compare logits and greedy token IDs with the reference. Verify the finish reason and sampling behavior. Record output tokens/s, latency, and peak memory. |
| P4 · Cache and MTP | Compare fresh execution with prefix hits, restore and replay, and native MTP on and off. Exercise zero, partial, and full acceptance, cancellation, eviction, and reuse. | Committed state and continuation agree with fresh target execution. Sampling preserves the target distribution. Rejected tokens never enter a shared prefix. Released storage is safe to reuse. |
| P5 · Serving and media | Use an independent client for text, image, and video requests. Exercise streams, all constraint modes, continuous batching, chunked prefill, cancellation, and a sustained concurrent run. | Verify the agreed API fixtures, media inputs and outputs, stop reasons, valid constrained outputs, and cleanup. Check for leaks and stale state across requests. |
| P6 · End-to-end performance | Profile a named workload. Change the limiting path. Repeat the baseline and candidate runs under the benchmark agreement. | Each required row meets its agreed throughput and latency targets and passes correctness. Retain raw samples, before/after profiles, and the decision to keep or revert the change. |

The proposed P0 time budget is three to five working days. Open questions need an owner and a discriminating experiment. Record raw runs in ignored `artifacts/` storage. Keep accepted decisions and reproducible procedures in the maintained documents.
