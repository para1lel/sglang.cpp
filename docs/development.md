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

**Adopted scope.** Target one B200 with the local checkpoint and project-owned model and CUDA operators.

Record verified failures and preventive checks in [development lessons](lessons.md).

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

K and V use four 256-channel E4M3 heads and one E8M0 scale per 32 contiguous channels. E4M3 weights use BF16 inverse scales per 128×128 block. [Checkpoint investigation](research/checkpoint.md)

Each 128-token page holds 131,072 payload bytes and 4,096 scale bytes per tensor. The device table maps logical pages to distinct physical pages. [Cache layout](../include/sglang/mxfp8_gqa.h#L22)<!-- prefix: "/// Separate" --> · [Page table](../include/sglang/mxfp8_gqa.h#L34)<!-- prefix: "struct Mxfp8" -->

Link `sglang_cuda`. Apply RoPE to Q and K. Append contiguous BF16 K and V `[tokens, 4, 256]`. Pass the ready prefix length and first query position to `mxfp8_gqa_attention`. BF16 Q and output are `[queries, 24, 256]`. Row `r` sees through `first_query_position + r` inclusive. Check launch status. Retain storage through stream completion. [Library](../xmake.lua#L46)<!-- prefix: "target(\"sgla" --> · [Append](../include/sglang/mxfp8_gqa.h#L48)<!-- prefix: "cudaError_t " --> · [Attention](../include/sglang/mxfp8_gqa.h#L55)<!-- prefix: "cudaError_t " -->

For group maximum magnitude `amax`, clamp `ceil(log2(max(amax, 1e-30) / 448)) + 127` to `[0, 254]`. Direct attention decodes K and V and converts Q to FP32. [Scale calculation](../src/mxfp8_gqa.cu#L36)<!-- prefix: "__device__ i" --> · [Direct attention](../src/mxfp8_gqa.cu#L99)<!-- prefix: "__global__ v" -->

Exact scoring reduces eight groups in FP32, sums in group order, and scales by `1/16`. Ordered replay uses token-ordered FMAs and divides in FP32 before BF16 rounding. [Score](../src/mxfp8_gqa.cu#L220)<!-- prefix: "__device__ f" --> · [Replay](../src/mxfp8_gqa.cu#L404)<!-- prefix: "__device__ _" -->

[Singleton](batch-graphs.md#page-parallel-gqa-decode) and [packed decode](batch-graphs.md#packed-paged-gqa-decode) group softmax and value reductions by KV pages.

#### Ordered multirow GQA prefill

The tensor-core projection policy pairs at least 64 rows within 8,192 visible tokens. [Dispatch](../src/mxfp8_gqa_prefill.cu#L441)<!-- prefix: "cudaError_t " -->

Each block shares decoded K/V across two rows and two heads. Its 32-token tiles preserve ordered FP32 arithmetic and per-row causality. [Tile loop](../src/mxfp8_gqa_prefill.cu#L371)<!-- prefix: "for (int fir" -->

Other valid shapes and odd tails use one-row attention. Both paths replay nonfinite visible scores by head and nonfinite quotients by channel in FP64. They keep finite FP64 replacements. [Recovery](../include/sglang/mxfp8_gqa.h#L63)<!-- prefix: "/// Both pai" -->

Zero rows succeed. Otherwise, negative positions, negative rows, and position-plus-row overflow return `cudaErrorInvalidValue`.

#### P2 GQA verification

Build from the repository root with Pixi.

```bash
bash tools/build.sh --build-dir build --mxfp8-check
```

Set `BUILD_WORK_DIR` to the printed directory and `GPU_UUID` to the reserved B200 UUID.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    "${BUILD_WORK_DIR:?set the printed build directory}/build/bin/mxfp8-gqa-check"
```

Expect `compute_capability=10.0`, `MXFP8 GQA check passed`, and exit zero. The [verifier](../tests/mxfp8_gqa_check.cu#L640)<!-- prefix: "void run_che" --> checks append, prefill, decode, and page reuse against packed KV and BF16 references.

The [ordered decode check](../tests/mxfp8_gqa_check.cu#L542)<!-- prefix: "void check_e" --> covers BF16 parity, tile and page boundaries, eager and graph execution, input preservation, bounds, and arguments.

The [tiled scale check](../tests/mxfp8_gqa_prefill_check.cu#L847)<!-- prefix: "void check_t" --> covers exponents 0 and 255. Prefill tests cover [remapping and graphs](../tests/mxfp8_gqa_check.cu#L873)<!-- prefix: "void check_o" -->, [dispatch edges](../tests/mxfp8_gqa_prefill_check.cu#L639)<!-- prefix: "void check_o" -->, [rounding](../tests/mxfp8_gqa_prefill_check.cu#L681)<!-- prefix: "void check_o" -->, and [score recovery](../tests/mxfp8_gqa_check.cu#L936)<!-- prefix: "void check_o" -->.

### P3 native text decoder

#### Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| `sglang-generate` | Parse a token-ID prompt, run requests serially, and write JSON. | [CLI entry](../apps/sglang-generate.cpp#L281)<!-- prefix: "int main(int" --> |
| `sglang_checkpoint` | Validate the text configuration, index, and safetensors metadata. Map immutable shard bytes and expose checked tensor views. | [Loader API](../include/sglang/checkpoint.h#L22)<!-- prefix: "class Checkp" --> |
| `sglang_decoder` | Own device weights and run requests with `sglang_cuda` operators. | [Decoder API](../include/sglang/decoder.h#L102)<!-- prefix: "class Decode" --> · [Layer dispatch](../src/decoder_forward.cpp#L451)<!-- prefix: "void Decoder" --> |

Checkpoint metadata and HTTP use `sglang_json`. Metadata parsing preserves historical underflow to signed zero and overflow to signed infinity, including in ignored fields. Required config values, index entries, tensor dimensions, and offsets still undergo validation. [Metadata parser](../src/checkpoint.cpp#L79)<!-- prefix: "json::Object" --> · [Runtime dependencies](text-serving.md#run-with-bundled-libraries) · [Link configuration](../xmake.lua#L89)<!-- prefix: "target(\"sgla" -->

#### Runtime path

The decoder validates tensor names, dtypes, and shapes before upload. Upload retains mappings and allocations until completion. Deferred cleanup retains shared weights. FP8 projections quantize BF16 inputs to E4M3 with one FP32 scale per 128 channels. [Weight upload](../src/decoder_weights.cpp#L98)<!-- prefix: "Decoder::Mod" --> · [Stream cleanup](../src/decoder_resources.cpp#L211)<!-- prefix: "CleanupError" --> · [Activation quantization](../src/text_ops.cu#L165)<!-- prefix: "__global__ v" -->

Serial K0 reuses its stream and grows storage on demand. Prefill rebuilds [MXFP8 pages](#p2-gqa-cache-contract) with an identity page table. Reset zeros BF16 convolution history and FP32 recurrent state across 48 GDN layers. [Generation loop](../src/decoder.cpp#L525)<!-- prefix: "void Decoder" --> · [State reset](../src/decoder_resources.cpp#L686)<!-- prefix: "void Request" -->

Prefill follows token order. The final prompt row gives initial logits. [Serial tiles](text-serving.md#serial-prefill) can exceed 128 rows. Serial K0 defaults to graph decode. Overlapping calls use private eager storage. [Prefill dispatch](../src/decoder_prefill.cpp#L417)<!-- prefix: "void Decoder" --> · [Decode graph](../src/decoder_forward.cpp#L420)<!-- prefix: "void Decoder" -->

Both binaries accept `--fp8-prefill exact|tensor-core`. Pass `Fp8PrefillPolicy` as the fourth `Decoder` argument. Default `exact` preserves SIMT arithmetic. Tensor cores can change logits and greedy IDs. [Policy](../include/sglang/decoder.h#L97)<!-- prefix: "enum class F" -->

Tensor-core projections require K0, at least three rows, widths divisible by 128, and contiguous, 16-byte-aligned E4M3 payloads. The `sm_100a` build uses TMA, TMEM, and `tcgen05`. It requires capability 10.0, a matching image, sufficient opt-in shared memory, and tensor maps. Ineligible calls use SIMT. Errors propagate. [Projection](../src/decoder_prefill.cpp#L313)<!-- prefix: "void Prefill" --> · [Eligibility](../src/prefill_fp8_sm100.cu#L467)<!-- prefix: "const int ma" --> · [Build](../xmake.lua#L64)<!-- prefix: "target(\"sgla" -->

At input widths ≥8192, eligible projections use 64×128 tiles below 1024 rows and 128×128 otherwise. [Projection tiles](../src/prefill_fp8_sm100.cu#L489)<!-- prefix: "const bool n" -->

Multirow K0 tensor-core prefill keeps these operators during projection fallback.

- [GDN prefill](gdn-prefill.md) fuses convolution, Q/K normalization, and gate preparation. Recurrence keeps token order. [Preparation](../src/text_attention_gdn.cu#L75)<!-- prefix: "__global__ v" -->
- [Ordered multirow GQA](#ordered-multirow-gqa-prefill) follows KV append. [Call](../src/decoder_prefill.cpp#L265)<!-- prefix: "const auto s" -->
- Fused residual RMSNorm and SwiGLU round to BF16 before FP8 quantization. [RMSNorm](../src/text_normalization.cu#L236)<!-- prefix: "cudaError_t " --> · [SwiGLU](../src/text_ops.cu#L665)<!-- prefix: "cudaError_t " -->

Other target prefill and draft decode use direct attention. Both prefill policies share target decode. Outside packed paged decode, eager decode uses exact attention within inline score capacity. Serial graphs use exact attention when page-rounded KV slots fit. Beyond these limits both use direct attention. [Row dispatch](../src/decoder_forward.cpp#L179)<!-- prefix: "cudaError_t " -->

The lane stream orders append, attention, and gating. Logits storage holds scores for 5,120 positions. See the [batch graph attention routes](batch-graphs.md#page-parallel-gqa-decode) for singleton eager fallback and multi-lane score allocation. [Workspace](../src/decoder_resources.cpp#L583)<!-- prefix: "ExactGqaWork" -->

Tuned FP8 projections require one target decode row in the forward group. This includes serial graphs, single-lane batch graphs, lone ordinary eager decode, and single-row verification. GQA combines Q/gate, K, and V. The MLP fusion rounds gate and up to BF16 before SwiGLU. GDN QKV and Z stay separate. [Lane selection](../src/decoder_forward.cpp#L50)<!-- prefix: "bool single_" --> · [Batch packing](../src/decoder_batch.cpp#L717)<!-- prefix: "if (rows[slo" -->

Tuned GEMV requires 16-byte-aligned FP8 weights and activations and input widths of 5,120, 6,144, or 17,408. Both fusions require the same alignment and width 5,120. Q/gate, K, and V require 12,288, 1,024, and 1,024 output rows. Gate and up require 17,408 each. Other valid inputs use separate operations. Output spans must be disjoint from inputs and each other. Fused SwiGLU permits identical gate and up spans. [Tuned GEMV](../src/text_decode_fp8.cu#L276)<!-- prefix: "cudaError_t " --> · [Fused QKV](../src/text_decode_fp8.cu#L298)<!-- prefix: "cudaError_t " --> · [Fused SwiGLU](../src/text_decode_fp8.cu#L321)<!-- prefix: "cudaError_t " --> · [Buffers](../include/sglang/decoder_cuda.cuh#L140)<!-- prefix: "// Read-only" -->

Row-wise prefill and MTP draft decode use GEMV. Each forward layer divides a multirow FP8 projection into balanced groups of at most five rows. Eligible groups of two to five use packed GEMM. Other groups use GEMV. Wide GEMV and packed GEMM require 512-divisible input widths and 16-byte-aligned FP8 payloads. [Grouping](../src/decoder_forward.cpp#L58)<!-- prefix: "int projecti" --> · [GEMV](../src/text_ops.cu#L674)<!-- prefix: "cudaError_t " --> · [GEMM](../src/text_ops.cu#L692)<!-- prefix: "cudaError_t " -->

Single-row BF16 RMSNorm uses vector loads at width 5,120 with 16-byte-aligned branch, gains, and optional residual. Other calls use generic kernels. [RMSNorm dispatch](../src/text_normalization.cu#L253)<!-- prefix: "cudaError_t " -->

Outside MTP verification, one target decode row can use warp quantization and combined GDN gates. Quantization needs whole 128-channel groups, 8-byte-aligned BF16 input, and 4-byte-aligned E4M3 values. It preserves FP32 round-to-nearest scale and reciprocal divisions and saturated E4M3 conversion. Other valid layouts use the generic quantizer. [Dispatch](../src/decoder_forward.cpp#L70)<!-- prefix: "void quantiz" --> · [Quantizer](../src/text_decode_bandwidth.cu#L187)<!-- prefix: "cudaError_t " -->

Combined GDN gates need matching 48-by-5,120 shapes. After the initial score, serial eager and graph decode can pair two vocabulary rows per warp. The paired LM head shares input loads across each pair and uses streaming weight loads. It needs a positive even row count. Both BF16 kernels require width 5,120 and 16-byte-aligned weights and input. They preserve two ordered FP32 FMA chains, their sum, ordered warp reduction, and BF16 rounding. Other valid shapes or alignments use the existing GEMVs. [GDN gates](../src/text_decode_bandwidth.cu#L154)<!-- prefix: "cudaError_t " --> · [LM head](../src/text_decode_bandwidth.cu#L175)<!-- prefix: "cudaError_t " -->

#### Constraints and failure behavior

Pass one nonempty prompt with `--prompt-ids` or `--prompt-file`. Separate decimal IDs with commas or ASCII whitespace. Prompt and stop IDs must be in `[0, 248320)`. Prompt length plus budget must fit 262,144 tokens. The default budget is one. Zero budget returns no tokens and `length`. [CLI options](../apps/sglang-generate.cpp#L34)<!-- prefix: "void usage(s" --> · [Request validation](../src/decoder.cpp#L64)<!-- prefix: "void validat" -->

The CLI defaults to greedy selection, with ties going to the lowest ID. `--stop-token` emits a match and returns `stop`, even at the budget. Exhaustion returns `length`. There is no implicit EOS. Failures print to stderr and exit 1. See [sampling](text-serving.md#sampling) and [C++ and HTTP constraints](text-serving.md#finite-choice-constraints). [Greedy selection](../src/text_ops.cu#L467)<!-- prefix: "__device__ T" -->

The decoder retains CUDA resources and transfer buffers while completion is uncertain. Model and request setup retry cleanup on the same device. Persistent errors block setup. Terminal errors and ambiguous frees quarantine resources until exit. [Deferred cleanup](../src/decoder_resources.cpp#L116)<!-- prefix: "static void " --> · [Destructor diagnostics](../src/decoder_resources.cpp#L379)<!-- prefix: "Allocations:" -->

Private CUDA pools separate workspaces at or below 128 rows from larger ones. One pool per device and class can set a release threshold. The thresholds are 32 MiB and 512 MiB respectively. Other pools use CUDA defaults. [Retention](../src/decoder_prefill.cpp#L43)<!-- prefix: "void configu" -->

Single rows and unavailable normal scratch use row-wise prefill. Release queues a stream-ordered free. After enqueue failure, recovery synchronizes the device, checks used bytes, and waits for any retried free. Explicit close reports the original error. [Generation](../src/decoder.cpp#L640)<!-- prefix: "reclaim_pref" -->, [batch admission](../src/decoder_batch.cpp#L543)<!-- prefix: "reclaim_pref" -->, and multirow allocation retry pending frees. [Allocation](../src/decoder_prefill.cpp#L344)<!-- prefix: "PrefillWorks" --> · [Release](../src/decoder_prefill.cpp#L399)<!-- prefix: "cudaError_t " -->

`--top-logits N` reports up to `N` raw scores per token, sorted by descending score then ascending ID. The range is 0–248,320 and defaults to 0. [Logit output](../apps/sglang-generate.cpp#L247)<!-- prefix: "void write_r" -->

#### Verification

Compare layer outputs and state on model shapes with frozen independent inputs and precision-aware tolerances. Require exact tensor names, shapes, dtypes, scales, and packed bytes. Run the [checkpoint metadata checks](../tests/decoder/README.md#checkpoint-metadata).

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

#### Native MTP

See [MTP](mtp.md) for exact tiled seeding, serial and joint target verification, and per-lane rollback. [Verify MTP](mtp.md#verification) before timing.

### Native streaming server

Follow the [server launch procedure](../README.md#serve-token-ids-over-http). Wait for `GET /ready` to return HTTP 200 with `{"ready":true}`. After permanent worker failure, it returns HTTP 503 with `{"ready":false,"reason":"worker_failed"}`. `GET /health` continues to return HTTP 200 with the configured limits. The defaults are 32 active requests and 32,768 target rows per step. The ranges are 1–32 and 1–32,768. Send both GET requests with an empty body. [Readiness response](../apps/server/http.cpp#L314)<!-- prefix: "void write_r" --> · [Health response](../apps/sglang-server.cpp#L128)<!-- prefix: "std::string " -->

Use `--max-active 1` with prefix reuse off for serial serving. Serial K0 defaults to graph replay. `--eager` requires the serial route and K0. [MTP](mtp.md#runtime-path) supports the full capacity range with prefix reuse off. Batch replay is [opt-in](batch-graphs.md). Extra connections can wait in the TCP backlog. Prefix reuse also enables a [parsed completion queue and controls](prefix-cache.md#native-http-controls). [CLI limits](../apps/sglang-server.cpp#L90)<!-- prefix: "} else if (o" --> · [Socket workers](../apps/sglang-server.cpp#L183)<!-- prefix: "void serve_b" -->

Batch lanes share weights and own KV, GDN state, and outputs. The batch applies [resident memory admission](batch-graphs.md#resident-memory-admission) with the context and step limits. Serial serving uses its own allocations. A serial allocation error can return HTTP 500 while readiness stays true. A permanent worker failure requires a restart. New completions return HTTP 500 after permanent failure. [Serial failure](../apps/server/completions.cpp#L242)<!-- prefix: "DecoderResul" --> · [Batch failure](../apps/server/batch_service.cpp#L183)<!-- prefix: "void BatchSe" -->

Follow the [benchmark token-ID wire contract](../benchmarks/streaming/README.md#wire-contract) and [decoder limits](#constraints-and-failure-behavior). Optional `stop_token_ids` takes one ID. A match emits it and reports `stop`, including at the budget. Zero budget reports `length` without a model step. Benchmark temperature accepts values that underflow to signed zero. [Request parser](../apps/server/completions.cpp#L28)<!-- prefix: "CompletionRe" --> · [Numeric retry](../apps/server/completions.cpp#L185)<!-- prefix: "json::Value " -->

A reset or failed write cancels that lane. The batch retires its storage before slot reuse. Ready peers can advance while a writer blocks. Clients can use `shutdown(SHUT_WR)` after a complete request. [Cancellation](../apps/server/batch_service.cpp#L428)<!-- prefix: "void BatchSe" --> · [Disconnect check](../apps/server/http.cpp#L263)<!-- prefix: "bool disconn" -->

#### Streaming verification

Run the [scheduler admission checks](batch-graphs.md#verification) on a reserved B200.

Build the release server. Set `GPU_UUID` to the reserved B200 UUID. Put `nsys` on `PATH`. Use a new evidence directory.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    /data1/shared/dongwu.chen/.pixi/envs/default/bin/python tests/batch/run_native.py \
    --binary build/bin/sglang-server --evidence artifacts/native-batch-check
```

Require `passed` in `status.json`. Check serial oracle streams and one graph launch per output after the first. At widths two and four, check exact streams, full decode on mixed-stop, and full width on page crossing. Check queueing and reset recovery with readable peers, exact replacements, and zero request bytes. Keep [runner](../tests/batch/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in li" --> identity and raw results. Run the [streaming smoke](../tests/test_native_streaming.py#L1)<!-- prefix: "\"\"\"Loopback " --> with `--port PORT` for zero budget, half-close, zero temperature forms, and strict JSON.

For capacities 16 and 32, follow the [wide K0 runbook](../tests/batch/README.md#wide-k0-serving).

## Runtime and state ownership

Verify [prefix checkpoints](prefix-cache.md#verification), [MTP state](mtp.md#verification), and stream completion before storage reuse.

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
