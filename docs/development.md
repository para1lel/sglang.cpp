# Single-B200 development contract

Build a C++ and CUDA C++ inference service for one B200. The [checkpoint investigation](research/checkpoint.md) records hashes, tensors, and SGLang sources for `/data0/shared/Qwen3.8-27B-FP8` at `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`.

## Scope at adoption

The initial `sm_100` probe returned 42 on B200 4 (`GPU-57c04c1a-9ae4-b231-f31a-5f4367e759a8`) after an XMake build with GCC 15.3.0 and CUDA 13.4.92.

| Part | Responsibility | Source |
| --- | --- | --- |
| Model | Define weights, mathematics, tokenization, media processing, and native multi-token prediction (MTP). | Official checkpoint, configuration, processor, tokenizer, and model code pinned in P0. |
| Service | Study scheduling, Radix prefix reuse, memory ownership, speculative verification, constraints, and streaming behavior. | Architectural and behavioral reference `sgl-project/sglang` at `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. |
| Performance | Reproduce a run independently and compare the candidate on each agreed workload. | The [benchmark agreement](performance.md) owns the vLLM pin, controls, and metrics. |
| Candidate | Build `src/` with XMake and default Pixi. Use `sglang/...` includes for headers under `include/sglang/`, with server headers in `include/sglang/server/`. | [Build options](../tools/build.sh#L5)<!-- prefix: "echo \"Usage:" --> and [XMake project](../xmake.lua#L2)<!-- prefix: "set_project(" -->. |

**Adopted scope.** Target one B200 with the local checkpoint and project-owned model and CUDA operators.

Record verified failures and preventive checks in [development lessons](lessons.md).

**Operator policy.** Production excludes cuBLAS, cuBLASLt, cuDNN, CUTLASS, FlashInfer, and equivalent operator libraries or generators. Record the justification, license, version, and execution path of HTTP, JSON, tokenizer, media, and system dependencies. Reference engines may use their own operators.

**Proposed scope.** Text generation, sampling, stopping, image and video input, MTP4, concurrent scheduling, and MXFP8 grouped-query attention (GQA) KV. Investigate GDN state and Radix prefix reuse. Verify checkpoint and service semantics before accepting OpenAI-compatible endpoints, streaming, cancellation, or JSON Schema, regex, and grammar constraints.

## First task and next evidence

From the repository root, set `GPU_INDEX` to the reserved B200's physical index.

```bash
bash tools/build.sh --gpu "${GPU_INDEX:?set the reserved B200 index}"
```

Check the `sm_100` image and runtime. Require `GPU returned 42 (expected 42)` and exit zero. Keep the command, UUID, toolchain identities, and binary hash.

For P0, use the [checkpoint investigation](research/checkpoint.md) for semantics and pinned SGLang mappings. Propose endpoints, API version, parameters, streaming events, and constraints. Record checkpoint conflicts.

The [baseline smoke](performance.md#historical-offline-smoke) found identical eight-ID greedy continuations across seven vLLM requests with BF16 KV and MTP off. Use the [GQA cache comparison](performance.md#gqa-cache-comparison) for paired runs.

### P1 runtime probe

Use managed Pixi. Set `GPU_INDEX` to the reserved B200's physical index.

```bash
bash tools/build.sh --build-dir build --nvrtc --gpu "${GPU_INDEX:?set the reserved B200 index}"
```

The script checks NVRTC inside Pixi and `libcuda.so.1` outside Pixi and exposes only the selected GPU. Check UUID, `compute_capability=10.0`, `architecture=sm_100`, and allocation size.

The [probe](../tools/nvrtc-probe.cpp#L265)<!-- prefix: "void run_pro" --> disables NVRTC's persistent cache. The cold lookup compiles and loads a cubin. The warm lookup reuses its function in the same CUDA context.

Expect these results in order:

- `phase=cold cache=miss compiled=1` with `compile_ms` for the NVRTC compilation call.
- `phase=warm cache=hit compiled=0` with `compile_ms=0.000000`.
- `NVRTC probe passed` followed by exit status zero.

Both phases verify all elements and report `verified_elements=257 first=42 last=298`. CUDA events measure `kernel_ms` after buffer reset and before the host copy.

Keep the command, output, allocation size, and device, library, and binary identities.

### P2 GQA cache contract

K and V use four 256-channel E4M3 heads and one E8M0 scale per 32 contiguous channels. E4M3 weights use BF16 inverse scales per 128×128 block. [Checkpoint investigation](research/checkpoint.md)

Each 128-token page holds 131,072 payload bytes and 4,096 scale bytes per tensor. The device table maps logical pages to distinct physical pages. [Cache layout](../include/sglang/mxfp8_gqa.h#L22)<!-- prefix: "/// Separate" --> · [Page table](../include/sglang/mxfp8_gqa.h#L34)<!-- prefix: "struct Mxfp8" -->

Link `sglang_cuda`. Apply RoPE to Q and K. Append contiguous BF16 K and V `[tokens, 4, 256]`. Pass the ready prefix length and first query position to `mxfp8_gqa_attention`. BF16 Q and output are `[queries, 24, 256]`. Row `r` sees through `first_query_position + r` inclusive. Check launch status. Retain storage through stream completion. [Library](../xmake.lua#L46)<!-- prefix: "target(\"sgla" --> · [Append](../include/sglang/mxfp8_gqa.h#L48)<!-- prefix: "cudaError_t " --> · [Attention](../include/sglang/mxfp8_gqa.h#L55)<!-- prefix: "cudaError_t " -->

For group maximum magnitude `amax`, clamp `ceil(log2(max(amax, 1e-30) / 448)) + 127` to `[0, 254]`. Direct attention decodes K and V and converts Q to FP32. [Scale calculation](../src/mxfp8_gqa.cu#L37)<!-- prefix: "__device__ i" --> · [Direct attention](../src/mxfp8_gqa.cu#L100)<!-- prefix: "__global__ v" -->

Exact scoring reduces eight groups in FP32, sums in group order, and scales by `1/16`. Ordered replay uses token-ordered FMAs and divides in FP32 before BF16 rounding. [Score](../src/mxfp8_gqa.cu#L221)<!-- prefix: "__device__ f" --> · [Replay](../src/mxfp8_gqa.cu#L405)<!-- prefix: "__device__ _" -->

[Singleton](batch-graphs.md#page-parallel-gqa-decode) and [packed decode](batch-graphs.md#packed-paged-gqa-decode) group softmax and value reductions by KV pages.
[Wide eager decode](long-paged-decode.md) selects its split after compaction.

#### Ordered multirow GQA prefill

Eligible tensor-core chunks use [Flash](flash-prefill.md#constraints-and-failure-behavior). Through 32,768 visible tokens, ordered fallback uses two-row tiles at 64–127 rows and four-row tiles at 128 or more. [Dispatch](../src/mxfp8_gqa_prefill.cu#L540)<!-- prefix: "cudaError_t " -->

Tiles share decoded K/V across rows and two query heads. Each 32-token block preserves causality, the FP32 score tree, token-ordered softmax and value updates, and BF16 rounding. [Scores](../src/mxfp8_gqa_prefill.cu#L37)<!-- prefix: "__device__ v" --> · [Tile loop](../src/mxfp8_gqa_prefill.cu#L160)<!-- prefix: "for (int fir" -->

Other shapes and tails use protected single-row attention. Both routes retain finite FP64 replacements for nonfinite visible scores per head and quotients per channel. [Recovery](../include/sglang/mxfp8_gqa.h#L63)<!-- prefix: "/// Shared a" -->

Zero rows succeed. Negative positions or rows and position-plus-row overflow otherwise return `cudaErrorInvalidValue`.

#### P2 GQA verification

```bash
bash tools/build.sh --build-dir build --mxfp8-check
```

Run from the repository root. Use the printed build directory and reserved B200 UUID.

```bash
export CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}"
checker="${BUILD_WORK_DIR:?set the printed build directory}/build/bin/mxfp8-gqa-check"
env -u SGLANG_EXPERIMENTAL_TMEM_PREFILL "$checker"
SGLANG_EXPERIMENTAL_TMEM_PREFILL=1 "$checker"
```

Expect `compute_capability=10.0`, `MXFP8 GQA check passed`, and exit zero in both [dispatch modes](flash-prefill.md#verification). The [verifier](../tests/mxfp8_gqa_check.cu#L639)<!-- prefix: "void run_che" --> covers append, attention, and page reuse with packed KV and BF16.

The [ordered decode check](../tests/mxfp8_gqa_check.cu#L541)<!-- prefix: "void check_e" --> covers BF16 parity, tile and page boundaries, graph and eager execution, input preservation, and validation.

Check [exponents 0 and 255](../tests/mxfp8_gqa_prefill_check.cu#L930)<!-- prefix: "void check_t" -->, [remaps and graphs](../tests/mxfp8_gqa_check.cu#L872)<!-- prefix: "void check_o" -->, [chunk edges and tails](../tests/mxfp8_gqa_prefill_check.cu#L698)<!-- prefix: "void check_o" -->, [rounding](../tests/mxfp8_gqa_prefill_check.cu#L746)<!-- prefix: "void check_o" -->, [32,768-token recovery](../tests/mxfp8_gqa_prefill_check.cu#L798)<!-- prefix: "void check_o" -->, and [masked-score recovery](../tests/mxfp8_gqa_check.cu#L935)<!-- prefix: "void check_o" -->.

### P3 native text decoder

#### Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| `sglang-generate` | Parse a token-ID prompt, run requests serially, and write JSON. | [CLI entry](../apps/sglang-generate.cpp#L281)<!-- prefix: "int main(int" --> |
| `sglang_checkpoint` | Validate the text configuration, index, and safetensors metadata. Map immutable shard bytes and expose checked tensor views. | [Loader API](../include/sglang/checkpoint.h#L22)<!-- prefix: "class Checkp" --> |
| `sglang_decoder` | Own device weights and run requests with `sglang_cuda` operators. | [Decoder API](../include/sglang/decoder.h#L104)<!-- prefix: "class Decode" --> · [Layer order](../include/sglang/decoder_internal.h#L198)<!-- prefix: "void execute" --> |

Checkpoint metadata and HTTP use `sglang_json`. Metadata parsing preserves historical underflow to signed zero and overflow to signed infinity, including in ignored fields. Required config values, index entries, tensor dimensions, and offsets still undergo validation. [Metadata parser](../src/checkpoint.cpp#L79)<!-- prefix: "json::Object" --> · [Runtime dependencies](text-serving.md#run-with-bundled-libraries) · [Link configuration](../xmake.lua#L90)<!-- prefix: "target(\"sgla" -->

#### Runtime path

The decoder validates tensor names, dtypes, and shapes before upload. Upload retains mappings and allocations until completion. Deferred cleanup retains shared weights. FP8 projections quantize BF16 inputs to E4M3 with one FP32 scale per 128 channels. [Weight upload](../src/decoder_weights.cpp#L98)<!-- prefix: "Decoder::Mod" --> · [Stream cleanup](../src/decoder_resources.cpp#L212)<!-- prefix: "CleanupError" --> · [Activation quantization](../src/text_ops.cu#L165)<!-- prefix: "__global__ v" -->

Serial K0 reuses its stream and grows storage on demand. Prefill rebuilds [MXFP8 pages](#p2-gqa-cache-contract) with an identity page table. Reset zeros BF16 convolution history and FP32 recurrent state across 48 GDN layers. [Generation loop](../src/decoder.cpp#L320)<!-- prefix: "void Decoder" --> · [State reset](../src/decoder_resources.cpp#L711)<!-- prefix: "void Request" -->

One model layer loop runs input RMSNorm, GQA or GDN, post-attention RMSNorm, then the MLP. The loop assigns separate KV and GDN state indexes. Tiled prefill and row-based eager, graph, and MTP verification passes supply their own buffers and operator dispatch. Embedding, draft seeding, and scoring stay with their callers. [Tile adapter](../src/decoder_prefill.cpp#L468)<!-- prefix: "execute_laye" --> · [Row adapter](../src/decoder_forward.cpp#L563)<!-- prefix: "void Decoder" -->

Prefill follows token order. The final prompt row gives initial logits. [Serial tiles](text-serving.md#serial-prefill) can exceed 128 rows. Serial K0 defaults to graph decode. Overlapping calls use private eager storage. [Prefill dispatch](../src/decoder_prefill.cpp#L419)<!-- prefix: "void Decoder" --> · [Decode graph](../src/decoder_forward.cpp#L524)<!-- prefix: "void Decoder" -->

Both binaries accept `--fp8-prefill exact|tensor-core`. Pass `Fp8PrefillPolicy` as the fourth `Decoder` argument. Default `exact` preserves SIMT arithmetic. Tensor cores can change logits and greedy IDs. [Policy](../include/sglang/decoder.h#L99)<!-- prefix: "enum class F" -->

Tensor-core projections require K0, at least three rows, widths divisible by 128, and contiguous, 16-byte-aligned E4M3 payloads. The `sm_100a` build uses TMA, TMEM, and `tcgen05`. It requires capability 10.0, a matching image, sufficient opt-in shared memory, and tensor maps. Ineligible calls use SIMT. Errors propagate. [Projection](../src/decoder_prefill.cpp#L315)<!-- prefix: "void Prefill" --> · [Eligibility](../src/prefill_fp8_sm100.cu#L467)<!-- prefix: "const int ma" --> · [Build](../xmake.lua#L65)<!-- prefix: "target(\"sgla" -->

At input widths ≥8192, eligible projections use 64×128 tiles below 1024 rows and 128×128 otherwise. [Projection tiles](../src/prefill_fp8_sm100.cu#L489)<!-- prefix: "const bool n" -->

Multirow K0 tensor-core prefill keeps these operators during projection fallback.

- [GDN prefill](gdn-prefill.md) prepares convolution, Q/K, and gates before ordered or chunked recurrence. [Preparation](../src/text_attention_gdn.cu#L76)<!-- prefix: "__global__ v" -->
- [Flash or ordered GQA](flash-prefill.md#runtime-path) follows KV append. [Call](../src/decoder_prefill.cpp#L267)<!-- prefix: "const auto s" -->
- Fused residual RMSNorm and SwiGLU round to BF16 before FP8 quantization. [RMSNorm](../src/text_normalization.cu#L236)<!-- prefix: "cudaError_t " --> · [SwiGLU](../src/text_ops.cu#L665)<!-- prefix: "cudaError_t " -->

Other target prefill and draft decode use direct attention. Wide long decode requires the tensor-core policy. Outside packed paged decode, eager decode uses exact attention within inline score capacity. Serial graphs use exact attention when page-rounded KV slots fit. Both use direct attention beyond these limits. [Row dispatch](../src/decoder_forward.cpp#L196)<!-- prefix: "cudaError_t " -->

The lane stream orders append, attention, and gating. Logits storage holds scores for 5,120 positions. See the [batch graph attention routes](batch-graphs.md#page-parallel-gqa-decode) for singleton eager fallback and multi-lane score allocation. [Workspace](../src/decoder_resources.cpp#L601)<!-- prefix: "ExactGqaWork" -->

Tuned FP8 projections require one target decode row in the forward group. This includes serial graphs, single-lane batch graphs, lone ordinary eager decode, and single-row verification. GQA combines Q/gate, K, and V. The MLP fusion rounds gate and up to BF16 before SwiGLU. GDN QKV and Z stay separate. [Lane selection](../src/decoder_forward.cpp#L51)<!-- prefix: "bool single_" --> · [Batch packing](../src/decoder_batch.cpp#L701)<!-- prefix: "if (rows[slo" -->

Tuned GEMV requires 16-byte-aligned FP8 weights and activations and input widths of 5,120, 6,144, or 17,408. Both fusions require the same alignment and width 5,120. Q/gate, K, and V require 12,288, 1,024, and 1,024 output rows. Gate and up require 17,408 each. Other valid inputs use separate operations. Output spans must be disjoint from inputs and each other. Fused SwiGLU permits identical gate and up spans. [Tuned GEMV](../src/text_decode_fp8.cu#L276)<!-- prefix: "cudaError_t " --> · [Fused QKV](../src/text_decode_fp8.cu#L298)<!-- prefix: "cudaError_t " --> · [Fused SwiGLU](../src/text_decode_fp8.cu#L321)<!-- prefix: "cudaError_t " --> · [Buffers](../include/sglang/decoder_cuda.cuh#L159)<!-- prefix: "// Read-only" -->

Row-wise prefill and MTP draft decode use GEMV. Each forward layer divides a multirow FP8 projection into balanced groups of at most five rows. Eligible groups of two to five use packed GEMM. Other groups use GEMV. Wide GEMV and packed GEMM require 512-divisible input widths and 16-byte-aligned FP8 payloads. [Grouping](../src/decoder_forward.cpp#L59)<!-- prefix: "int projecti" --> · [GEMV](../src/text_ops.cu#L674)<!-- prefix: "cudaError_t " --> · [GEMM](../src/text_ops.cu#L692)<!-- prefix: "cudaError_t " -->

Single-row BF16 RMSNorm uses vector loads at width 5,120 with 16-byte-aligned branch, gains, and optional residual. Other calls use generic kernels. [RMSNorm dispatch](../src/text_normalization.cu#L253)<!-- prefix: "cudaError_t " -->

Outside MTP verification, one target decode row can use warp quantization and combined GDN gates. Quantization needs whole 128-channel groups, 8-byte-aligned BF16 input, and 4-byte-aligned E4M3 values. It preserves FP32 round-to-nearest scale and reciprocal divisions and saturated E4M3 conversion. Other valid layouts use the generic quantizer. [Dispatch](../src/decoder_forward.cpp#L71)<!-- prefix: "void quantiz" --> · [Quantizer](../src/text_decode_bandwidth.cu#L247)<!-- prefix: "cudaError_t " -->

Combined GDN gates need matching 48-by-5,120 shapes. After the initial score, serial eager and graph decode can pair two vocabulary rows per warp. The paired LM head shares input loads across each pair and uses streaming weight loads. It needs a positive even row count. Both BF16 kernels require width 5,120 and 16-byte-aligned weights and input. They preserve two ordered FP32 FMA chains, their sum, ordered warp reduction, and BF16 rounding. Other valid shapes or alignments use the existing GEMVs. [GDN gates](../src/text_decode_bandwidth.cu#L194)<!-- prefix: "cudaError_t " --> · [Serial LM head](../src/text_decode_bandwidth.cu#L215)<!-- prefix: "cudaError_t " -->

The [width-four batch route](batch-graphs.md#runtime-path) uses a paired BF16 LM head that shares weight reads across four inputs and preserves the same per-output accumulation and rounding. It requires a positive even vocabulary size, width 5,120, and 16-byte-aligned weights and inputs. Other shapes and alignments use the generic BF16 batch projection. [Grouped LM head](../src/text_decode_bandwidth.cu#L227)<!-- prefix: "cudaError_t " -->

#### Constraints and failure behavior

Pass one nonempty prompt with `--prompt-ids` or `--prompt-file`. Separate decimal IDs with commas or ASCII whitespace. Prompt and stop IDs must be in `[0, 248320)`. Prompt length plus budget must fit 262,144 tokens. The default budget is one. Zero budget returns no tokens and `length`. [CLI options](../apps/sglang-generate.cpp#L34)<!-- prefix: "void usage(s" --> · [Request validation](../src/decoder.cpp#L61)<!-- prefix: "void validat" -->

The CLI defaults to greedy selection, with ties going to the lowest ID. `--stop-token` emits a match and returns `stop`, even at the budget. Exhaustion returns `length`. There is no implicit EOS. Failures print to stderr and exit 1. See [sampling](text-serving.md#sampling) and [C++ and HTTP constraints](text-serving.md#finite-choice-constraints). [Greedy selection](../src/text_ops.cu#L467)<!-- prefix: "__device__ T" -->

The decoder retains CUDA resources and transfer buffers while completion is uncertain. Model and request setup retry cleanup on the same device. Persistent errors block setup. Terminal errors and ambiguous frees quarantine resources until exit. [Deferred cleanup](../src/decoder_resources.cpp#L116)<!-- prefix: "static void " --> · [Destructor diagnostics](../src/decoder_resources.cpp#L380)<!-- prefix: "Allocations:" -->

Private CUDA pools separate workspaces at or below 128 rows from larger ones. One pool per device and class can set a release threshold. The thresholds are 32 MiB and 512 MiB respectively. Other pools use CUDA defaults. [Retention](../src/decoder_prefill.cpp#L43)<!-- prefix: "void configu" -->

Single rows and unavailable normal scratch use row-wise prefill. Release queues a stream-ordered free. After enqueue failure, recovery synchronizes the device, checks used bytes, and waits for any retried free. Explicit close reports the original error. [Generation](../src/decoder.cpp#L435)<!-- prefix: "reclaim_pref" -->, [batch admission](../src/decoder_batch.cpp#L521)<!-- prefix: "reclaim_pref" -->, and multirow allocation retry pending frees. [Allocation](../src/decoder_prefill.cpp#L346)<!-- prefix: "PrefillWorks" --> · [Release](../src/decoder_prefill.cpp#L401)<!-- prefix: "cudaError_t " -->

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

Run the [exact prefill checks](../tests/prefill/README.md). Match the frozen oracle. Check the default FP8 prefill policy with the 511-token probe. Check chunk boundaries, mixed steps, reset, and slot reuse. Use the [tensor-core comparison](performance.md#fp8-tensor-core-prefill) for the opt-in mode.

#### Native MTP

See [MTP](mtp.md) for exact tiled seeding, joint target verification, and per-lane rollback. [Verify MTP](mtp.md#verification) before timing.

### Native streaming server

See [native HTTP transport](http-server.md) for request admission, overload responses, health, readiness, and shutdown. Set `--max-active` to a value from 1 to 32. Set `--max-num-batched-tokens` to a value from 1 to 32,768. Both default to the maximum. [MTP](mtp.md#runtime-path) requires prefix reuse off. Graph replay is [opt-in](batch-graphs.md). [CLI limits](../apps/sglang-server.cpp#L99)<!-- prefix: "} else if (o" -->

Lanes share weights and own KV, GDN state, and outputs. [Resident memory admission](batch-graphs.md#resident-memory-admission) applies at every capacity.

Follow the [benchmark token-ID wire contract](../benchmarks/streaming/README.md#wire-contract) and [decoder limits](#constraints-and-failure-behavior). A matching `stop_token_ids` entry emits the ID with `stop`, including at the budget. Zero budget reports `length` without a model step. Benchmark temperature accepts underflow to signed zero. [Request parser](../apps/server/completions.cpp#L27)<!-- prefix: "CompletionRe" --> · [Numeric retry](../apps/server/completions.cpp#L176)<!-- prefix: "json::Value " -->

#### Streaming verification

Run the [CPU transport checks](http-server.md#verification).

Run the [scheduler admission checks](batch-graphs.md#verification) on a reserved B200.

Build the release server. Set `GPU_UUID` to the reserved B200 UUID. Put `nsys` on `PATH`. Use a new evidence directory.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    /data1/shared/dongwu.chen/.pixi/envs/default/bin/python tests/batch/run_native.py \
    --binary build/bin/sglang-server --evidence artifacts/native-batch-check
```

Require `passed` in `status.json`. Check output parity at widths one, two, and four. With graph replay enabled, check one graph launch per eligible decode step. Check full decode width with mixed stops and at a KV page crossing. Check queueing and reset recovery with readable peers, exact replacements, and zero request bytes. Keep [runner](../tests/batch/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in li" --> identity and raw results. Run the [streaming smoke](../tests/test_native_streaming.py#L1)<!-- prefix: "\"\"\"Loopback " --> with `--port PORT` for zero budget, half-close, zero temperature forms, and strict JSON.

For capacities 16 and 32, follow the [wide K0 runbook](../tests/batch/README.md#wide-k0-serving).

## Runtime and state ownership

Design one execution path for all supported feature combinations, including a batch of one. Put specialized behavior at the smallest applicable boundary, such as operator parameters or launch configuration. Consolidate existing duplicate end-to-end logic into the shared path. Base dispatch on computation shapes and hardware resource limits. Verify several valid shapes. Keep benchmark identity, fixed prompt contents, and scenario labels out of dispatch decisions.

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
