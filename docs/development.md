# Native B200 development guide

Build a C++ and CUDA C++ inference service for one B200. The [checkpoint investigation](research/checkpoint.md) records hashes, tensors, and SGLang sources for `/data0/shared/Qwen3.8-27B-FP8` at `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`.

<a id="scope-at-adoption"></a>
## Current architecture and scope

The supported deployment is one physical B200 with the exact checkpoint above. The decoder accepts `sm_100` devices, but only the exact `NVIDIA B200` device name enables the B200 operator set; other `sm_100` devices have no supported optimized path.

| Part | Responsibility | Source |
| --- | --- | --- |
| Model | Qwen3.8-27B-FP8 text and multimodal decoding with native MTP4. | [Qwen model](../include/sglang/runtime/model/qwen_text_model.h#L5)<!-- prefix: "struct QwenT" --> |
| Backend | Derive B200 operators, prefill, graph, split, and residency policy. | [Backend policy](../include/sglang/runtime/backend/cuda_backend_policy.h#L12)<!-- prefix: "struct CudaB" --> · [Execution policy](../include/sglang/runtime/execution/decoder_execution_policy.h#L5)<!-- prefix: "struct Decod" --> · [Residency policy](../include/sglang/runtime/cache/kv_residency_policy.h#L15)<!-- prefix: "class KvResi" --> |
| Service | OpenAI-compatible text and chat serving, streaming, cancellation, constraints, concurrent scheduling, and exact prefix reuse. | [HTTP transport](http-server.md) · [Text API](text-serving.md) · [Chat API](chat-serving.md) · [Prefix cache](prefix-cache.md) |
| Build | Use XMake, default Pixi, user CUDA 13.4 compatibility libraries, and the host driver. | [Build options](../tools/build.sh#L5)<!-- prefix: "echo \"Usage:" --> · [XMake project](../xmake.lua#L2)<!-- prefix: "set_project(" --> |

Production CUDA operators must use kernels authored in this project or open-source components distributed with the CUDA Toolkit. The native runtime excludes cuBLAS, cuBLASLt, and cuDNN. Reference engines may use their own operators. Record the justification, license, version, and execution path of HTTP, JSON, tokenizer, media, and system dependencies. Record failures and preventive checks in [development lessons](lessons.md). Use [isolated XMake builds](serving/isolated-builds.md) for private verification.

## Current matched K4 status

The [matched K4 report](serving/paired-native-defaults-results.md) records timing passes on W1 and W2. Formal W3–W6 remain unmeasured.

Run one workload at a time on the same reserved B200 with a verified Release binary:

```bash
export PAIRED_CHECKPOINT_MANIFEST=/path/to/published/checkpoint.sha256
export PAIRED_NATIVE_SOURCE='verified-40-character-source-revision'
export PAIRED_NATIVE_SHA256='verified-64-character-binary-digest'
bash benchmarks/paired/run_k4.sh /path/to/sglang-server 2 artifacts/paired-k4 W1
```

The default `server-defaults` profile resolves MTP4, 32 active requests, an 8,192-token scheduling cap, CUDA graphs, and prefix caching off. Use W2 through W6 in order only after each prior row passes. The [matched K4 recipe](../benchmarks/paired/README.md#matched-k4-timing-protocol) owns the full protocol and evidence requirements. The detailed historical W1–W6 K0 report and ledger are retained in pinned Git history at `8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3`.

<a id="p2-gqa-cache-contract"></a>
## CUDA operator contracts

K and V use four 256-channel E4M3 heads and one E8M0 scale per 32 contiguous channels. E4M3 weights use BF16 inverse scales per 128×128 block. [Checkpoint investigation](research/checkpoint.md)

Each 128-token page holds 131,072 payload bytes and 4,096 scale bytes per tensor. The device table maps logical pages to distinct physical pages. [Cache layout](../include/sglang/mxfp8_gqa.h#L22)<!-- prefix: "/// Separate" --> · [Page table](../include/sglang/mxfp8_gqa.h#L34)<!-- prefix: "struct Mxfp8" -->

Link `sglang_cuda`. Apply RoPE to Q and K. Append contiguous BF16 K and V `[tokens, 4, 256]`. Pass the ready prefix length and first query position to `mxfp8_gqa_attention`. BF16 Q and output are `[queries, 24, 256]`. Row `r` sees through `first_query_position + r` inclusive. Check launch status. Retain storage through stream completion. [Library](../xmake.lua#L46)<!-- prefix: "target(\"sgla" --> · [Append](../include/sglang/mxfp8_gqa.h#L48)<!-- prefix: "cudaError_t " --> · [Attention](../include/sglang/mxfp8_gqa.h#L55)<!-- prefix: "cudaError_t " -->

For group maximum magnitude `amax`, clamp `ceil(log2(max(amax, 1e-30) / 448)) + 127` to `[0, 254]`. Direct attention decodes K and V and converts Q to FP32. [Scale calculation](../src/mxfp8_gqa.cu#L40)<!-- prefix: "__device__ i" --> · [Direct attention](../src/mxfp8_gqa.cu#L114)<!-- prefix: "__global__ v" -->

Exact scoring reduces eight groups in FP32, sums in group order, and scales by `1/16`. Ordered replay uses token-ordered FMAs and divides in FP32 before BF16 rounding. [Score](../src/mxfp8_gqa.cu#L226)<!-- prefix: "__device__ f" --> · [Replay](../src/mxfp8_gqa.cu#L416)<!-- prefix: "__device__ _" -->

[Singleton](batch-graphs.md#page-parallel-gqa-decode) and [legacy narrow decode](batch-graphs.md#legacy-narrow-paged-gqa-decode) group softmax and value reductions by KV pages.
[Fused packed decode](long-paged-decode.md#fused-packed-gqa) groups reductions by 64-token chunks. [Legacy wide eager decode](long-paged-decode.md#legacy-wide-eager-decode) selects its split after compaction.

### Ordered multirow GQA prefill

The [ordered prefill contract](ordered-gqa-prefill.md) covers dispatch, tile selection, causal scoring, and FP64 recovery.

<a id="p2-gqa-verification"></a>
### GQA verification

```bash
bash tools/build.sh --build-dir build --mxfp8-check
```

Run from the repository root. Use the printed build directory and reserved B200 UUID.

```bash
export CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}"
checker="${BUILD_WORK_DIR:?set the printed build directory}/build/bin/mxfp8-gqa-check"
"$checker"
```

Expect `compute_capability=10.0`, `MXFP8 GQA check passed`, and exit zero. The
[Flash checks](flash-prefill.md#verification) cover automatic dispatch,
regular Flash, split graph nodes, and ordered replay. The [verifier](../tests/mxfp8_gqa_check.cu#L639)<!-- prefix: "void run_che" --> covers append, attention, and page reuse with packed KV and BF16.

The [ordered decode check](../tests/mxfp8_gqa_check.cu#L541)<!-- prefix: "void check_e" --> covers BF16 parity, tile and page boundaries, graph and eager execution, input preservation, and validation.

Check [exponents 0 and 255](../tests/mxfp8_gqa_prefill_check.cu#L931)<!-- prefix: "void check_t" -->, [remaps and graphs](../tests/mxfp8_gqa_check.cu#L872)<!-- prefix: "void check_o" -->, [chunk edges and tails](../tests/mxfp8_gqa_prefill_check.cu#L698)<!-- prefix: "void check_o" -->, [rounding](../tests/mxfp8_gqa_prefill_check.cu#L746)<!-- prefix: "void check_o" -->, and [masked-score recovery](../tests/mxfp8_gqa_check.cu#L935)<!-- prefix: "void check_o" -->.

Use the [ordered prefill replay checks](ordered-gqa-prefill.md#verification) for staged tails and overflow controls.

## Native text decoder

#### Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| `sglang-generate` | Parse a token-ID prompt, run requests serially, and write JSON. | [CLI entry](../apps/sglang-generate.cpp#L279)<!-- prefix: "int main(int" --> |
| `sglang_checkpoint` | Validate the text configuration, index, and safetensors metadata. Map immutable shard bytes and expose checked tensor views. | [Loader API](../include/sglang/checkpoint.h#L22)<!-- prefix: "class Checkp" --> |
| `sglang_decoder` | Own device weights and run requests with `sglang_cuda` operators. | [Decoder API](../include/sglang/decoder.h#L147)<!-- prefix: "class Decode" --> · [Layer order](../include/sglang/decoder_internal.h#L273)<!-- prefix: "void execute" --> |

Checkpoint metadata and HTTP use `sglang_json`. Metadata parsing preserves historical underflow to signed zero and overflow to signed infinity, including in ignored fields. Required config values, index entries, tensor dimensions, and offsets still undergo validation. [Metadata parser](../src/checkpoint.cpp#L79)<!-- prefix: "json::Object" --> · [Runtime dependencies](text-serving.md#run-with-bundled-libraries) · [Link configuration](../xmake.lua#L98)<!-- prefix: "target(\"sgla" -->

#### Runtime path

The decoder validates tensor names, dtypes, and shapes before upload. Upload retains mappings and allocations until completion. Deferred cleanup retains shared weights. FP8 projections quantize BF16 inputs to E4M3 with one FP32 scale per 128 channels. [Weight upload](../src/runtime/model/decoder_weights.cpp#L140)<!-- prefix: "Decoder::Mod" --> · [Stream cleanup](../src/decoder_resources.cpp#L225)<!-- prefix: "CleanupError" --> · [Activation quantization](../src/text_ops.cu#L163)<!-- prefix: "__global__ v" -->

The supported checkpoint is the pinned Qwen3.8-27B-FP8 text model. The [shared model descriptor](../include/sglang/runtime/model/qwen_text_model.h#L17)<!-- prefix: "inline const" --> defines its sizes and four-layer full-attention interval.

Model construction reads CUDA properties, still rejects every device other than `sm_100`, and derives the effective policies. The exact `NVIDIA B200` name enables the B200 operator set. Other `sm_100` devices keep those operators disabled, so they are accepted only by this compatibility check and do not have a supported optimized path. [Device check](../src/runtime/model/decoder_weights.cpp#L145)<!-- prefix: "if (properti" --> · [Backend construction](../src/runtime/model/decoder_weights.cpp#L149)<!-- prefix: "backend = Cu" --> · [Backend policy](../include/sglang/runtime/backend/cuda_backend_policy.h#L12)<!-- prefix: "struct CudaB" -->

[DecoderExecutionPolicy](../include/sglang/runtime/execution/decoder_execution_policy.h#L5)<!-- prefix: "struct Decod" --> combines backend capabilities with configured limits. On B200, regular batch decode graphs require one text lane with MTP disabled. The backend's split endpoint governs when serial decode uses eager execution for split attention. [Graph policy](../src/runtime/execution/decoder_execution_policy.cpp#L19)<!-- prefix: "bool Decoder" --> · [Split policy](../src/runtime/execution/decoder_execution_policy.cpp#L33)<!-- prefix: "bool Decoder" --> · [Model field](../include/sglang/decoder_internal.h#L567)<!-- prefix: "decoder::Dec" -->

Batch text prefill without MTP and text label scoring use a 1,024-row cap on every accepted device. Media uses the 128-row cap. Text MTP prefill also uses 128 rows when ordered attention is disabled or prefix reuse is enabled. [Batch prefill cap](../src/runtime/execution/decoder_execution_policy.cpp#L24)<!-- prefix: "int DecoderE" --> · [Label scoring cap](../src/runtime/execution/decoder_execution_policy.cpp#L29)<!-- prefix: "int DecoderE" --> · [Serial prefill caps](text-serving.md#serial-prefill)

[KvResidencyPolicy](../include/sglang/runtime/cache/kv_residency_policy.h#L15)<!-- prefix: "class KvResi" --> is active only when a batch requests a resident-memory limit. Its capacity is the smaller of the requested limit and the available bytes after reserving up to 2 GiB. [Residency calculation](../src/runtime/cache/kv_residency_policy.cpp#L21)<!-- prefix: "void KvResid" --> · [Capacity measurement](../src/decoder_batch.cpp#L226)<!-- prefix: "if (options." -->

Serial K0 reuses its stream and grows storage on demand. Prefill rebuilds [MXFP8 pages](#p2-gqa-cache-contract) with an identity page table. Reset zeros BF16 convolution history and FP32 recurrent state across 48 GDN layers. [Generation loop](../src/decoder.cpp#L534)<!-- prefix: "void Decoder" --> · [State reset](../src/decoder_resources.cpp#L937)<!-- prefix: "void Request" -->

One model layer loop runs input RMSNorm, GQA or GDN, post-attention RMSNorm, then the MLP. The loop assigns separate KV and GDN state indexes. Tiled prefill and row-based eager, graph, and MTP verification passes supply their own buffers and operator dispatch. Embedding, draft seeding, and scoring stay with their callers. [Tile adapter](../src/decoder_prefill.cpp#L903)<!-- prefix: "execute_laye" --> · [Row adapter](../src/decoder_forward.cpp#L897)<!-- prefix: "void Decoder" -->

Prefill follows token order. The final prompt row gives initial logits. [Serial tiles](text-serving.md#serial-prefill) can exceed 128 rows. Serial K0 defaults to graph decode. Overlapping calls use private eager storage. [Prefill dispatch](../src/decoder_prefill.cpp#L849)<!-- prefix: "void Decoder" --> · [Decode graph](../src/decoder_forward.cpp#L858)<!-- prefix: "void Decoder" -->

The runtime selects [Tensor Core projections](fp8-prefill-projections.md#tensor-core-projections) for eligible B200 text. Exact SIMT projections serve label scoring and fallback cases. The two paths can produce different logits and greedy IDs.

See [FP8 prefill projections](fp8-prefill-projections.md) for SM100 eligibility, tile selection, and fallback.

Multirow K0 tensor-core prefill keeps these operators during projection fallback.

- [GDN prefill](gdn-prefill.md) prepares convolution, Q/K, and gates before ordered or chunked recurrence. [Preparation dispatch](../src/text_attention_gdn.cu#L660)<!-- prefix: "const bool g" -->
- [Flash or ordered GQA](flash-prefill.md#runtime-path) follows KV append. [Call](../src/decoder_prefill.cpp#L362)<!-- prefix: "const auto s" -->
- [Fused residual RMSNorm](fp8-prefill-projections.md#fused-activation-quantization) and SwiGLU round to BF16 before FP8 quantization. [RMSNorm](../src/text_normalization.cu#L376)<!-- prefix: "cudaError_t " --> · [SwiGLU](../src/text_ops.cu#L674)<!-- prefix: "cudaError_t " -->

Other target prefill uses direct attention. Eligible B200 target decode uses the [grouped split route](singleton-gqa-decode.md#split-kv-target-attention). Eager decode and serial graphs can use exact attention when their workspace fits. Serial decode falls back to direct attention. [Row dispatch](../src/decoder_forward.cpp#L413)<!-- prefix: "cudaError_t " --> · [Draft attention](native-model-info.md#draft-attention)

The lane stream orders append, attention, and gating. Logits storage holds scores for 5,120 positions. See the [batch attention routes](batch-graphs.md#constraints-and-failure-behavior) for score-backed singleton decode and multi-lane score allocation. [Workspace](../src/decoder_resources.cpp#L826)<!-- prefix: "ExactGqaWork" -->

Tuned FP8 projections require one target decode row in the forward group. This includes serial graphs, single-lane batch graphs, lone ordinary eager decode, and single-row verification. GQA combines Q/gate, K, and V. The MLP fusion rounds gate and up to BF16 before SwiGLU. GDN QKV and Z stay separate. [Lane selection](../src/decoder_forward.cpp#L72)<!-- prefix: "bool single_" --> · [Batch packing](../src/decoder_batch.cpp#L582)<!-- prefix: "if (rows[slo" -->

Tuned GEMV requires 16-byte-aligned FP8 weights and activations and input widths of 5,120, 6,144, or 17,408. Both fusions require the same alignment and width 5,120. Q/gate, K, and V require 12,288, 1,024, and 1,024 output rows. Gate and up require 17,408 each. Other valid inputs use separate operations. Output spans must be disjoint from inputs and each other. Fused SwiGLU permits identical gate and up spans. [Tuned GEMV](../src/text_decode_fp8.cu#L276)<!-- prefix: "cudaError_t " --> · [Fused QKV](../src/text_decode_fp8.cu#L298)<!-- prefix: "cudaError_t " --> · [Fused SwiGLU](../src/text_decode_fp8.cu#L321)<!-- prefix: "cudaError_t " --> · [Buffers](../include/sglang/decoder_cuda.cuh#L335)<!-- prefix: "// Read-only" -->

Row-wise prefill and MTP draft use GEMV. Legacy FP8 projections balance groups of up to five rows. Packed GEMM uses two to five rows. Packed GEMM and wide GEMV require 512-divisible input widths and 16-byte-aligned FP8 payloads. [Fused packed K0](batch-graphs.md#packed-paged-gqa-decode) uses bucketed projections. [Legacy grouping](../src/decoder_forward.cpp#L101)<!-- prefix: "int projecti" --> · [GEMV](../src/text_ops.cu#L691)<!-- prefix: "cudaError_t " --> · [GEMM](../src/text_ops.cu#L709)<!-- prefix: "cudaError_t " -->

Five eligible rows use double-buffered FP8 staging for MLP gate, up, and down. GDN QKV (5,120→10,240), GDN z (5,120→6,144), and GDN/GQA output (6,144→5,120) use the same kernel. Each warp computes four channels with ordered FP32 arithmetic and BF16 rounding. Other valid groups use packed GEMM or GEMV. [Shape check](../src/text_ops.cu#L725)<!-- prefix: "if (is_fp8_m" --> · [Staged kernel](../include/sglang/text/fp8_mlp_verify.cuh#L67)<!-- prefix: "__global__ v" -->

With exactly five target verification rows, each layer fuses RMSNorm with FP8 quantization and quantizes the five attention outputs together. The MLP projects gate and up separately, then fuses SwiGLU with quantization. Both fusions quantize BF16 results. [Row selection](../src/decoder_forward.cpp#L80)<!-- prefix: "bool fuse_ve" --> · [RMSNorm](../src/decoder_forward.cpp#L912)<!-- prefix: "check(verify" --> · [MLP](../src/decoder_forward.cpp#L668)<!-- prefix: "const bool f" -->

Single-row BF16 RMSNorm uses vector loads at width 5,120 with 16-byte-aligned branch, gains, and optional residual. Other calls use generic kernels. [RMSNorm dispatch](../src/text_normalization.cu#L406)<!-- prefix: "cudaError_t " -->

Outside MTP verification, one target decode row can use warp quantization and combined GDN gates. Quantization needs whole 128-channel groups, 8-byte-aligned BF16 input, and 4-byte-aligned E4M3 values. It preserves FP32 round-to-nearest scale and reciprocal divisions and saturated E4M3 conversion. Other valid layouts use the generic quantizer. [Dispatch](../src/decoder_forward.cpp#L113)<!-- prefix: "void quantiz" --> · [Quantizer](../src/text_decode_bandwidth.cu#L624)<!-- prefix: "cudaError_t " -->

Combined GDN gates need matching 48-by-5,120 shapes. After the initial score, serial eager and graph decode can pair two vocabulary rows per warp. The paired LM head shares input loads across each pair and uses streaming weight loads. It needs a positive even row count. Both BF16 kernels require width 5,120 and 16-byte-aligned weights and input. They preserve two ordered FP32 FMA chains, their sum, ordered warp reduction, and BF16 rounding. Other valid shapes or alignments use the existing GEMVs. [GDN gates](../src/text_decode_bandwidth.cu#L492)<!-- prefix: "cudaError_t " --> · [Serial LM head](../src/text_decode_bandwidth.cu#L540)<!-- prefix: "cudaError_t " -->

The [width-four batch route](batch-graphs.md#runtime-path) uses a paired BF16 LM head that shares weight reads across four inputs and preserves the same per-output accumulation and rounding. It requires a positive even vocabulary size, width 5,120, and 16-byte-aligned weights and inputs. Other shapes and alignments use the generic BF16 batch projection. [Grouped LM head](../src/text_decode_bandwidth.cu#L604)<!-- prefix: "cudaError_t " -->

#### Constraints and failure behavior

Pass one nonempty prompt with `--prompt-ids` or `--prompt-file`. Separate decimal IDs with commas or ASCII whitespace. Prompt and stop IDs must be in `[0, 248320)`. Prompt length plus budget must fit 262,144 tokens. The default budget is one. Zero budget returns no tokens and `length`. [CLI options](../apps/sglang-generate.cpp#L31)<!-- prefix: "void usage(s" --> · [Request validation](../src/decoder.cpp#L52)<!-- prefix: "void validat" -->

The CLI defaults to greedy selection, with ties going to the lowest ID. It uses the [K4 default](mtp.md#runtime-path). `--stop-token` emits a match and returns `stop`, even at the budget. Exhaustion returns `length`. There is no implicit EOS. Failures print to stderr and exit 1. See [sampling](text-serving.md#sampling) and [C++ and HTTP constraints](text-serving.md#finite-choice-constraints). [Greedy selection](../src/text_ops.cu#L427)<!-- prefix: "__device__ T" -->

The decoder retains CUDA resources and transfer buffers while completion is uncertain. Model and request setup retry cleanup on the same device. Persistent errors block setup. Terminal errors and ambiguous frees quarantine resources until exit. [Deferred cleanup](../src/decoder_resources.cpp#L108)<!-- prefix: "static void " --> · [Destructor diagnostics](../src/decoder_resources.cpp#L512)<!-- prefix: "Allocations:" -->

Private CUDA pools separate workspaces at or below 128 rows from larger ones.
One pool per device and class can set a release threshold. The thresholds
are 32 MiB and 1 GiB respectively. Other pools use CUDA defaults. See the
[active prefill workspace sizes](native-prefill-workspace.md#projection-pairs-for-mtp).
[Retention](../src/decoder_prefill.cpp#L48)<!-- prefix: "void configu" -->

Single rows and unavailable normal scratch use row-wise prefill. Release queues a stream-ordered free. After enqueue failure, recovery synchronizes the device, checks used bytes, and waits for any retried free. Explicit close reports the original error. [Generation](../src/decoder.cpp#L654)<!-- prefix: "reclaim_pref" -->, [batch admission](../src/decoder_batch.cpp#L419)<!-- prefix: "reclaim_pref" -->, and multirow allocation retry pending frees. [Allocation](../src/decoder_prefill.cpp#L736)<!-- prefix: "PrefillWorks" --> · [Release](../src/decoder_prefill.cpp#L791)<!-- prefix: "cudaError_t " -->

`--top-logits N` reports up to `N` raw scores per token, sorted by descending score then ascending ID. The range is 0–248,320 and defaults to 0. [Logit output](../apps/sglang-generate.cpp#L241)<!-- prefix: "void write_r" -->

#### Verification

Compare layer outputs and state on model shapes with frozen independent inputs and precision-aware tolerances. Require exact tensor names, shapes, dtypes, scales, and packed bytes. Run the [checkpoint metadata checks](../tests/decoder/README.md#checkpoint-metadata).

Use the [P2 verifier](#p2-gqa-verification) for MXFP8 encoding, causal attention, and page reuse. Check KV positions and ownership through append and decode. Check BF16 convolution history and FP32 recurrent state. Compare supported cache restores and continuations with fresh native execution under the same precision policy. [Independent GDN check](../tests/decoder/gdn_check.cu#L239)<!-- prefix: "void verify_" -->

Build the [host generator](../README.md#build-on-the-project-host). Set `SGLANG_GENERATE` to its path and `GPU_UUID` to the reserved B200 UUID. CUDA exposes it as ordinal 0. The default checkpoint is `/data0/shared/Qwen3.8-27B-FP8`. [Generator build](../tools/build-probe.sh#L76)<!-- prefix: "\"$xmake\" bui" -->

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    "${SGLANG_GENERATE:?set the printed generator path}" \
    --device 0 --prompt-ids 760,6511,314,9338,369 --max-new-tokens 8 --repeats 2 --mtp-steps 0
```

Check the prompt echo, eight in-range IDs, and `length`. Require identical IDs and finish reasons across runs. At budget one, set `--stop-token` to the first native ID. Require one ID and `stop`. At zero budget, require no IDs and `length`. Check reset by repeating the prompt after another request on the same decoder.

Check BF16 argmax and lowest-ID tie breaking. Compare logits with independent calculations. Use `--top-logits 248320` to check vocabulary shape and ranking on a short request. [Request ranking check](../tests/decoder/request_check.cpp#L171)<!-- prefix: "void check_r" -->

Historical diagnostics use the [FP8 prompt](../benchmarks/vllm/small-fp8.json#L2)<!-- prefix: "\"prompt_toke" -->, [vLLM IDs](../benchmarks/vllm/small-fp8-result.json#L20)<!-- prefix: "\"output_ids\"" -->, and frozen logits. See `artifacts/native-decoder-verification/final-evidence.sha256` and the [cross-engine verifier](../tests/decoder/run_native.py#L117)<!-- prefix: "def validate" -->.

#### Chunked prefill verification

Run the [prefill checks](../tests/prefill/README.md). Compare exact scoring and cache storage with their frozen oracles. Check 511-token prompts, chunk boundaries, mixed steps, reset, and slot reuse. Use the [Tensor Core verification criteria](performance.md#fp8-tensor-core-prefill) for eligible B200 text.

#### Native MTP

See [MTP](mtp.md) for exact tiled seeding, joint target verification, and per-lane rollback. [Verify MTP](mtp.md#verification) before timing.

## Native streaming server

See [native HTTP transport](http-server.md) for admission, overload, health, readiness, and shutdown. `--max-active` accepts 1–32 and defaults to 32. `--max-num-batched-tokens` accepts 1–32,768 and defaults to 8,192. The server uses the [K4 default](mtp.md#runtime-path) and supports [exact prefix reuse](prefix-cache.md). [Batch graphs](batch-graphs.md) are enabled by default for eligible steps. Use `--eager` to disable replay. [CLI limits](../apps/sglang-server.cpp#L134)<!-- prefix: "} else if (o" -->

The [single-device batch executor](../include/sglang/runtime/server/single_device_batch_executor.h#L12)<!-- prefix: "class Single" --> constructs `DecoderBatch` on its worker thread and propagates startup failures to the caller. `BatchService` records initial prefix-cache statistics and handles permanent processing failures. On shutdown, the service wakes the worker and joins it through the executor. [Service callbacks](../apps/server/batch_service.cpp#L143)<!-- prefix: "executor = s" --> · [Worker run loop](../src/runtime/server/single_device_batch_executor.cpp#L35)<!-- prefix: "void SingleD" --> · [Shutdown](../apps/server/batch_service.cpp#L157)<!-- prefix: "BatchService" -->

Lanes share weights and own KV, GDN state, and outputs. [Resident memory admission](batch-graphs.md#resident-memory-admission) applies at every capacity.

See [`POST /generate`](native-generate.md) for the native request and response format.

Follow the [benchmark wire contract](../benchmarks/streaming/README.md#wire-contract), [stream transport](http-server.md#completion-streams), and [decoder limits](#constraints-and-failure-behavior). A `stop_token_ids` match emits the ID with `stop`, including at the budget. Zero budget reports `length` without a model step. Prompt IDs appear in the first token event or the zero-budget terminal choice. Benchmark temperature accepts underflow to signed zero. [Request parser](../apps/server/completions.cpp#L28)<!-- prefix: "CompletionRe" --> · [Numeric retry](../apps/server/completions.cpp#L175)<!-- prefix: "json::Value " --> · [Terminal choice](../apps/server/completions.cpp#L113)<!-- prefix: "void finish_" -->

See [launch migration](native-model-info.md#migrate-old-launch-commands) for
earlier FP8, attention, and draft switches.

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

## Development workflow

Start with source and checkpoint identity, then an isolated Release build. Use focused operator and framework checks before end-to-end serving. Run the [GQA verification](#gqa-verification), [native decoder verification](#verification), [chunked prefill verification](#chunked-prefill-verification), [MTP verification](mtp.md#verification), and [streaming verification](#streaming-verification) in that order as they apply to a change. Keep raw evidence in ignored `artifacts/` and accepted procedures in maintained documentation.

Profile a slow path before changing it. Repeat paired measurements under the [performance contract](performance.md), preserving workload, device, checkpoint, build, and launch identity. Start W5 only after W1–W4 pass, and W6 only after W5 passes. Accept a workload only when its native median makespan is strictly lower than matched vLLM.
