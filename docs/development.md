# Native development guide

The supported deployment uses one physical NVIDIA B200 and the Qwen3.8-27B-FP8 checkpoint. The checkpoint investigation records the model and tensor identities and the pinned SGLang sources. [Checkpoint investigation](research/checkpoint.md)

<a id="scope-at-adoption"></a>
## Current architecture and scope

The CUDA backend accepts `sm_100` devices. The exact `NVIDIA B200` name enables the B200 operator set. Project headers belong in `include/sglang/`. [Device resolution](../src/runtime/backend/cuda_backend_policy.cpp#L14)<!-- prefix: "CudaBackendP" --> · [Repository rules](../AGENTS.md)

| Part | Responsibility | Source |
| --- | --- | --- |
| Model | `QwenTextModel` defines checkpoint identity, geometry, and layer types. The Qwen serving profile owns tokenizer identity, chat templates, and parser selection. | [Model descriptor](../include/sglang/runtime/model/qwen_text_model.h#L7)<!-- prefix: "struct QwenT" --> · [Serving profile](../src/runtime/model/qwen_serving_profile.cpp#L134)<!-- prefix: "const Servin" --> |
| Backend | `CudaBackendPolicy` resolves device capabilities. The CUDA backend owns allocation, stream, graph, and deferred cleanup primitives. `DecoderExecutionPolicy` selects operators within request limits. | [Backend capabilities](../include/sglang/runtime/backend/cuda_backend_policy.h#L24)<!-- prefix: "struct CudaB" --> · [Execution policy](../include/sglang/runtime/execution/decoder_execution_policy.h#L7)<!-- prefix: "struct Decod" --> |
| Service | Startup supplies the executor to the scheduler. The scheduler owns admission and output publication. The executor owns the worker session and reports topology. | [Executor contract](../include/sglang/runtime/server/batch_executor.h#L9)<!-- prefix: "class BatchE" --> · [Session contract](../include/sglang/runtime/server/batch_session.h#L18)<!-- prefix: "class BatchS" --> · [Topology](../include/sglang/server/batch_service.h#L166)<!-- prefix: "BatchTopolog" --> |
| Prefix cache | `KvResidencyPolicy` sizes native GPU reservations. The cache client sends allocation views and prefix identities. LMCache owns data movement and retention. | [Prefix cache](prefix-cache.md) · [Request residency](../include/sglang/runtime/cache/kv_residency_policy.h#L15)<!-- prefix: "class KvResi" --> |
| Build | Use XMake, default Pixi, user CUDA 13.4 compatibility libraries, and the host driver. Use private output and cache directories for concurrent work. | [Host build](../README.md#build-on-the-project-host) · [Isolated XMake builds](serving/isolated-builds.md) |

Implement production CUDA operators in this project. Implementations may use CUTLASS primitives and templates or components from the CUDA Toolkit. Keep ready-made external operators such as FlashInfer FMHA outside the native execution path. The native runtime excludes cuBLAS, cuBLASLt, and cuDNN. Reference engines may use their own operators, and tests may use PyTorch as a numerical reference. Record the justification, license, version, and execution path of HTTP, JSON, tokenizer, media, and system dependencies. Record verified failures and preventive checks in [development lessons](lessons.md). Project code files carry no comments. The pre-commit hook inspects the entire staged content of every added or changed code file, including comments already present. Shebangs, compiler directives, and string literals are preserved. Untouched legacy files are not cleaned up. Explanations live in these documents, not in code.

## Current matched K4 status

Use a verified Release executable and run one row at a time on the same reserved B200. The benchmark agreement sets the acceptance gate and records the archived evidence. Use the driver recipe for launch and evidence capture. [Benchmark agreement](performance.md#matched-k4-evidence) · [Matched K4 driver](../benchmarks/paired/README.md#matched-k4-timing-protocol)

<a id="p2-gqa-cache-contract"></a>
## CUDA operator contracts

K and V use four 256-channel E4M3 heads and one E8M0 scale per 32 contiguous channels. E4M3 weights use BF16 inverse scales per 128×128 block. [Checkpoint investigation](research/checkpoint.md)

Each 128-token page holds 131,072 payload bytes and 4,096 scale bytes per tensor. The device table maps logical pages to distinct physical pages. [Cache layout](../include/sglang/mxfp8_gqa.h#L31)<!-- prefix: "struct Mxfp8" --> · [Page table](../include/sglang/mxfp8_gqa.h#L40)<!-- prefix: "struct Mxfp8" -->

Link `sglang_cuda`. Apply RoPE to Q and K. Append contiguous BF16 K and V `[tokens, 4, 256]`. Pass the ready prefix length and first query position to `mxfp8_gqa_attention`. BF16 Q and output are `[queries, 24, 256]`. Row `r` sees through `first_query_position + r` inclusive. Check launch status. Retain storage through stream completion. [Library](../xmake.lua#L65)<!-- prefix: "target(\"sgla" --> · [Append](../include/sglang/mxfp8_gqa.h#L53)<!-- prefix: "cudaError_t " --> · [Attention](../include/sglang/mxfp8_gqa.h#L58)<!-- prefix: "cudaError_t " -->

For group maximum magnitude `amax`, clamp `ceil(log2(max(amax, 1e-30) / 448)) + 127` to `[0, 254]`. See the [scratch-free reader](singleton-gqa-decode.md#scratch-free-ordered-attention) for direct attention. [Scale calculation](../src/mxfp8_gqa.cu#L42)<!-- prefix: "__device__ i" -->

Exact scoring reduces eight groups in FP32, sums in group order, and scales by `1/16`. Ordered replay uses token-ordered FMAs and divides in FP32 before BF16 rounding. [Score](../src/mxfp8_gqa.cu#L227)<!-- prefix: "__device__ f" --> · [Replay](../include/sglang/text/gqa/ordered_gqa_vector.cuh#L153)<!-- prefix: "__device__ _" -->

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

The [checkpoint loader](../include/sglang/checkpoint.h#L24)<!-- prefix: "class Checkp" --> validates text config before opening the tensor index and shards. Use `Checkpoint::text_model()` for validated model geometry. Metadata parsing preserves underflow to signed zero and overflow to signed infinity, including in ignored fields. Required values, shapes, and offsets are validated independently. The decoder validates tensor names, shapes, and dtypes before upload and keeps weights and mappings live through asynchronous use. [Metadata parser](../src/checkpoint.cpp#L80)<!-- prefix: "json::Object" --> · [Weight upload](../src/runtime/model/decoder_weights.cpp#L107)<!-- prefix: "Decoder::Mod" -->

### Runtime path

The shared layer loop uses the model layout to select each layer's KV or GDN bank. Tile and row adapters supply buffers and operator dispatch. The outer paths own embedding, draft seeding, and scoring. [Layer mapping](../include/sglang/runtime/model/qwen_text_model.h#L40)<!-- prefix: "constexpr La" --> · [Execution bank selection](../include/sglang/runtime/model/qwen_layers.h#L96)<!-- prefix: "const auto [" --> · [Tile adapter](../src/decoder_prefill.cpp#L640)<!-- prefix: "execute_laye" --> · [Row adapter](../src/decoder_forward.cpp#L83)<!-- prefix: "void Decoder" -->

Serial K0 reuses its stream and grows storage on demand. Prefill rebuilds [MXFP8 pages](#p2-gqa-cache-contract) with an identity page table. Reset clears BF16 convolution history and FP32 recurrent state across the GDN layers. The final prompt row supplies the initial logits. Overlapping calls use private eager storage. [Generation loop](../src/decoder.cpp#L684)<!-- prefix: "void Decoder" --> · [State reset](../src/runtime/model/qwen_request.cpp#L316)<!-- prefix: "void Request" -->

Use the feature contracts for dispatch and numerical details.

| Path | Contract |
| --- | --- |
| Text prefill | [Serial tiles](text-serving.md#serial-prefill), [batch tiles and workspace](native-prefill-workspace.md), [FP8 projections and activation rounding](fp8-prefill-projections.md), [GDN recurrence](gdn-prefill.md), and [GQA dispatch](flash-prefill.md#runtime-path) |
| Decode | [Singleton GQA and exact/direct fallback](singleton-gqa-decode.md), [batch and graph selection](batch-graphs.md#runtime-path), and [GDN state updates](gdn-decode.md) |
| Speculation and reuse | [MTP seeding, verification, and rollback](mtp.md#runtime-path), [periodic and request-end checkpoints](prefix-cache.md#checkpoint-boundaries), and [prefix restoration](prefix-cache.md#checkpoint-compatibility-and-resume) |
| Media and scoring | [Multimodal decoder](multimodal-decoder.md) and [pointwise scoring](serving/pointwise-score.md) |

### Constraints and failure behavior

Pass one nonempty prompt with `--prompt-ids` or `--prompt-file`. Separate decimal IDs with commas or ASCII whitespace. Prompt and stop IDs must be in `[0, 248320)`. Prompt length plus budget must fit 262,144 tokens. The default budget is one. Zero budget returns no tokens and `length`. [CLI options](../apps/sglang-generate.cpp#L57)<!-- prefix: "void usage(s" --> · [Request validation](../src/decoder.cpp#L62)<!-- prefix: "void validat" -->

The CLI defaults to greedy selection, with ties going to the lowest ID. It uses the [K4 default](mtp.md#runtime-path). `--stop-token` emits a match and returns `stop`, even at the budget. Exhaustion returns `length`. There is no implicit EOS. Failures print to stderr and exit 1. See [sampling](text-serving.md#sampling) and [C++ and HTTP constraints](text-serving.md#finite-choice-constraints). [Greedy selection](../src/text_ops.cu#L423)<!-- prefix: "__device__ T" -->

The decoder retains CUDA resources and transfer buffers while completion is uncertain. Model and request setup retry cleanup on the same device. Persistent errors block setup. Terminal errors and ambiguous frees quarantine resources until exit. [Deferred cleanup](../src/runtime/backend/cuda_resources.cpp#L130)<!-- prefix: "static void " --> · [Destructor diagnostics](../src/runtime/backend/cuda_resources.cpp#L476)<!-- prefix: "Allocations:" -->

Workspace sizing, row-wise fallback, and pool retention are documented with the prefill allocation contract. [Prefill workspace](native-prefill-workspace.md#constraints-and-failure-behavior)

`--top-logits N` reports up to `N` raw scores per token, sorted by descending score then ascending ID. The range is 0–248,320 and defaults to 0. [Logit output](../apps/sglang-generate.cpp#L273)<!-- prefix: "void write_r" -->

### Verification

Compare layer outputs and state on model shapes with frozen independent inputs and precision-aware tolerances. Require exact tensor names, shapes, dtypes, scales, and packed bytes. Run the [checkpoint metadata checks](../tests/decoder/README.md#checkpoint-metadata).

Use the [operator qualification procedure](performance.md#operator-qualification) when changing arithmetic or dispatch.

Use the [P2 verifier](#p2-gqa-verification) for MXFP8 encoding, causal attention, and page reuse. Check KV positions and ownership through append and decode. Check BF16 convolution history and FP32 recurrent state. Compare supported cache restores and continuations with fresh native execution under the same precision policy. [Independent GDN check](../tests/decoder/gdn_check.cu#L239)<!-- prefix: "void verify_" -->

Build the [host generator](../README.md#build-on-the-project-host). Set `SGLANG_GENERATE` to its path and `GPU_UUID` to the reserved B200 UUID. CUDA exposes it as ordinal 0. The default checkpoint is `/data0/shared/Qwen3.8-27B-FP8`. [Generator build](../tools/build-probe.sh#L110)<!-- prefix: "\"$xmake\" bui" -->

The optional vendor GDN prefill kernel is provisioned ahead of time. `tools/aot_build_vendor_gdn.py` compiles the pinned CuTe kernel into `libgdn_cute_bf16_8_24.a`. The build links that archive when it exists, unless `SGLANG_GDN_VENDOR_BUILD` is `0`. [AOT provisioning](../tools/aot_build_vendor_gdn.py#L1)<!-- prefix: "\"\"\"Reproduci" --> · [Build gate](../xmake.lua#L516)<!-- prefix: "if os.getenv" -->

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    "${SGLANG_GENERATE:?set the printed generator path}" \
    --device 0 --prompt-ids 760,6511,314,9338,369 --max-new-tokens 8 --repeats 2 --mtp-steps 0
```

Check the prompt echo, eight in-range IDs, and `length`. Require identical IDs and finish reasons across runs. At budget one, set `--stop-token` to the first native ID. Require one ID and `stop`. At zero budget, require no IDs and `length`. Check reset by repeating the prompt after another request on the same decoder.

Check BF16 argmax and lowest-ID tie breaking. Compare logits with independent calculations. Use `--top-logits 248320` to check vocabulary shape and ranking on a short request. [Request ranking check](../tests/decoder/request_check.cpp#L171)<!-- prefix: "void check_r" -->

For cross-engine diagnostics, compare frozen inputs and logits on identical token histories. Treat ID drift under different KV or projection precision as diagnostic. [Cross-engine verifier](../tests/decoder/run_native.py#L117)<!-- prefix: "def validate" --> · [Precision controls](performance.md#gqa-cache-comparison)

### Chunked prefill verification

Run the [prefill checks](../tests/prefill/README.md). Compare exact scoring and cache storage with their frozen oracles. Check 511-token prompts, chunk boundaries, mixed steps, reset, and slot reuse. Use the [Tensor Core verification criteria](performance.md#fp8-tensor-core-prefill) for eligible B200 text.

## Native streaming server

Use `--max-active` for the 1–32 HTTP lane limit and `--max-num-batched-tokens` for the 1–32,768 aggregate step budget. Verify the effective values at `/health` and the resolved graph and MTP settings at `/server_info`. See the transport guide for admission, overload, readiness, and shutdown. [CLI limits](../apps/sglang-server.cpp#L180)<!-- prefix: "} else if (o" --> · [HTTP transport](http-server.md) · [Server information](native-model-info.md#server-information)

Server startup selects the Qwen serving profile and passes a single-device executor to `BatchService`. The executor creates and destroys `DecoderBatch` on its worker thread. Startup waits for the first cache snapshot and propagates initialization failures. On shutdown the service marks the queue closed and wakes the worker before joining. [Composition](../apps/sglang-server.cpp#L356)<!-- prefix: "auto executo" --> · [Executor lifecycle](../src/runtime/server/single_device_batch_executor.cpp#L81)<!-- prefix: "void SingleD" --> · [Startup handshake](../apps/server/batch_service.cpp#L47)<!-- prefix: "void BatchSe" --> · [Shutdown](../apps/server/batch_service.cpp#L59)<!-- prefix: "BatchService" -->

The scheduler uses the session for admission, cancellation, and whole batch steps. The [batch planner](batch-graphs.md#runtime-path) allocates rows. Operator dispatch runs within each planned step. Lanes share device weights and own request KV, GDN state, and output. The [resident-memory limit](batch-graphs.md#resident-memory-admission) applies separately from row and lane limits. [Scheduler loop](../apps/server/batch_service.cpp#L333)<!-- prefix: "void BatchSe" -->

Use the serving contracts for [native generation](native-generate.md), [text](text-serving.md), [chat](chat-serving.md), [benchmark token-ID streams](../benchmarks/streaming/README.md#wire-contract), and [launch migration](native-model-info.md#migrate-old-launch-commands).

### Streaming verification

Run the [CPU transport checks](http-server.md#verification).

Run the [scheduler admission checks](batch-graphs.md#verification) on a reserved B200.

Build `sglang-server` in Release mode and use the locked Pixi environment with the pinned checkpoint and Nsight Systems (`nsys`) on `PATH`. Set `GPU_UUID` to the reserved B200 UUID. From the repository root, run with a fresh evidence directory and substitute the path to your Release binary if needed.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    python -B tests/batch/run_native.py --binary build/bin/sglang-server \
    --checkpoint /data0/shared/Qwen3.8-27B-FP8 --evidence artifacts/native-batch-check
```

Require `"result": "passed"` in `status.json`. The runner checks streamed output parity at widths one, two, and four, with one Nsight Systems graph launch per eligible width-one decode step. At widths two and four, it checks full decode width with mixed stops, KV page transitions, and queueing and reset recovery. Require remaining peers to stay readable, replacements to match their serial outputs, and retired request bytes to reach zero. Keep `identity.json` and the raw evidence. [Live batch regression](../tests/batch/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in li" -->

After the runner exits, start a separate server on the same reserved B200 using the same Release binary and checkpoint. Use a free loopback port and `--max-active 1` to exercise single-slot queueing and disconnect recovery.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    build/bin/sglang-server --checkpoint /data0/shared/Qwen3.8-27B-FP8 \
    --mtp-steps 0 --port 17384 --max-active 1
```

Once the server is listening, run `python -B tests/test_native_streaming.py --port 17384` from another shell at the repository root, using the same port. Check zero budget, half-close, zero temperature forms, strict JSON, queueing, and disconnect recovery. Stop the dedicated server when the smoke ends. [Streaming smoke](../tests/test_native_streaming.py#L1)<!-- prefix: "\"\"\"Loopback " -->

Use the batch runbook for graph and eager parity at widths one through four and for wide K0 serving at capacities 16 and 32. [Batch checks](../tests/batch/README.md) · [Wide K0 serving](../tests/batch/README.md#wide-k0-serving)

## Runtime and state ownership

Design one execution path for all supported feature combinations, including a batch of one. Put specialized behavior at the smallest applicable boundary, such as operator parameters or launch configuration. Consolidate existing duplicate end-to-end logic into the shared path. Base dispatch on computation shapes and hardware resource limits. Verify several valid shapes. Keep benchmark identity, fixed prompt contents, and scenario labels out of dispatch decisions.

Verify [prefix checkpoints](prefix-cache.md#verification), [MTP state](mtp.md#verification), and stream completion before storage reuse.

## Development workflow

Start with source and checkpoint identity and an [isolated Release build](serving/isolated-builds.md). Run the applicable [GQA](#gqa-verification), [decoder](#verification), [chunked prefill](#chunked-prefill-verification), [MTP](mtp.md#verification), and [streaming checks](#streaming-verification) before measuring end to end. Keep raw evidence in ignored `artifacts/` and accepted procedures in maintained documentation.

Profile slow paths in a separate diagnostic run. Compare native operators with those the pinned vLLM build actually runs at matching row counts, visible context, precision, and fusion boundaries. Rank the gaps by their impact on request completion. Inspect the closest public implementation, such as DeepGEMM, FlashQLA, or FlashInfer. Record its source and revision before adapting applicable techniques in native code. Validate the operator and the full execution step. Reprofile the same workload on the same device and checkpoint with the same launch settings to choose the next target. Use the [benchmark agreement](performance.md#measurement) for acceptance.

Recover the [previous guide](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/development.md) for the detailed decoder operator notes and historical diagnostic inputs.
