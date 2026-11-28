# Native development guide

Qwen3.8-27B-FP8 serves on one or two NVIDIA B200s. Defaults and acceptance depend on tensor-parallel width. [Checkpoint identity](research/checkpoint.md) · [TP2 serving](serving/tp2.md)

<a id="scope-at-adoption"></a>
## Current architecture and scope

The CUDA backend accepts `sm_100` devices. The exact `NVIDIA B200` name enables the B200 operator set. Keep tracked project headers declaration-only, including internal headers, `.ipp`, and `.tpp`. Put function definitions in compiled sources. A deleted function uses `= delete` on its first declaration. [Device resolution](../src/runtime/backend/cuda_backend_policy.cpp#L13)<!-- prefix: "CudaBackendP" --> · [Repository rules](../AGENTS.md#integrating-changes)

| Part | Responsibility | Source |
| --- | --- | --- |
| Model | `QwenTextModel` defines checkpoint identity, geometry, and layers. | [Model descriptor](../include/sglang/runtime/model/qwen_text_model.h#L7)<!-- prefix: "struct QwenT" --> |
| Backend | `CudaBackendPolicy` resolves device capabilities. The CUDA backend owns allocation, stream, graph, and deferred cleanup primitives. `DecoderExecutionPolicy` selects operators within request limits. | [Backend capabilities](../include/sglang/runtime/backend/cuda_backend_policy.h#L22)<!-- prefix: "struct CudaB" --> · [Execution policy](../include/sglang/runtime/execution/decoder_execution_policy.h#L7)<!-- prefix: "struct Decod" --> |
| Service | Startup supplies the executor to the scheduler. The scheduler owns admission and output publication. The executor owns the worker session and reports topology. | [Executor contract](../include/sglang/runtime/server/batch_executor.h#L9)<!-- prefix: "class BatchE" --> · [Session contract](../include/sglang/runtime/server/batch_session.h#L17)<!-- prefix: "class BatchS" --> · [Topology](../include/sglang/server/batch_service.h#L129)<!-- prefix: "BatchTopolog" --> |
| HTTP | The Qwen serving profile selects the tokenizer, templates, parser, and tool constraints. The scheduler's state, trace serialization, and output mailbox are private to HTTP. | [Serving profile](../apps/server/qwen/qwen_serving_profile.cpp#L134)<!-- prefix: "const Servin" --> · [Service state](../include/sglang/server/detail/batch_service_state.h#L88)<!-- prefix: "struct Batch" --> |
| Prefix cache | `KvResidencyPolicy` accounts for request reservations. `PrefixResidency` manages separate GQA pages and GDN checkpoints. LMCache stores pressure backups and restores them on demand. | [Prefix cache](prefix-cache.md) · [Request residency](../include/sglang/runtime/cache/kv_residency_policy.h#L15)<!-- prefix: "class KvResi" --> · [GPU pools](serving/request-memory.md) |
| Build | Use XMake, default Pixi, user CUDA 13.4 compatibility libraries, and the host driver. Use private output and cache directories for concurrent work. | [Host build](../README.md#build-on-the-project-host) · [Isolated XMake builds](serving/isolated-builds.md) |

Keep production CUDA operators in this project. Use CUDA and NVIDIA open-source building blocks such as CUTLASS. Reference engines may use their own operators, and tests may use PyTorch as a numerical reference. Record the license, version, and execution path of host dependencies. Keep explanations in the docs and record verified failures in [development lessons](lessons.md). [Repository rules](../AGENTS.md)

Production C/C++/CUDA grew from 67,153 physical lines at the baseline to 68,714 at the declaration-only cutover, then fell to 68,663 after prefill and dependency consolidation. Internal include directives changed from 704 to 757 to 743. Upward directives fell from 85 across 16 layer pairs to 74 across 14, then to 66 across the same 14 pairs. The last cut removed three backend-to-execution, three kernel-to-execution, and two cache-to-execution directives. Model and runtime server together went from 14 direct HTTP includes and 13 transitively reachable HTTP files to zero. [Baseline source](https://github.com/para1lel/sglang.cpp/tree/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1)

For this architecture comparison, count physical lines in `.c`, `.cc`, `.cpp`, `.cxx`, `.cu`, `.h`, `.hh`, `.hpp`, `.cuh`, `.inl`, and `.ipp` files under `apps/`, `src/`, and `include/sglang/`, excluding `include/sglang/testing/`. Count repeated and conditional literal includes. Resolve quoted paths beside the including file first, then resolve both forms under `include/`. Count targets among the measured files. A directive is upward when its source layer precedes its target. The layer order is utility, tokenizer, backend, media, kernels, cache, model, execution, runtime server, HTTP server, application.

Assign `runtime/` files to their named layer and `apps/server/` and `include/sglang/server/` to HTTP. Other `apps/` files are application. Under `src/` and `include/sglang/`, assign `json.*` and `utf8.*` to utility, `tokenizer/` to tokenizer, `media*`, `multimodal_prompt.*`, `vision*`, and `decoder_media.*` to media, and `checkpoint.*` to model. `text/`, `text_*`, `mxfp8_*`, `prefill_fp8*`, `decoder_cuda.*`, `decoder_sampling.*`, and `decoder_constraint_cuda.*` are kernels. The remaining `decoder*` files are execution.

The cutover and consolidation were measured on canonical Git trees `6a745945d81ecdd78102b5f0a6cd3bb9cf3a9fac` and `2e18e98c29ab0cc35c5de645fc625da9bc834caf`, respectively. The consolidated production trees are `6e4b41639ba24254a426d76103f92863592c0f40` for `apps/`, `37929c2c714603ab3694f100443eb72683241653` for `src/`, and `9d8f3e3b2498176820bbae8289a71cd88a7d3027` for `include/sglang/`.

## Framework code size

All maintained framework code must fit within **30,000 physical lines**. Count production code, headers, tests, benchmarks, tools, hooks, build code, and configurations that define executable steps. Include maintained generated and vendored code and any maintained program embedded in a document or fixture. Count blank lines, comments, and a final line without a newline. Exclude prose, licenses, fixtures consumed only as data, scalar tool settings, and unmodified dependencies outside the tree. Follow the coding standards and preserve supported behavior, numerical semantics, resource ownership, failure behavior, and AgentX performance. [Implementation standards](../.agents/skills/coding-mode/references/code-standards.md#maintained-size-limits) · [AgentX performance standard](performance.md#agentx-performance-standard)

The fixed source baseline contains **133,369 physical lines across 547 code files**. Meeting the cap requires a net reduction of at least **103,369 lines** from that baseline. [Baseline source](https://github.com/para1lel/sglang.cpp/tree/adf795a5966574e4f38a8b8abe0babe5ac5afd26)

| Baseline paths | Physical lines |
| --- | ---: |
| `src/` | 43,886 |
| `include/` | 10,968 |
| `apps/` | 17,393 |
| `tests/` | 46,023 |
| `benchmarks/` | 9,186 |
| `tools/` | 2,391 |
| `.agents/` | 2,866 |
| `.githooks/` | 37 |
| `xmake.lua` | 619 |
| Total | 133,369 |

Reproduce the baseline from its Git tree. Audit each candidate for added or relocated code, including new languages, extensionless programs, executable configuration, and code embedded in documents or fixture files. Report the candidate inventory against its exact commit or tree. When a dependency takes over a responsibility, record its pin, license, purpose, local integration or adaptation, and execution or deployment cost. [Execution identity](../.agents/skills/coding-mode/references/execution-identity.md)

```sh
revision=adf795a5966574e4f38a8b8abe0babe5ac5afd26
git grep -a -c -e '^' "$revision" -- \
  '*.cpp' '*.cu' '*.cuh' '*.h' '*.py' '*.sh' '*.lua' .githooks .agents/gates.json |
  awk -F: '{ files++; lines += $NF } END { print files, lines }'
```

<a id="p2-gqa-cache-contract"></a>
## CUDA operator contracts

K and V use four 256-channel E4M3 heads and one E8M0 scale per 32 contiguous channels. E4M3 weights use BF16 inverse scales per 128×128 block. [Checkpoint investigation](research/checkpoint.md)

Each 128-token page holds 131,072 payload bytes and 4,096 scale bytes per tensor. The device table maps logical pages to distinct physical pages. [Cache layout](../include/sglang/mxfp8_gqa.h#L28)<!-- prefix: "struct Mxfp8" --> · [Page table](../include/sglang/mxfp8_gqa.h#L37)<!-- prefix: "struct Mxfp8" -->

`gqa_geometry_valid` and `gqa_geometry_matches` are runtime predicates. For compile-time checks, use the fields of a `constexpr GqaGeometry` directly. Check geometry against the live cache with `gqa_geometry_matches` at runtime. [Geometry constants](../include/sglang/mxfp8_gqa.h#L23)<!-- prefix: "inline const" --> · [Runtime predicates](../src/text/gqa/geometry.cpp#L9)<!-- prefix: "bool gqa_geo" -->

```cpp
constexpr auto geometry = sglang::gqa_rank_geometry;
static_assert(geometry.kv_heads == sglang::mxfp8_kv_heads / 2);
static_assert(geometry.query_heads == sglang::mxfp8_queries_per_kv_head * geometry.kv_heads);
```

Link `sglang_cuda`. Apply RoPE to Q and K. Append contiguous BF16 K and V `[tokens, 4, 256]`. Pass the ready prefix length and first query position to `mxfp8_gqa_attention`. BF16 Q and output are `[queries, 24, 256]`. Row `r` sees through `first_query_position + r` inclusive. Check launch status. Retain storage through stream completion. [Library](../xmake.lua#L136)<!-- prefix: "target(\"sgla" --> · [Append](../include/sglang/mxfp8_gqa.h#L45)<!-- prefix: "cudaError_t " --> · [Attention](../include/sglang/mxfp8_gqa.h#L50)<!-- prefix: "cudaError_t " -->

For group maximum magnitude `amax`, clamp `ceil(log2(max(amax, 1e-30) / 448)) + 127` to `[0, 254]`. See the [scratch-free reader](singleton-gqa-decode.md#scratch-free-ordered-attention) for direct attention. [Scale calculation](../src/text/gqa/kv_append.cu#L14)<!-- prefix: "__device__ i" -->

Exact scoring reduces eight groups in FP32, sums in group order, and scales by `1/16`. Ordered replay uses token-ordered FMAs and divides in FP32 before BF16 rounding. [Score](../src/mxfp8_gqa.cu#L51)<!-- prefix: "__device__ _" --> · [Replay](../src/mxfp8_gqa.cu#L229)<!-- prefix: "__device__ v" -->

See [singleton and grouped split decode](singleton-gqa-decode.md) for current routes, and [archived packed decode](long-paged-decode.md#fused-packed-gqa) for the former K0 implementation.

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
regular Flash, split graph nodes, and ordered replay. The [verifier](../tests/mxfp8_gqa_check.cu#L598)<!-- prefix: "void run_che" --> covers append, attention, and page reuse with packed KV and BF16.

The [ordered decode check](../tests/mxfp8_gqa_check.cu#L501)<!-- prefix: "void check_e" --> covers BF16 parity, tile and page boundaries, graph and eager execution, input preservation, and validation.

Check [exponents 0 and 255](../tests/mxfp8_gqa_prefill_check.cu#L639)<!-- prefix: "void check_o" -->, [remaps and graphs](../tests/mxfp8_gqa_check.cu#L811)<!-- prefix: "void check_o" -->, [chunk edges and tails](../tests/mxfp8_gqa_prefill_check.cu#L591)<!-- prefix: "void check_o" -->, [rounding](../tests/mxfp8_gqa_prefill_check.cu#L639)<!-- prefix: "void check_o" -->, and [masked-score recovery](../tests/mxfp8_gqa_check.cu#L874)<!-- prefix: "void check_o" -->.

Use the [ordered prefill replay checks](ordered-gqa-prefill.md#verification) for staged tails and overflow controls.

## Native text decoder

The [checkpoint loader](../include/sglang/checkpoint.h#L24)<!-- prefix: "class Checkp" --> validates text config before opening the tensor index and shards. Use `Checkpoint::text_model()` for validated model geometry. Metadata parsing preserves underflow to signed zero and overflow to signed infinity, including in ignored fields. Required values, shapes, and offsets are validated independently. The decoder validates tensor names, shapes, and dtypes before upload and keeps weights and mappings live through asynchronous use. [Metadata parser](../src/checkpoint.cpp#L80)<!-- prefix: "json::Object" --> · [Weight upload](../src/runtime/model/decoder_weights.cpp#L107)<!-- prefix: "Decoder::Mod" -->

### Runtime path

The shared layer loop uses the model layout to select each layer's KV or GDN bank. Tile and row adapters supply buffers and operator dispatch. The outer paths own embedding, draft seeding, and scoring. [Layer mapping](../src/runtime/model/qwen_text_model.cpp#L13)<!-- prefix: "QwenTextMode" --> · [Execution bank selection](../src/decoder_prefill.cpp#L29)<!-- prefix: "const auto r" --> · [Tile adapter](../src/decoder_prefill.cpp#L416)<!-- prefix: "model.backen" --> · [Row adapter](../src/decoder_forward.cpp#L77)<!-- prefix: "void Decoder" -->

Serial K0 reuses its stream and grows storage on demand. Prefill rebuilds [MXFP8 pages](#p2-gqa-cache-contract) with an identity page table. Reset clears BF16 convolution history and FP32 recurrent state across the GDN layers. The final prompt row supplies the initial logits. Overlapping calls use private eager storage. [Generation loop](../src/decoder.cpp#L741)<!-- prefix: "void Decoder" --> · [State reset](../src/runtime/model/qwen_request.cpp#L421)<!-- prefix: "void Request" -->

Use the feature contracts for dispatch and numerical details.

| Path | Contract |
| --- | --- |
| Text prefill | [Serial tiles](text-serving.md#serial-prefill), [batch tiles and workspace](native-prefill-workspace.md), [FP8 projections and activation rounding](fp8-prefill-projections.md), [GDN recurrence](gdn-prefill.md), and [GQA dispatch](flash-prefill.md#runtime-path) |
| Decode | [Singleton GQA and exact/direct fallback](singleton-gqa-decode.md), [batch and graph selection](batch-graphs.md#runtime-path), and [GDN state updates](gdn-decode.md) |
| Speculation and reuse | [MTP seeding, verification, and rollback](mtp.md#runtime-path) and [prefix restoration](prefix-cache.md#checkpoint-compatibility-and-resume) |
| Media and scoring | [Multimodal decoder](multimodal-decoder.md) and [pointwise scoring](serving/pointwise-score.md) |

### Constraints and failure behavior

Use the [generator contract](native-generate.md) for token validation, budgets, sampling, stops, and exit status. The checkpoint has 248,320 vocabulary IDs and a 262,144-token context.

The decoder retains CUDA resources and transfer buffers while completion is uncertain. Model and request setup retry cleanup on the same device. Persistent errors block setup. Terminal errors and ambiguous frees quarantine resources until exit. [Deferred cleanup](../src/runtime/backend/cuda_resources.cpp#L117)<!-- prefix: "static void " --> · [Destructor diagnostics](../src/runtime/backend/cuda_resources.cpp#L461)<!-- prefix: "Allocations:" -->

Workspace sizing, row-wise fallback, and pool retention are documented with the prefill allocation contract. [Prefill workspace](native-prefill-workspace.md#constraints-and-failure-behavior)

Use `--top-logits N` for raw scores sorted by descending score then ascending ID. [Logit output](../apps/sglang-generate.cpp#L286)<!-- prefix: "void write_r" -->

### Verification

Compare layer outputs and state on model shapes with frozen independent inputs and precision-aware tolerances. Require exact tensor names, shapes, dtypes, scales, and packed bytes. Run the [checkpoint metadata checks](../tests/decoder/README.md#checkpoint-metadata).

Use the [operator qualification procedure](performance.md#operator-qualification) when changing arithmetic or dispatch.

Use the [P2 verifier](#p2-gqa-verification) for MXFP8 encoding, causal attention, and page reuse. Check KV positions and ownership through append and decode. Check BF16 convolution history and FP32 recurrent state. Compare supported cache restores and continuations with fresh native execution under the same precision policy. [Independent GDN check](../tests/decoder/gdn_check.cu#L239)<!-- prefix: "void verify_" -->

Build the [host generator](../README.md#build-on-the-project-host). Set `SGLANG_GENERATE` to its path and `GPU_UUID` to the reserved B200 UUID. CUDA exposes it as ordinal 0. The default checkpoint is `/data0/shared/Qwen3.8-27B-FP8`. [Generator build](../tools/build-probe.sh#L119)<!-- prefix: "\"$xmake\" bui" -->

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    "${SGLANG_GENERATE:?set the printed generator path}" \
    --device 0 --prompt-ids 760,6511,314,9338,369 --max-new-tokens 8 --repeats 2 --mtp-steps 0
```

Check the prompt echo, eight in-range IDs, and `length`. Require identical IDs and finish reasons across runs. At budget one, set `--stop-token` to the first native ID. Require one ID and `stop`. At zero budget, require no IDs and `length`. Check reset by repeating the prompt after another request on the same decoder.

Check BF16 argmax and lowest-ID tie breaking. Compare logits with independent calculations. Use `--top-logits 248320` to check vocabulary shape and ranking on a short request. [Request ranking check](../tests/decoder/request_check.cpp#L166)<!-- prefix: "void check_r" -->

For cross-engine diagnostics, compare frozen inputs and logits on identical token histories. Treat ID drift under different KV or projection precision as diagnostic. [Cross-engine verifier](../tests/decoder/run_native.py#L117)<!-- prefix: "def validate" --> · [Precision controls](performance.md#gqa-cache-comparison)

### Chunked prefill verification

Run the [prefill checks](../tests/prefill/README.md). Compare exact scoring and cache storage with their frozen oracles. Check 511-token prompts, chunk boundaries, mixed steps, reset, and slot reuse. Use the [Tensor Core verification criteria](performance.md#fp8-tensor-core-prefill) for eligible B200 text.

## Native streaming server

Use `--max-active` for the 1–32 HTTP lane limit and `--max-num-batched-tokens` for the 1–32,768 aggregate step budget. `--gpu-memory-utilization` sets the arena’s startup HBM fraction and defaults to `0.88`. `--prefix-gqa-pages` and `--prefix-gdn-slots` default to automatic sizing. Verify caps at `/health` and resolved settings at `/server_info`. [CLI limits](../apps/sglang-server.cpp#L207)<!-- prefix: "} else if (o" --> · [HTTP transport](http-server.md) · [Server information](native-model-info.md#server-information)

Startup injects the Qwen serving profile and a batch executor into `BatchService`. The worker owns the batch and topology. Startup waits for the first cache snapshot. Shutdown closes the queue and wakes the worker before joining. [Composition](../apps/sglang-server.cpp#L412)<!-- prefix: "auto executo" --> · [Executor lifecycle](../src/runtime/server/single_device_batch_executor.cpp#L84)<!-- prefix: "void SingleD" --> · [Startup handshake](../apps/server/batch_service.cpp#L41)<!-- prefix: "void BatchSe" -->

The scheduler uses the session for admission, cancellation, and whole batch steps. The [batch planner](batch-graphs.md#runtime-path) allocates rows. Operator dispatch runs within each planned step. Lanes share device weights and own request KV, GDN state, and output. The [resident-memory limit](batch-graphs.md#resident-memory-admission) applies separately from row and lane limits. [Scheduler loop](../apps/server/batch_service.cpp#L295)<!-- prefix: "void BatchSe" -->

Call `BatchSession` operations on the worker thread that owns the session. `admit()` returns the number of prompt positions restored. After an admission exception, the scheduler checks `available()` to see whether the session can continue. If it can, the scheduler ends the affected request. Otherwise, the scheduler rethrows the exception. [Worker session](../src/runtime/server/single_device_batch_executor.cpp#L84)<!-- prefix: "void SingleD" --> · [Admission and failure](../apps/server/detail/batch_service_cache.cpp#L94)<!-- prefix: "bool BatchSe" -->

Use the serving contracts for [native generation](native-generate.md), [text](text-serving.md), [chat](chat-serving.md), [benchmark token-ID streams](text-serving.md#token-id-streaming-alias), and [launch migration](native-model-info.md#migrate-old-launch-commands).

### Streaming verification

Run the [CPU transport checks](http-server.md#verification) and [admission checks](batch-graphs.md#verification). Use the locked Pixi environment, a Release server, the pinned checkpoint, `nsys` on `PATH`, the reserved B200 UUID, and a fresh evidence directory.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    python -B tests/batch/run_native.py --binary build/bin/sglang-server \
    --checkpoint /data0/shared/Qwen3.8-27B-FP8 --evidence artifacts/native-batch-check
```

Require `"result": "passed"` in `status.json`. Check widths one, two, and four, eligible singleton graph launches, mixed stops, KV transitions, queueing, reset, and zero retired request bytes. Keep `identity.json` and raw evidence. [Live batch regression](../tests/batch/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in li" -->

After the runner exits, start a separate server on the same reserved B200 using the same Release binary and checkpoint. Use a free loopback port and `--max-active 1` to exercise single-slot queueing and disconnect recovery.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    build/bin/sglang-server --checkpoint /data0/shared/Qwen3.8-27B-FP8 \
    --mtp-steps 0 --port 17384 --max-active 1
```

Once the server is listening, run `python -B tests/test_native_streaming.py --port 17384` from another shell at the repository root, using the same port. Check zero budget, half-close, zero temperature forms, strict JSON, queueing, and disconnect recovery. Stop the dedicated server when the smoke ends. [Streaming smoke](../tests/test_native_streaming.py#L1)<!-- prefix: "\"\"\"Loopback " -->

Use the batch runbook for graph and eager parity at widths one through four and for wide K0 serving at capacities 16 and 32. [Batch checks](../tests/batch/README.md) · [Wide K0 serving](../tests/batch/README.md#wide-k0-serving)

## Runtime and state ownership

Share one execution path across supported features and batch sizes. Specialize by shape and resource limits at the operator boundary. Keep fixture IDs and scenario names out of dispatch.

Verify [prefix checkpoints](prefix-cache.md#verification), [MTP state](mtp.md#verification), and stream completion before storage reuse.

## Development workflow

Start with source and checkpoint identity and an [isolated Release build](serving/isolated-builds.md). Run the applicable [GQA](#gqa-verification), [decoder](#verification), [chunked prefill](#chunked-prefill-verification), [MTP](mtp.md#verification), and [streaming checks](#streaming-verification) before measuring end to end. Keep raw evidence in ignored `artifacts/` and accepted procedures in maintained documentation.

Profile separately from formal measurement. Match the pinned vLLM dispatch by shape, visible length, precision, and fusion boundary for operator comparisons. Record reference source and revision. Validate changes at the operator and execution-step boundaries before comparing end-to-end performance. [AgentX performance standard](performance.md#agentx-performance-standard)

See the pinned decoder notes for additional operator details and diagnostic inputs. [Decoder notes](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/development.md#native-text-decoder)

### Check and format staged source

Use the [installed hooks in default Pixi](../AGENTS.md#integrating-changes). To reproduce pre-commit checks without committing, run this command from the repository root.

```bash
.githooks/pre-commit
```

The hook checks the selected index against `HEAD` and ignores unstaged and untracked files. Check or tool failures return a nonzero status. The hook leaves the index and working tree unchanged.

Clang-format covers C, C++, CUDA, and their headers. Ruff covers Python and stubs. Format checks use staged additions and replacements. New files, renamed files, and files whose type changes are checked in full.

Clang-tidy checks changed lines and new critical analyzer findings. CUDA analysis covers host and `sm_100` or `sm_100a` device code, with extra device-only passes for selected sources. A deleted header can trigger analysis of its surviving consumers. Parse and dependency errors also fail the check.

Ruff checks changed-line and newly introduced diagnostics, including annotation and type-only import rules. Syntax errors also fail the check. Test assertions and script output have scoped exceptions.

Selected XMake builds run offline. For a missing CUTLASS archive, set `SGLANG_CUTLASS_ARCHIVE` to a verified archive or run `bash tools/build.sh` to populate the cache.

Run the formatter command printed by the hook on working-tree content that matches the staged ranges. Stage the intended edits and repeat the check. Omit range options to format a full file.

```bash
clang-format --style=file:.clang-format -i ./path/to/file.cu
ruff format --config=ruff.toml --no-cache ./path/to/file.py
```

To replay the historical handoff proof, use a disposable clone and follow the original README's pinned Python and toolchain setup at this revision. Write the JSONL outside the checkout. [Original setup and proof](https://github.com/para1lel/sglang.cpp/blob/569818d31ee2d54b21dc41e8381e99b81762c97f/README.md#L42-L63)

```bash
git switch --detach 569818d31ee2d54b21dc41e8381e99b81762c97f
git config --local core.hooksPath .githooks
python -B tests/prove_handoff.py --revision 569818d31ee2d54b21dc41e8381e99b81762c97f > /outside/checkout/handoff.jsonl
```
