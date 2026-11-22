# SM100 page-128 split prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Model | Try the page-128 launcher under the tensor-core prefill policy. | [Model dispatch](../include/sglang/text/prefill_attn_sm100.h#L46)<!-- prefix: "inline cudaE" --> |
| Batch planner | Select the tile row count. | [Batch tile cap](../src/decoder_batch.cpp#L701)<!-- prefix: "const bool w" --> |
| SM100 launcher | Check eligibility, select staging, and submit stream work. | [Launcher](../src/text/prefill_attn_sm100.cu#L914)<!-- prefix: "std::optiona" --> |

## Runtime path

Set `SGLANG_PREFILL_ATTN` with the model's `tensor-core` prefill policy to try the experimental SM100 launcher. When the launcher declines, including with the selector unset, the model uses [native Flash dispatch](flash-prefill.md#runtime-path). That dispatcher applies its own paired ordered TMEM guard and overrides.

| Setting | SM100 route | Batch planner |
| --- | --- | --- |
| `sm100` | Original page-128 split producer. | Can use 2,048 rows for B200 text prompts up to 32,768 tokens with no draft or active prefix-cache tile planner. |
| `sm100-tma` | Page-128 split producer with TMA staging on eligible tiles. | Uses the normal tile planner. |

TMA stages raw payloads and scales for conversion to BF16. Both producers use the same split selection, QK, online softmax, PV, FP32 merge, and protected repair. Both rescale when the softmax maximum increases and wait for PV completion before accumulator reuse. [TMA producer](../src/text/prefill_attn_sm100.cu#L329)<!-- prefix: "__device__ v" --> · [Conversion](../src/text/prefill_attn_sm100.cu#L349)<!-- prefix: "__device__ v" -->

For ordinary multisplit results, each half of the finish block handles one query head. Each thread rescales positive-mass splits to their largest score, accumulates in FP32 in split order, and writes two adjacent BF16 channels. A combined mass that fails `> 0` yields two positive zeros. The producer writes ordinary single-split results. Protected repair uses the whole finish block. [Merge](../src/text/prefill_attn_sm100.cu#L661)<!-- prefix: "__device__ v" --> · [Finish](../src/text/prefill_attn_sm100.cu#L697)<!-- prefix: "prefill_attn" --> · [Producer output](../src/text/prefill_attn_sm100.cu#L504)<!-- prefix: "__device__ v" -->

Both modes use the existing stream-ordered split scratch. Allocation, flag initialization, producer, finish, and release use the attention stream. Wide Flash does not need this split scratch. Graph replay retains the captured maps by value. Keep the cache, page table, queries, and output valid through replay. [Allocation](../src/text/prefill_attn_sm100.cu#L972)<!-- prefix: "auto status " --> · [Release](../src/text/prefill_attn_sm100.cu#L996)<!-- prefix: "const auto r" -->

## Constraints and failure behavior

The shared launcher accepts up to 8,192 query rows and 262,144 visible tokens. It requires at least 144 rows through 32,768 visible tokens. Align Q and output to 16 bytes and K and V payloads to 32 bytes. The device needs compute capability 10.0, CUDA memory pools, and sufficient opt-in shared memory. [Device checks](../src/text/prefill_attn_sm100.cu#L776)<!-- prefix: "cudaError_t " -->

TMA requires 1,024 query rows and a start position of at least 8,192 divisible by 128. It needs 16-byte scale alignment, representable signed 32-bit token coordinates, and 230,400 bytes of opt-in dynamic shared memory. [Tile checks](../src/text/prefill_attn_sm100.cu#L954)<!-- prefix: "if (mode == " --> · [TMA setup](../src/text/prefill_attn_sm100.cu#L842)<!-- prefix: "cudaError_t " -->

If the shared launcher accepts a request but TMA is ineligible, it selects the original split producer before scratch allocation. An unavailable encoder or an `INVALID_VALUE` or `NOT_SUPPORTED` result from tensor-map encoding can trigger this fallback. The same applies when setting the TMA kernel attribute returns invalid value, unsupported operation, invalid device function, or no kernel image. [Encoder status](../src/text/prefill_attn_sm100.cu#L822)<!-- prefix: "cudaError_t " --> · [Attribute status](../src/text/prefill_attn_sm100.cu#L899)<!-- prefix: "if (status =" -->

Validation returns `cudaErrorInvalidValue` for negative positions or row counts, position overflow, null buffers, and insufficient page counts. CUDA failures such as an invalid context, pending execution error, or unexpected setup error return to the caller. Allocation and submission errors also return to the caller without a split retry. [Pending error check](../src/text/prefill_attn_sm100.cu#L852)<!-- prefix: "if (const au" --> · [Dispatch](../src/text/prefill_attn_sm100.cu#L956)<!-- prefix: "if (const au" -->

Keep `sm100-tma` opt-in. Archived B200 replays showed bitwise parity and lower operator time against the page-128 split producer. A fixed-history checkpoint probe replaced wide Flash on eligible tiles and changed top tokens, including cases at 16,384 and 32,768 prompt tokens. Prompt length alone did not qualify the route for default use. Operator results leave checkpoint quality and end-to-end serving qualification open.

## Verification

In the default Pixi environment, run these commands from the repository root with a reserved B200.

```bash
xmake f -P . -m release -o build --mxfp8-gqa-check=y
xmake build -P . mxfp8-gqa-sm100-check
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    ./build/bin/mxfp8-gqa-sm100-check
```

Require exit zero and rising-max absolute error below 0.00002. For the page-128 graph regression, require a 576-thread TMA producer on the aligned tile at position 8,192. Require a 544-thread original split producer for `sm100` and for the misaligned-scale fallback under `sm100-tma`. Compare all BF16 outputs after poisoned-output replay. [Build target](../xmake.lua#L206)<!-- prefix: "target(\"mxfp" --> · [Rising-max assertion](../tests/prefill/mxfp8_gqa_sm100_check.cpp#L144)<!-- prefix: "require(maxi" -->

Use a separate fault probe for TMA setup rejection. Check split selection for a known descriptor or TMA attribute rejection. Check that an unexpected error or pending execution failure returns before allocation or launch. The alignment regression only exercises preflight eligibility.
