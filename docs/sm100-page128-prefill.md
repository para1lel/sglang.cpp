# Historical SM100 page-128 split prefill

This document describes the earlier page-128 selector study. See
[native Flash dispatch](flash-prefill.md#runtime-path) for current text prefill.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Model | Try the page-128 launcher under the tensor-core prefill policy. | [Model dispatch](../include/sglang/text/prefill_attn_sm100.h#L45)<!-- prefix: "inline cudaE" --> |
| Batch planner | Select the historical tile row count. | [Batch tile cap at the earlier revision](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/decoder_batch.cpp#L715)<!-- prefix: "const bool w" --> |
| SM100 launcher | Check eligibility, select staging, and submit stream work. | [Launcher](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L914)<!-- prefix: "std::optiona" --> |

## Runtime path

The study set `SGLANG_PREFILL_ATTN` with the historical `tensor-core` prefill policy. When the experimental launcher declined, the model used native Flash.

| Setting | SM100 route | Batch planner |
| --- | --- | --- |
| `sm100` | Original page-128 split producer. | Can use 2,048 rows for B200 text prompts up to 32,768 tokens with no draft or active prefix-cache tile planner. |
| `sm100-tma` | Page-128 split producer with TMA staging on eligible tiles. | Uses the normal tile planner. |

TMA stages raw payloads and scales for conversion to BF16. Both producers use the same split selection, QK, online softmax, PV, FP32 merge, and protected repair. Both rescale when the softmax maximum increases and wait for PV completion before accumulator reuse. [TMA producer](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L329)<!-- prefix: "__device__ v" --> · [Conversion](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L349)<!-- prefix: "__device__ v" -->

For ordinary multisplit results, each half of the finish block handles one query head. Each thread rescales positive-mass splits to their largest score, accumulates in FP32 in split order, and writes two adjacent BF16 channels. A combined mass that fails `> 0` yields two positive zeros. The producer writes ordinary single-split results. Protected repair uses the whole finish block. [Merge](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L661)<!-- prefix: "__device__ v" --> · [Finish](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L697)<!-- prefix: "prefill_attn" --> · [Producer output](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L504)<!-- prefix: "__device__ v" -->

Both modes use the existing stream-ordered split scratch. Allocation, flag initialization, producer, finish, and release use the attention stream. Wide Flash does not need this split scratch. Graph replay retains the captured maps by value. Keep the cache, page table, queries, and output valid through replay. [Allocation](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L972)<!-- prefix: "auto status " --> · [Release](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L996)<!-- prefix: "const auto r" -->

## Constraints and failure behavior

The shared launcher accepts up to 8,192 query rows and 262,144 visible tokens. It requires at least 144 rows through 32,768 visible tokens. Align Q and output to 16 bytes and K and V payloads to 32 bytes. The device needs compute capability 10.0, CUDA memory pools, and sufficient opt-in shared memory. [Device checks](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L776)<!-- prefix: "cudaError_t " -->

TMA requires 1,024 query rows and a start position of at least 8,192 divisible by 128. It needs 16-byte scale alignment, representable signed 32-bit token coordinates, and 230,400 bytes of opt-in dynamic shared memory. [Tile checks](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L954)<!-- prefix: "if (mode == " --> · [TMA setup](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L842)<!-- prefix: "cudaError_t " -->

If the shared launcher accepts a request but TMA is ineligible, it selects the original split producer before scratch allocation. An unavailable encoder or an `INVALID_VALUE` or `NOT_SUPPORTED` result from tensor-map encoding can trigger this fallback. The same applies when setting the TMA kernel attribute returns invalid value, unsupported operation, invalid device function, or no kernel image. [Encoder status](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L822)<!-- prefix: "cudaError_t " --> · [Attribute status](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L899)<!-- prefix: "if (status =" -->

Validation returns `cudaErrorInvalidValue` for negative positions or row counts, position overflow, null buffers, and insufficient page counts. CUDA failures such as an invalid context, pending execution error, or unexpected setup error return to the caller. Allocation and submission errors also return to the caller without a split retry. [Pending error check](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L852)<!-- prefix: "if (const au" --> · [Dispatch](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/prefill_attn_sm100.cu#L956)<!-- prefix: "if (const au" -->

Archived B200 replays showed bitwise parity and lower operator time against the page-128 split producer. A fixed-history checkpoint probe replaced wide Flash on eligible tiles and changed top tokens at 16,384 and 32,768 prompt tokens. The study left checkpoint quality and end-to-end serving qualification open.

## Verification

Run these commands from the root of [revision `9364826`](https://github.com/para1lel/sglang.cpp/tree/93648269d852e0fee78f7af4d1c3ffe4897aa361) in the default Pixi environment with a reserved B200.

```bash
xmake f -P . -m release -o build --mxfp8-gqa-check=y
xmake build -P . mxfp8-gqa-sm100-check
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    ./build/bin/mxfp8-gqa-sm100-check
```

Require exit zero and rising-max absolute error below 0.00002. For the page-128 graph regression, require a 576-thread TMA producer on the aligned tile at position 8,192. Require a 544-thread original split producer for `sm100` and for the misaligned-scale fallback under `sm100-tma`. Compare all BF16 outputs after poisoned-output replay. [Build target](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/xmake.lua#L215)<!-- prefix: "target(\"mxfp" --> · [Rising-max assertion](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/prefill/mxfp8_gqa_sm100_check.cpp#L144)<!-- prefix: "require(maxi" -->

Use a separate fault probe for TMA setup rejection. Check split selection for a known descriptor or TMA attribute rejection. Check that an unexpected error or pending execution failure returns before allocation or launch. The alignment regression only exercises preflight eligibility.
