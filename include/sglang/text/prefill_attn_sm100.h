#pragma once

#include "sglang/mxfp8_gqa.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <optional>

namespace sglang {

namespace decoder {

struct PairedPrefillScratch {
    std::byte* keys = nullptr;
    std::size_t key_bytes = 0;
    std::byte* values = nullptr;
    std::size_t value_bytes = 0;
};

struct PagedPrefillSequence {
    Mxfp8KvCache cache;
    Mxfp8PageTable pages;
    int position;
    int rows;
    const __nv_bfloat16* queries;
    __nv_bfloat16* output;
};

std::optional<cudaError_t> try_batched_paged_prefill(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream);

std::optional<cudaError_t> try_batched_partial_paged_prefill(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream);

inline constexpr int staged_prefill_batch_size = 4;
inline constexpr int staged_prefill_batch_min_position = 1024;
inline constexpr int staged_prefill_batch_max_position = 7168;
inline constexpr int staged_prefill_batch_long_max_position = 130048;
inline constexpr int staged_serial_min_position = 32768;
inline constexpr int staged_serial_max_tokens = 131072;
inline constexpr int staged_serial_min_rows = 1024 - mxfp8_group_size;

bool wants_long_serial_staging(int position, int rows);

std::optional<cudaError_t> try_long_serial_staged_paged_prefill(
    const PagedPrefillSequence& sequence, cudaStream_t stream, PairedPrefillScratch scratch);

std::optional<cudaError_t> try_batched_staged_paged_prefill(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream,
    PairedPrefillScratch scratch);

std::optional<cudaError_t> try_long_staged_paged_prefill(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream,
    PairedPrefillScratch scratch);

cudaError_t tensor_core_multirow_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream,
    PairedPrefillScratch scratch);

}

}
