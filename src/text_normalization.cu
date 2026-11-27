#include "sglang/text_ops.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_internal.h"
#include "sglang/prefill_fp8.h"
#include "sglang/text/text_cuda_primitives.cuh"
#include "sglang/text/quantization/quantize_warp.cuh"
#include "sglang/text/quantization/rms_quantize.cuh"
#include "sglang/text/quantization/prefill_rms_vector.cuh"
#include "sglang/text/mtp/draft_batch_ops.h"
#include "sglang/text/mtp/tc_vocab.h"

#include <cuda_fp8.h>

namespace sglang {
namespace {

using namespace text_kernel;

__device__ __forceinline__ void rms_norm_row(const __nv_bfloat16* branch,
                                             const __nv_bfloat16* residual,
                                             const __nv_bfloat16* weight, int width,
                                             __nv_bfloat16* output,
                                             __nv_bfloat16* residual_output) {
    __shared__ float warp_squares[block_warps];
    __shared__ float inverse_rms;
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    const auto count = static_cast<std::size_t>(width);
    float squares = 0.0f;
    for (std::size_t col = threadIdx.x; col < count; col += block_threads) {
        const float value = residual_sum(branch, residual, col);
        squares = fmaf(value, value, squares);
    }
    squares = warp_sum(squares);
    if (lane == 0) {
        warp_squares[warp] = squares;
    }
    __syncthreads();
    if (warp == 0) {
        squares = warp_sum(lane < block_warps ? warp_squares[lane] : 0.0f);
        if (lane == 0) {
            inverse_rms = rsqrtf(squares / static_cast<float>(width) + rms_epsilon);
        }
    }
    __syncthreads();
    for (std::size_t col = threadIdx.x; col < count; col += block_threads) {
        const float value = residual_sum(branch, residual, col);
        const float gain = 1.0f + __bfloat162float(weight[col]);
        if (residual_output != nullptr) {
            residual_output[col] = __float2bfloat16_rn(value);
        }
        output[col] = __float2bfloat16_rn((value * inverse_rms) * gain);
    }
}

template <bool Batched>
__global__ void rms_norm_kernel(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                                const __nv_bfloat16* weight, int width, __nv_bfloat16* output,
                                __nv_bfloat16* residual_output) {
    const auto offset = Batched ? static_cast<std::size_t>(blockIdx.x) * width : 0;
    rms_norm_row(branch + offset, residual ? residual + offset : nullptr, weight, width,
                 output + offset, residual_output ? residual_output + offset : nullptr);
}

__global__ void rms_norm_pair_rows_kernel(const __nv_bfloat16* first, const __nv_bfloat16* second,
                                          const __nv_bfloat16* first_weight,
                                          const __nv_bfloat16* second_weight, int width,
                                          __nv_bfloat16* output) {
    const auto offset = static_cast<std::size_t>(blockIdx.x) * width;
    const bool second_half = blockIdx.y != 0;
    rms_norm_row((second_half ? second : first) + offset, nullptr,
                 second_half ? second_weight : first_weight, width,
                 output + 2 * offset + blockIdx.y * static_cast<std::size_t>(width), nullptr);
}

__device__ void stage_rms_vector(const __nv_bfloat16* input, __nv_bfloat16* shared,
                                 __nv_bfloat16* output = nullptr) {
    const uint4 vector = load_128(input);
    if (output) *reinterpret_cast<uint4*>(output) = vector;
    const auto address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    asm("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
        :: "r"(address), "r"(vector.x), "r"(vector.y), "r"(vector.z), "r"(vector.w)
        : "memory");
}

__device__ void store_rms(float value, float inverse_rms, const __nv_bfloat16* weight,
                          std::size_t col, __nv_bfloat16* output,
                          __nv_bfloat16* residual_output) {
    const float gain = 1.0f + __bfloat162float(weight[col]);
    if (residual_output != nullptr) {
        residual_output[col] = __float2bfloat16_rn(value);
    }
    output[col] = __float2bfloat16_rn((value * inverse_rms) * gain);
}

template <bool HasResidual, bool StoresResidual, bool Pack = false, bool EmitsState = false>
__device__ void rms_norm_vector(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                                const __nv_bfloat16* weight, __nv_bfloat16* output,
                                __nv_bfloat16* residual_output, __nv_bfloat16* packed = nullptr,
                                const TextModelStateRow* state = nullptr) {
    __shared__ alignas(16) __nv_bfloat16 branch_values[rms_hidden_width];
    __shared__ alignas(16) __nv_bfloat16 residual_values[HasResidual ? rms_hidden_width : 1];
    __shared__ alignas(16) __nv_bfloat16 gains[rms_hidden_width];
    constexpr int vector_width = sizeof(uint4) / sizeof(__nv_bfloat16);
    static_assert(rms_hidden_width % block_threads == 0 && rms_hidden_width % vector_width == 0);
    TextModelStateRow destination{};
    if constexpr (EmitsState) destination = *state;
    for (int col = threadIdx.x * vector_width; col < rms_hidden_width;
         col += block_threads * vector_width) {
        stage_rms_vector(branch + col, branch_values + col,
                         destination.hidden ? destination.hidden + col : nullptr);
        if constexpr (HasResidual) {
            stage_rms_vector(residual + col, residual_values + col,
                             destination.residual ? destination.residual + col : nullptr);
        }
        stage_rms_vector(weight + col, gains + col);
    }
    __syncthreads();

    constexpr int values_per_thread = rms_hidden_width / block_threads;
    float values[values_per_thread];
    float squares = 0.0f;
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
        const int col = threadIdx.x + index * block_threads;
        const float value = residual_sum(branch_values, HasResidual ? residual_values : nullptr,
                                         col);
        values[index] = value;
        squares = fmaf(value, value, squares);
    }
    const float inverse_rms = rms_inverse(squares, rms_hidden_width);
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
        const int col = threadIdx.x + index * block_threads;
        if constexpr (Pack) {
            const float gain = 1.0f + __bfloat162float(gains[col]);
            const auto normalized = __float2bfloat16_rn((values[index] * inverse_rms) * gain);
            output[col] = normalized;
            packed[col] = normalized;
        } else {
            store_rms(values[index], inverse_rms, gains, col, output,
                      StoresResidual ? residual_output : nullptr);
        }
    }
}

template <bool HasResidual, bool StoresResidual>
__global__ void rms_norm_vector_kernel(const __nv_bfloat16* branch,
                                       const __nv_bfloat16* residual,
                                       const __nv_bfloat16* weight, __nv_bfloat16* output,
                                       __nv_bfloat16* residual_output) {
    rms_norm_vector<HasResidual, StoresResidual>(branch, residual, weight, output, residual_output);
}

__global__ void final_rms_bound(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                                const __nv_bfloat16* weight, __nv_bfloat16* output,
                                const TextModelStateRow* state) {
    rms_norm_vector<true, false, false, true>(branch, residual, weight, output, nullptr,
                                             nullptr, state);
}

__global__ void draft_input_rms(const __grid_constant__ decoder::DraftRmsRows embeddings,
                               const __grid_constant__ decoder::DraftRmsRows hidden,
                               const __nv_bfloat16* embedding_weight,
                               const __nv_bfloat16* hidden_weight) {
    const auto& row = blockIdx.y == 0 ? embeddings.rows[blockIdx.x] : hidden.rows[blockIdx.x];
    const auto* weight = blockIdx.y == 0 ? embedding_weight : hidden_weight;
    rms_norm_vector<false, false>(row.branch, nullptr, weight, row.output, nullptr);
}

__global__ void draft_output_rms(const __grid_constant__ decoder::DraftRmsRows rows,
                                const __nv_bfloat16* weight) {
    const auto& row = rows.rows[blockIdx.x];
    rms_norm_vector<true, false>(row.branch, row.residual, weight, row.output, nullptr);
}

__global__ void packed_vocab_rms_kernel(const __grid_constant__ decoder::TcVocabRmsRows rows,
                                       const __nv_bfloat16* weight, __nv_bfloat16* packed) {
    const auto& row = rows.rows[blockIdx.x];
    rms_norm_vector<true, false, true>(row.branch, row.residual, weight, row.normalized,
                                      nullptr, packed + blockIdx.x * rms_hidden_width);
}

template <bool HasResidual, bool PackedGroups = false, bool LaunchDependents = false,
          bool AsyncStage = PackedGroups>
__device__ __forceinline__ void rms_quantize(const __nv_bfloat16* branch, __nv_bfloat16* residual,
                                     const __nv_bfloat16* weight, __nv_bfloat16* normalized,
                                     TextFp8Activation activation) {
    constexpr int vector_width = sizeof(uint4) / sizeof(__nv_bfloat16);
    __shared__ alignas(16) __nv_bfloat16 branch_values[rms_hidden_width];
    __shared__ alignas(16) __nv_bfloat16 residual_values[HasResidual ? rms_hidden_width : 1];
    __shared__ alignas(16) __nv_bfloat16 gains[rms_hidden_width];
    for (int col = threadIdx.x * vector_width; col < rms_hidden_width;
         col += block_threads * vector_width) {
        if constexpr (AsyncStage) {
            stage_rms_vector_async(branch + col, branch_values + col);
            if constexpr (HasResidual) {
                stage_rms_vector_async(residual + col, residual_values + col);
            }
            stage_rms_vector_async(weight + col, gains + col);
        } else {
            stage_rms_vector(branch + col, branch_values + col);
            if constexpr (HasResidual) {
                stage_rms_vector(residual + col, residual_values + col);
            }
            stage_rms_vector(weight + col, gains + col);
        }
    }
    if constexpr (AsyncStage) {
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    if constexpr (LaunchDependents) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    rms_quantize_staged<HasResidual, PackedGroups>(branch_values, residual_values, gains,
                                                  residual, normalized, activation);
}

template <bool HasResidual, bool PackedGroups = false, bool LaunchDependents = false,
          bool AsyncStage = PackedGroups>
__global__ void prefill_rms_quantize(const __nv_bfloat16* branch, __nv_bfloat16* residual,
                                     const __nv_bfloat16* weight, __nv_bfloat16* normalized,
                                     TextFp8Activation activation) {
    const auto offset = static_cast<std::size_t>(blockIdx.x) * rms_hidden_width;
    activation.values += offset;
    activation.scales += offset / text_fp8_group_size;
    rms_quantize<HasResidual, PackedGroups, LaunchDependents, AsyncStage>(
        branch + offset, HasResidual ? residual + offset : nullptr, weight,
        normalized + offset, activation);
}

template <bool HasResidual, typename Rows>
__global__ void verify_rms_kernel(const __grid_constant__ Rows rows,
                                  const __nv_bfloat16* weight) {
    const auto& row = rows.entries[blockIdx.x];
    rms_quantize<HasResidual, true>(row.branch, row.residual, weight, row.normalized, row.output);
}

template <bool ApplySwiglu, typename Rows>
__global__ void verify_quantize_kernel(const __grid_constant__ Rows rows, int width) {
    const auto& row = rows.entries[blockIdx.y];
    quantize_warp<ApplySwiglu>(row.input, row.up, width, row.output);
}

template <typename Rows>
cudaError_t launch_verify_quantize(Rows rows, int count, int width, cudaStream_t stream) {
    const int groups = width / text_fp8_group_size;
    const dim3 grid((groups + block_warps - 1) / block_warps, count);
    void* args[] = {&rows, &width};
    const auto kernel = rows.entries[0].up ? verify_quantize_kernel<true, Rows>
                                           : verify_quantize_kernel<false, Rows>;
    return cudaLaunchKernel(kernel, grid, block_threads, args, 0, stream);
}

cudaError_t wide_verify_launch_status(cudaError_t direct) {
    const auto recorded = cudaPeekAtLastError();
    if (!decoder::CudaError::terminal(direct) && decoder::CudaError::terminal(recorded)) {
        return cudaGetLastError();
    }
    if (direct != cudaSuccess && recorded == direct) {
        (void)cudaGetLastError();
    }
    return direct;
}
}

cudaError_t decoder::pack_tc_vocab_rms(const TcVocabRmsRows& rows, const Bf16* weight,
                                       Bf16* packed, cudaStream_t stream) {
    void* args[] = {const_cast<TcVocabRmsRows*>(&rows), &weight, &packed};
    return wide_verify_launch_status(cudaLaunchKernel(packed_vocab_rms_kernel,
        tc_vocab_wide_rows, block_threads, args, 0, stream));
}

cudaError_t decoder::normalize_draft_inputs(const DraftRmsRows& embeddings,
                                            const DraftRmsRows& hidden,
                                            const Bf16* embedding_weight,
                                            const Bf16* hidden_weight, int count,
                                            cudaStream_t stream) {
    void* args[] = {const_cast<DraftRmsRows*>(&embeddings), const_cast<DraftRmsRows*>(&hidden),
                    &embedding_weight, &hidden_weight};
    return cudaLaunchKernel(draft_input_rms, dim3(count, 2), block_threads,
                            args, 0, stream);
}

cudaError_t decoder::normalize_draft_output(const DraftRmsRows& rows, const Bf16* weight,
                                             int count, cudaStream_t stream) {
    void* args[] = {const_cast<DraftRmsRows*>(&rows), &weight};
    return cudaLaunchKernel(draft_output_rms, count, block_threads,
                            args, 0, stream);
}

cudaError_t decoder::verify_quantize_fp8(VerifyQuantizeRows rows, int width, cudaStream_t stream) {
    return launch_verify_quantize(rows, fused_verify_rows, width, stream);
}

cudaError_t decoder::verify_quantize_partial_fp8(VerifyQuantizeRows rows, int count, int width,
                                                 cudaStream_t stream) {
    if (count < 3 || count >= fused_verify_rows || width <= 0 || width % text_fp8_group_size) {
        return cudaErrorInvalidValue;
    }
    return launch_verify_quantize(rows, count, width, stream);
}

cudaError_t decoder::verify_quantize_fp8_wide(WideVerifyQuantizeRows rows, int count, int width,
                                             cudaStream_t stream) {
    if (count <= fused_verify_rows || count > WideVerifyQuantizeRows::capacity ||
        width <= 0 || width % text_fp8_group_size != 0) {
        return cudaErrorInvalidValue;
    }
    return wide_verify_launch_status(launch_verify_quantize(rows, count, width, stream));
}

cudaError_t decoder::verify_rms_quantize(VerifyRmsQuantizeRows rows, const __nv_bfloat16* weight,
                                        cudaStream_t stream) {
    void* args[] = {&rows, &weight};
    const auto kernel = rows.entries[0].residual ? verify_rms_kernel<true, VerifyRmsQuantizeRows>
                                                 : verify_rms_kernel<false, VerifyRmsQuantizeRows>;
    return cudaLaunchKernel(kernel, fused_verify_rows, block_threads, args, 0, stream);
}

cudaError_t decoder::verify_rms_quantize_partial(VerifyRmsQuantizeRows rows, int count,
                                                 const __nv_bfloat16* weight, cudaStream_t stream) {
    if (count < 3 || count >= fused_verify_rows) return cudaErrorInvalidValue;
    void* args[] = {&rows, &weight};
    const auto kernel = rows.entries[0].residual ? verify_rms_kernel<true, VerifyRmsQuantizeRows>
                                                 : verify_rms_kernel<false, VerifyRmsQuantizeRows>;
    return cudaLaunchKernel(kernel, count, block_threads, args, 0, stream);
}

cudaError_t decoder::verify_rms_quantize_wide(WideVerifyRmsQuantizeRows rows, int count,
                                             const __nv_bfloat16* weight, cudaStream_t stream) {
    if (count <= fused_verify_rows || count > WideVerifyRmsQuantizeRows::capacity) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&rows, &weight};
    const auto kernel = rows.entries[0].residual
        ? verify_rms_kernel<true, WideVerifyRmsQuantizeRows>
        : verify_rms_kernel<false, WideVerifyRmsQuantizeRows>;
    return wide_verify_launch_status(cudaLaunchKernel(kernel, count, block_threads, args, 0,
                                                       stream));
}

namespace {

bool aligned_prefill_norm(const decoder::PrefillPeerNormRow& row) {
    return row.branch && row.gain && row.activation.values && row.activation.scales &&
        aligned(row.branch, alignof(uint4)) && aligned(row.residual, alignof(uint4)) &&
        aligned(row.gain, alignof(uint4)) && aligned(row.normalized, alignof(uint4)) &&
        aligned(row.activation.values, alignof(uint2)) &&
        aligned(row.activation.scales, alignof(float));
}

template <bool Peer>
cudaError_t launch_prefill_norm_vector(decoder::PrefillPeerNormRow row,
                                       const __nv_bfloat16* remote, cudaStream_t stream) {
    const auto pending = cudaGetLastError();
    if (pending != cudaSuccess) return pending;
    using decoder::prefill_rms_detail::vector_rms;
    const auto kernel = (Peer || row.residual)
        ? (row.normalized ? vector_rms<true, true, Peer> : vector_rms<true, false, Peer>)
        : (row.normalized ? vector_rms<false, true, Peer> : vector_rms<false, false, Peer>);
    void* args[] = {
        &row.branch, &row.residual, &row.gain, &row.normalized, &row.activation, &remote};
    const auto submitted = cudaLaunchKernel(kernel, grouped_prefill_max_rows,
                                            block_threads, args, 0, stream);
    const auto recorded = cudaGetLastError();
    if (decoder::CudaError::terminal(recorded) && !decoder::CudaError::terminal(submitted)) {
        return recorded;
    }
    return submitted == cudaSuccess ? recorded : submitted;
}

}

bool decoder::aligned_prefill_peer_norm(const PrefillPeerNormRow& row) {
    return row.residual && aligned_prefill_norm(row);
}

cudaError_t decoder::prefill_peer_rms_quantize(PrefillPeerNormRow row, const __nv_bfloat16* peer,
                                              cudaStream_t stream) {
    return launch_prefill_norm_vector<true>(row, peer, stream);
}

std::optional<cudaError_t> decoder::try_prefill_rms_quantize_vector(
    const __nv_bfloat16* branch, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    __nv_bfloat16* normalized, TextFp8Activation activation, cudaStream_t stream) {
    const PrefillPeerNormRow row{branch, residual, weight, normalized, activation};
    if (!aligned_prefill_norm(row)) return std::nullopt;
    cudaStreamCaptureStatus capture{};
    const auto query = cudaStreamIsCapturing(stream, &capture);
    if (query != cudaSuccess) return query;
    if (capture != cudaStreamCaptureStatusNone) return std::nullopt;
    return launch_prefill_norm_vector<false>(row, nullptr, stream);
}

cudaError_t decoder::tensor_core_prefill_rms_quantize(
    const __nv_bfloat16* branch, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    int width, int rows, __nv_bfloat16* normalized, TextFp8Activation activation,
    cudaStream_t stream, RmsQuantizeLaunch launch) {
    if (width != rms_hidden_width || !aligned(branch, alignof(uint4)) ||
        !aligned(residual, alignof(uint4)) || !aligned(weight, alignof(uint4))) {
        const auto status = text_rms_norm_rows(branch, residual, weight, width, rows,
                                              normalized, residual, stream);
        return status == cudaSuccess
            ? text_quantize_fp8_rows(normalized, width, rows, activation, stream)
            : status;
    }
    void* args[] = {&branch, &residual, &weight, &normalized, &activation};
    const bool prefetch_weights = launch == RmsQuantizeLaunch::prefetch_decode_weights;
    if (launch == RmsQuantizeLaunch::full_group_prefill && rows == 4096 &&
        aligned(activation.values, alignof(uint2))) {
        const auto pending = cudaGetLastError();
        if (pending != cudaSuccess) return pending;
        const auto kernel = residual ? prefill_rms_quantize<true, true, false, false>
                                     : prefill_rms_quantize<false, true, false, false>;
        const auto status = cudaLaunchKernel(kernel, rows, block_threads, args, 0, stream);
        const auto recorded = cudaGetLastError();
        return status == cudaSuccess ? recorded : status;
    }
    if (residual && rows >= 2 && rows <= 32 && aligned(activation.values, alignof(uint2))) {
        const bool early = prefetch_weights ||
                           (launch == RmsQuantizeLaunch::packed_gate_up && rows == 32);
        const auto kernel = early
            ? prefill_rms_quantize<true, true, true> : prefill_rms_quantize<true, true>;
        return cudaLaunchKernel(kernel, rows, block_threads, args, 0, stream);
    }
    if (prefetch_weights) {
        const auto kernel = residual ? prefill_rms_quantize<true, false, true>
                                     : prefill_rms_quantize<false, false, true>;
        return cudaLaunchKernel(kernel, rows, block_threads, args, 0, stream);
    }
    const auto kernel = residual ? prefill_rms_quantize<true> : prefill_rms_quantize<false>;
    return cudaLaunchKernel(kernel, rows, block_threads, args, 0, stream);
}

cudaError_t text_rms_norm(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                         const __nv_bfloat16* weight, int width, __nv_bfloat16* output,
                         __nv_bfloat16* residual_output, cudaStream_t stream) {
    if (branch == nullptr || weight == nullptr || output == nullptr || width <= 0 ||
        output == residual_output) {
        return cudaErrorInvalidValue;
    }
    if (width == rms_hidden_width && aligned(branch, alignof(uint4)) &&
        aligned(residual, alignof(uint4)) && aligned(weight, alignof(uint4))) {
        void* args[] = {&branch, &residual, &weight, &output, &residual_output};
        const bool store_residual = residual_output != nullptr;
        const auto kernel = residual == nullptr
            ? (store_residual ? rms_norm_vector_kernel<false, true>
                              : rms_norm_vector_kernel<false, false>)
            : (store_residual ? rms_norm_vector_kernel<true, true>
                              : rms_norm_vector_kernel<true, false>);
        return cudaLaunchKernel(kernel, 1, block_threads, args, 0, stream);
    }
    void* args[] = {&branch, &residual, &weight, &width, &output, &residual_output};
    return cudaLaunchKernel(rms_norm_kernel<false>, 1, block_threads, args, 0, stream);
}

cudaError_t text_final_rms_bound(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                                const __nv_bfloat16* weight, __nv_bfloat16* output,
                                const TextModelStateRow* device_state, cudaStream_t stream) {
    void* args[] = {&branch, &residual, &weight, &output, &device_state};
    return cudaLaunchKernel(final_rms_bound, 1, block_threads, args, 0, stream);
}

cudaError_t text_rms_norm_rows(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                              const __nv_bfloat16* weight, int width, int rows,
                              __nv_bfloat16* output, __nv_bfloat16* residual_output,
                              cudaStream_t stream) {
    if (!branch || !weight || !output || width <= 0 || rows <= 0 || output == residual_output) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&branch, &residual, &weight, &width, &output, &residual_output};
    return cudaLaunchKernel(rms_norm_kernel<true>, rows, block_threads, args, 0, stream);
}

cudaError_t text_rms_norm_pair_rows(const __nv_bfloat16* first, const __nv_bfloat16* second,
                                   const __nv_bfloat16* first_weight,
                                   const __nv_bfloat16* second_weight, int width, int rows,
                                   __nv_bfloat16* output, cudaStream_t stream) {
    if (!first || !second || !first_weight || !second_weight || !output ||
        width <= 0 || rows <= 0) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&first, &second, &first_weight, &second_weight, &width, &output};
    return cudaLaunchKernel(rms_norm_pair_rows_kernel, dim3(rows, 2), block_threads, args, 0,
                            stream);
}

}
