#include "sglang/text/gdn_local.h"
#include "sglang/text/gdn/gdn_prefill_kernels.cuh"
#include "sglang/text/gdn/gdn_finish_history.cuh"
#include "sglang/text/gdn/gdn_replay_batch.h"
#include <cstring>

#include <cuda_runtime.h>
#include <type_traits>
#include <new>
#include <stdexcept>

namespace sglang {

// Eligibility for the batched GDN prefix replay: mirrors valid_replay
// (src/text_attention_gdn.cu), which gates the per-layer launches. The
// batched kernels dereference exactly the buffers checked here.
bool text_gdn_replay_batchable(int count, const TextGdnReplayEntry& entry) {
    return count >= 1 && count <= text_gdn_max_verify_rows && entry.rows.qkv &&
           entry.rows.convolved && entry.rows.b && entry.rows.a && entry.weights.a_log &&
           entry.weights.dt_bias && entry.initial.history && entry.initial.recurrent &&
           entry.result.history && entry.result.recurrent &&
           entry.result.history != entry.initial.history &&
           entry.result.recurrent != entry.initial.recurrent;
}

namespace {

/// Per-block entry view: selects this block's table entry once, then hands
/// the recur kernel the same locals the per-layer kernel derives from its
/// arguments.
struct GdnReplayPreamble {
    int local_block;
    int token_count;
    const __nv_bfloat16* convolved;
    const __nv_bfloat16* b;
    const __nv_bfloat16* a;
    TextGdnWeights weights;
    TextGdnState initial;
    TextGdnState final;
};

template <int BlocksPerEntry>
__device__ GdnReplayPreamble gdn_replay_preamble(const TextGdnReplayEntry* table) {
    const TextGdnReplayEntry entry = table[blockIdx.x / BlocksPerEntry];
    return {blockIdx.x % BlocksPerEntry, entry.count, entry.rows.convolved, entry.rows.b,
            entry.rows.a, entry.weights, entry.initial, entry.result};
}

/// Batched twin of gdn_recur (src/text_attention_gdn.cu) for the prefix
/// replay instantiation: round_qk = false, observe = false. The per-block
/// entry is selected by blockIdx.x; the arithmetic after the indirection is
/// unchanged, so each entry's recurrent state is bit-identical to the per-layer
/// kernel.
template <typename S, bool round_qk, bool observe>
__global__ void gdn_batched_recur(const TextGdnReplayEntry* table) {
    __shared__ float qk[2][text_gdn_head_dim];
    __shared__ float warp_squares[recurrent_threads / warp_size];
    __shared__ float decay;
    __shared__ float beta;

    constexpr int recurrent_blocks = S::value_heads * value_blocks_per_head;
    const auto [local_block, token_count, convolved, b, a, weights, initial, final] =
        gdn_replay_preamble<recurrent_blocks>(table);
    const __nv_bfloat16* const observable = nullptr;

    const int head = local_block / value_blocks_per_head;
    const int qk_head = head / S::heads_per_qk;
    const int warp = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int value = (local_block % value_blocks_per_head) * value_rows_per_block + warp;
    const int feature = head * text_gdn_head_dim + value;
    const int q_or_k = threadIdx.x / text_gdn_head_dim;
    const int qk_channel = threadIdx.x % text_gdn_head_dim;
    const int qk_offset = q_or_k * S::qk_width + qk_head * text_gdn_head_dim;
    const int first_warp = q_or_k * (text_gdn_head_dim / warp_size);
    const float negative_a = -expf(__bfloat162float(weights.a_log[head]));
    const float dt_bias = __bfloat162float(weights.dt_bias[head]);

    const auto* recurrent = recurrent_value(initial);
    float state_keys[keys_per_lane];
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state_keys[part] = recurrent[feature * text_gdn_head_dim + part * warp_size + lane];
    }

    for (int row = 0; row < token_count; ++row) {
        const __nv_bfloat16* const mixed =
            convolved + static_cast<std::size_t>(row) * S::qkv_width;
        const float component = __bfloat162float(mixed[qk_offset + qk_channel]);
        const float partial = warp_sum(component * component);
        if (lane == 0) {
            warp_squares[warp] = partial;
        }
        if (threadIdx.x == 0) {
            const int gate = row * S::value_heads + head;
            decay = expf(negative_a * softplus(__bfloat162float(a[gate]) + dt_bias));
            beta = sigmoid(__bfloat162float(b[gate]));
        }
        __syncthreads();

        float squares = 0.0f;
        #pragma unroll
        for (int part = 0; part < text_gdn_head_dim / warp_size; ++part) {
            squares += warp_squares[first_warp + part];
        }
        float normalized = component * rsqrtf(squares + norm_epsilon);
        if constexpr (round_qk) {
            normalized = __bfloat162float(__float2bfloat16(normalized));
        }
        constexpr float query_scale = 0.08838834764831845f;
        qk[q_or_k][qk_channel] = q_or_k == 0 ? normalized * query_scale : normalized;
        __syncthreads();

        float prediction = 0.0f;
        #pragma unroll
        for (int part = 0; part < keys_per_lane; ++part) {
            state_keys[part] *= decay;
            prediction = fmaf(state_keys[part], qk[1][part * warp_size + lane], prediction);
        }
        const float target = __bfloat162float(mixed[2 * S::qk_width + feature]);
        const float correction = beta * (target - warp_sum(prediction));
        float result = 0.0f;
        #pragma unroll
        for (int part = 0; part < keys_per_lane; ++part) {
            const int key = part * warp_size + lane;
            state_keys[part] = fmaf(correction, qk[1][key], state_keys[part]);
            if constexpr (observe) {
                result = fmaf(state_keys[part], qk[0][key], result);
            }
        }
        if constexpr (observe) {
            const float sum = warp_sum(result);
            if (lane == 0) {
                observable[static_cast<std::size_t>(row) * S::value_width + feature] =
                    __float2bfloat16(sum);
            }
        }
        __syncthreads();
    }

    auto* final_recurrent = recurrent_value(final);
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        final_recurrent[feature * text_gdn_head_dim + part * warp_size + lane] =
            state_keys[part];
    }
}

// finish_history comes from the shared private header
// include/sglang/text/gdn/gdn_finish_history.cuh (identical definition to the
// one src/text_attention_gdn.cu used to carry privately).

/// Batched twin of gdn_finish for the replay case (z == nullptr,
/// observable == nullptr, output == nullptr): every block writes one feature's
/// history taps.
template <typename S>
__global__ void gdn_batched_finish(const TextGdnReplayEntry* table) {
    constexpr int finish_blocks_per_entry = S::qkv_width / text_gdn_head_dim;
    const TextGdnReplayEntry entry = table[blockIdx.x / finish_blocks_per_entry];
    const int channel = threadIdx.x;
    const int feature =
        (blockIdx.x % finish_blocks_per_entry) * text_gdn_head_dim + channel;
    finish_history<S>(entry.count, entry.rows.qkv, history_value(entry.initial),
                      history_value(entry.result), feature);
}

template <typename S>
cudaError_t launch_batched_replay(const TextGdnReplayEntry* table, int entries,
                                  cudaStream_t stream) {
    const int recur_blocks = entries * (S::value_heads * value_blocks_per_head);
    gdn_batched_recur<S, false, false>
        <<<dim3(recur_blocks), dim3(recurrent_threads), 0, stream>>>(table);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    const int finish_blocks = entries * (S::qkv_width / text_gdn_head_dim);
    gdn_batched_finish<S>
        <<<dim3(finish_blocks), dim3(text_gdn_head_dim), 0, stream>>>(table);
    return cudaGetLastError();
}

}

TextGdnReplayBatch::~TextGdnReplayBatch() {
    // An in-flight host-to-device copy reads the pinned table; wait for it
    // before the allocations go away.
    if (copy_pending_ && copy_done_) {
        (void)cudaEventSynchronize(copy_done_);
        copy_pending_ = false;
    }
    if (pinned_) (void)cudaFreeHost(pinned_);
    if (device_) (void)cudaFree(device_);
    if (copy_done_) (void)cudaEventDestroy(copy_done_);
}

bool TextGdnReplayBatch::reserve(int capacity) {
    if (capacity <= capacity_) return true;
    // Pre-size the host staging vector first: add() must never allocate on
    // the commit path, and a staging failure must leave the old tables (and
    // the previous capacity contract) intact so the caller can keep the
    // per-layer dispatch.
    try {
        entries_.reserve(capacity);
    } catch (const std::bad_alloc&) {
        return false;
    } catch (const std::length_error&) {
        return false;
    }
    TextGdnReplayEntry* pinned = nullptr;
    TextGdnReplayEntry* device = nullptr;
    cudaEvent_t copy_done = nullptr;
    const std::size_t bytes =
        static_cast<std::size_t>(capacity) * sizeof(TextGdnReplayEntry);
    if (cudaMallocHost(&pinned, bytes) != cudaSuccess ||
        cudaMalloc(&device, bytes) != cudaSuccess ||
        cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming) != cudaSuccess) {
        if (pinned) (void)cudaFreeHost(pinned);
        if (device) (void)cudaFree(device);
        if (copy_done) (void)cudaEventDestroy(copy_done);
        return false;
    }
    // The old pinned table may still be the source of an in-flight upload.
    if (copy_pending_) {
        (void)cudaEventSynchronize(copy_done_);
        copy_pending_ = false;
    }
    if (pinned_) (void)cudaFreeHost(pinned_);
    if (device_) (void)cudaFree(device_);
    if (copy_done_) (void)cudaEventDestroy(copy_done_);
    pinned_ = pinned;
    device_ = device;
    copy_done_ = copy_done;
    capacity_ = capacity;
    entries_.clear();
    return true;
}

cudaError_t TextGdnReplayBatch::launch(bool rank_local, cudaStream_t stream) {
    const int count = size();
    if (count == 0) return cudaSuccess;
    // A previous launch's host-to-device copy may still be executing; the
    // stream orders device work, not this host-side table write, so gate the
    // overwrite on the recorded completion event (a no-op once it finished).
    if (copy_pending_) {
        const auto waited = cudaEventSynchronize(copy_done_);
        if (waited != cudaSuccess) return waited;
        copy_pending_ = false;
    }
    const std::size_t bytes =
        static_cast<std::size_t>(count) * sizeof(TextGdnReplayEntry);
    std::memcpy(pinned_, entries_.data(), bytes);
    auto status = cudaMemcpyAsync(device_, pinned_, bytes, cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) return status;
    status = cudaEventRecord(copy_done_, stream);
    if (status != cudaSuccess) {
        // The copy is enqueued but unfenced: re-establish safety with a full
        // stream sync before handing the error back.
        (void)cudaStreamSynchronize(stream);
        return status;
    }
    copy_pending_ = true;
    entries_.clear();
    return rank_local ? launch_batched_replay<GdnLocal>(device_, count, stream)
                      : launch_batched_replay<GdnFull>(device_, count, stream);
}

}
