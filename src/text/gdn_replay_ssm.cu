#include "sglang/text/gdn/replay_ssm.h"

#include "sglang/text/gdn_local.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sglang::gdn_replay_ssm {
namespace {

constexpr int warp_size = 32;
constexpr int head_dim = text_gdn_head_dim;
constexpr int taps = text_gdn_history_taps;
constexpr int drafts = text_gdn_max_verify_rows;
constexpr int max_committed = window + drafts - 1;
constexpr int key_parts = head_dim / warp_size;
constexpr int value_warps = 8;
constexpr int values_per_warp = 4;
constexpr int tile_values = value_warps * values_per_warp;
constexpr int tiles = head_dim / tile_values;
constexpr int finish_warps = 4;
constexpr float epsilon = 1.0e-6F;
constexpr float query_scale = 0.08838834764831845F;
static_assert(head_dim == 128 && drafts == 5 && max_committed + drafts == capacity);

struct SingleLane {
    Lane job;
    const State* device_state;
};

struct HistoryTile {
    float keys[max_committed][head_dim];
    float updates[max_committed][tile_values];
    float decay[max_committed];
};

template <int Rows> struct DraftTile {
    float2 gates[Rows];
    float qk[Rows][2][head_dim];
};

__device__ float warp_sum(float value) {
    for (int offset = warp_size / 2; offset != 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ float sigmoid(float value) {
    return 1.0F / (1.0F + expf(-value));
}

__device__ Lane get_lane(const SingleLane& jobs) {
    auto job = jobs.job;
    if (jobs.device_state != nullptr) {
        job.state = *jobs.device_state;
    }
    return job;
}

__device__ const Lane& get_lane(const Lanes& jobs) {
    return jobs.lanes[blockIdx.z];
}

__device__ const Lane& get_lane(const Lane* jobs) {
    return jobs[blockIdx.z];
}

__host__ __device__ bool valid_state(State state) {
    return (state.checkpoint.history != nullptr) && (state.checkpoint.recurrent != nullptr) &&
           (state.keys != nullptr) && (state.updates != nullptr) && (state.decay != nullptr) &&
           (state.raw != nullptr) && (state.cursor != nullptr);
}

__host__ __device__ bool valid_rows(int count, int maximum) {
    return count > 0 && count <= maximum;
}

__host__ __device__ bool valid_lane(const Lane& job, int maximum) {
    return valid_rows(job.rows, maximum) && (job.inputs.qkv != nullptr) &&
           (job.inputs.convolved != nullptr) &&
           reinterpret_cast<std::uintptr_t>(job.inputs.convolved) % alignof(float) == 0 &&
           (job.inputs.a != nullptr) && (job.inputs.b != nullptr) && (job.z != nullptr) &&
           (job.observable != nullptr) && (job.output != nullptr);
}

__device__ int slot(Cursor cursor, int offset) {
    const int index = cursor.first + offset;
    return index < capacity ? index : index - capacity;
}

__device__ bool idle(Cursor cursor) {
    return cursor.first >= 0 && cursor.first < capacity && cursor.committed >= 0 &&
           cursor.committed <= max_committed && cursor.pending == 0;
}

template <typename S>
__device__ __nv_bfloat16 raw_sample(State state, Cursor cursor, int offset, int channel) {
    if (offset < 0) {
        return state.checkpoint.history[(channel * taps) + offset + taps];
    }
    return state.raw[slot(cursor, offset) * S::qkv_width + channel];
}

template <typename S> __device__ float* query_row(const Lane& job, int row) {
    return reinterpret_cast<float*>(job.inputs.convolved + row * S::qkv_width);
}

template <typename S> __device__ __nv_bfloat16* value_row(const Lane& job, int row) {
    return job.inputs.convolved + row * S::qkv_width + 2 * S::qk_width;
}

template <typename S, int Rows>
__device__ void mix_channel(const Lane& job, Cursor cursor, const __nv_bfloat16* weights,
                            int channel, __nv_bfloat16 (&mixed)[Rows]) {
    float weights_local[taps + 1];
    float samples[taps];
#pragma unroll
    for (int tap = 0; tap <= taps; ++tap) {
        weights_local[tap] = __bfloat162float(weights[(channel * (taps + 1)) + tap]);
    }
#pragma unroll
    for (int tap = 0; tap < taps; ++tap) {
        const auto raw = raw_sample<S>(job.state, cursor, cursor.committed - taps + tap, channel);
        samples[tap] = __bfloat162float(raw);
        if (cursor.committed >= window) {
            const auto held = cursor.committed == window
                                  ? raw
                                  : raw_sample<S>(job.state, cursor, window - taps + tap, channel);
            job.state.checkpoint.history[(channel * taps) + tap] = held;
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row >= job.rows)
            break;
        const int append = slot(cursor, cursor.committed + row);
        const auto raw = job.inputs.qkv[row * S::qkv_width + channel];
        job.state.raw[append * S::qkv_width + channel] = raw;
        float sum = 0.0F;
#pragma unroll
        for (int tap = 0; tap < taps; ++tap) {
            sum = fmaf(samples[tap], weights_local[tap], sum);
        }
        const float current = __bfloat162float(raw);
        sum = fmaf(current, weights_local[taps], sum);
        mixed[row] = __float2bfloat16(sum * sigmoid(sum));
#pragma unroll
        for (int tap = 0; tap < taps - 1; ++tap) {
            samples[tap] = samples[tap + 1];
        }
        samples[taps - 1] = current;
    }
}

template <int Rows> struct NormTile {
    float partial[2][Rows][key_parts];
};

template <typename S, int Rows>
__device__ void normalize_draft(const Lane& job, Cursor cursor, const __nv_bfloat16* weights,
                                int qk, NormTile<Rows>& tile) {
    const int component = static_cast<int>(threadIdx.x % head_dim);
    const int key = static_cast<int>(threadIdx.x / head_dim);
    const int channel = key * S::qk_width + qk + component;
    __nv_bfloat16 mixed[Rows];
    mix_channel<S, Rows>(job, cursor, weights, channel, mixed);
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row >= job.rows)
            break;
        const float value = __bfloat162float(mixed[row]);
        const float squares = warp_sum(value * value);
        if (component % warp_size == 0)
            tile.partial[key][row][component / warp_size] = squares;
    }
    __syncthreads();
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row >= job.rows)
            break;
        float squares = 0.0F;
#pragma unroll
        for (int part = 0; part < key_parts; ++part)
            squares += tile.partial[key][row][part];
        const float scale = rsqrtf(squares + epsilon);
        const float normalized = __bfloat162float(mixed[row]) * scale;
        if (key) {
            const int append = slot(cursor, cursor.committed + row);
            job.state.keys[append * S::qk_width + qk + component] = normalized;
        } else {
            query_row<S>(job, row)[qk + component] = normalized * query_scale;
        }
    }
}

template <typename S, int Rows, typename Jobs>
__global__ void convolve(const __grid_constant__ Jobs jobs, int max_rows,
                         const __nv_bfloat16* weights) {
    const auto& job = get_lane(jobs);
    const auto state = job.state;
    if (!valid_lane(job, max_rows) || !valid_state(state))
        return;
    const auto cursor = *state.cursor;
    if (!idle(cursor))
        return;
    const int group = static_cast<int>(blockIdx.x);
    if (group < S::qk_width / head_dim) {
        __shared__ NormTile<Rows> tile;
        const int qk = group * head_dim;
        normalize_draft<S, Rows>(job, cursor, weights, qk, tile);
        return;
    }
    if (threadIdx.x >= head_dim)
        return;
    const int channel = S::qk_width + group * head_dim + threadIdx.x;
    __nv_bfloat16 mixed[Rows];
    mix_channel<S, Rows>(job, cursor, weights, channel, mixed);
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row >= job.rows)
            break;
        value_row<S>(job, row)[channel - 2 * S::qk_width] = mixed[row];
    }
}

template <typename S>
__device__ void load_history_weights(State state, Cursor cursor, int head, HistoryTile& history,
                                     int count) {
    const auto row = threadIdx.x;
    if (row < static_cast<unsigned int>(count)) {
        const int append = slot(cursor, static_cast<int>(row));
        history.decay[row] = state.decay[append * S::value_heads + head];
    }
}

template <typename S>
__device__ void load_history_keys(State state, Cursor cursor, int head, HistoryTile& history,
                                  int count) {
    const int key_group = head / S::heads_per_qk * head_dim;
    const auto elements = static_cast<unsigned int>(count * head_dim);
    for (auto index = threadIdx.x; index < elements; index += blockDim.x) {
        const int row = static_cast<int>(index / head_dim);
        const int key = static_cast<int>(index % head_dim);
        history.keys[row][key] = state.keys[slot(cursor, row) * S::qk_width + key_group + key];
    }
}

template <typename S>
__device__ void load_history_updates(State state, Cursor cursor, int feature, HistoryTile& history,
                                     int count) {
    const auto elements = static_cast<unsigned int>(count * tile_values);
    for (auto index = threadIdx.x; index < elements; index += blockDim.x) {
        const int row = static_cast<int>(index / tile_values);
        const int val = static_cast<int>(index % tile_values);
        history.updates[row][val] =
            state.updates[slot(cursor, row) * S::value_width + feature + val];
    }
}

__device__ void read_held(float (&held)[values_per_warp][key_parts], const float* anchor,
                          int lane) {
#pragma unroll
    for (int pair = 0; pair < values_per_warp; ++pair) {
#pragma unroll
        for (int part = 0; part < key_parts; ++part) {
            held[pair][part] = anchor[(pair * value_warps * head_dim) + (part * warp_size) + lane];
        }
    }
}

__device__ void write_held(const float (&held)[values_per_warp][key_parts], float* output,
                           int lane) {
#pragma unroll
    for (int pair = 0; pair < values_per_warp; ++pair) {
#pragma unroll
        for (int part = 0; part < key_parts; ++part) {
            output[(pair * value_warps * head_dim) + (part * warp_size) + lane] = held[pair][part];
        }
    }
}

__device__ void replay_committed(float (&held)[values_per_warp][key_parts], int val,
                                 const HistoryTile& history, int first, int count) {
    const int lane = static_cast<int>(threadIdx.x % warp_size);
    for (int row = first; row < count; ++row) {
        const float decay = history.decay[row];
#pragma unroll
        for (int pair = 0; pair < values_per_warp; ++pair) {
            const float correction = history.updates[row][val + (pair * value_warps)];
#pragma unroll
            for (int part = 0; part < key_parts; ++part) {
                const float key = history.keys[row][(part * warp_size) + lane];
                held[pair][part] = fmaf(correction, key, held[pair][part] * decay);
            }
        }
    }
}

template <typename S>
__device__ const float* draft_key(State state, Cursor cursor, int qk, int row) {
    return state.keys + slot(cursor, cursor.committed + row) * S::qk_width + qk;
}

template <typename S, int Rows>
__device__ void load_draft_gates(const Lane& job, Cursor cursor, TextGdnWeights weights,
                                 DraftTile<Rows>& draft, int head, bool store_decay) {
    const int row = static_cast<int>(threadIdx.x);
    if (row >= job.rows)
        return;
    const int index = row * S::value_heads + head;
    const float input =
        __bfloat162float(job.inputs.a[index]) + __bfloat162float(weights.dt_bias[head]);
    const float dt = fmaxf(input, 0.0F) + log1pf(expf(-fabsf(input)));
    const float decay = expf(-expf(__bfloat162float(weights.a_log[head])) * dt);
    draft.gates[row] = {decay, sigmoid(__bfloat162float(job.inputs.b[index]))};
    if (store_decay) {
        const int append = slot(cursor, cursor.committed + row);
        job.state.decay[append * S::value_heads + head] = decay;
    }
}

template <typename S, int Rows>
__device__ void load_draft_qk(const Lane& job, Cursor cursor, int qk, DraftTile<Rows>& draft) {
    const int channel = static_cast<int>(threadIdx.x % head_dim);
    const int key = static_cast<int>(threadIdx.x / head_dim);
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row >= job.rows)
            break;
        const auto* input =
            key ? draft_key<S>(job.state, cursor, qk, row) : query_row<S>(job, row) + qk;
        draft.qk[row][key][channel] = input[channel];
    }
}

struct DraftResult {
    float update[values_per_warp];
    float observable[values_per_warp];
};

__device__ DraftResult solve_row(float (&held)[values_per_warp][key_parts],
                                 const float (&qk)[2][head_dim], int lane, float2 gates,
                                 const float (&target)[values_per_warp]) {
    DraftResult result{};
    float prediction[values_per_warp]{};
#pragma unroll
    for (int part = 0; part < key_parts; ++part) {
        const int channel = (part * warp_size) + lane;
        const float key = qk[1][channel];
#pragma unroll
        for (int pair = 0; pair < values_per_warp; ++pair) {
            held[pair][part] *= gates.x;
            prediction[pair] = fmaf(held[pair][part], key, prediction[pair]);
        }
    }
    for (int offset = warp_size / 2; offset != 0; offset /= 2) {
#pragma unroll
        for (auto& value : prediction) {
            value += __shfl_xor_sync(0xffffffffU, value, offset);
        }
    }
#pragma unroll
    for (int pair = 0; pair < values_per_warp; ++pair) {
        result.update[pair] = gates.y * (target[pair] - prediction[pair]);
    }
#pragma unroll
    for (int part = 0; part < key_parts; ++part) {
        const int channel = (part * warp_size) + lane;
        const float key = qk[1][channel];
        const float query = qk[0][channel];
#pragma unroll
        for (int pair = 0; pair < values_per_warp; ++pair) {
            held[pair][part] = fmaf(result.update[pair], key, held[pair][part]);
            result.observable[pair] = fmaf(held[pair][part], query, result.observable[pair]);
        }
    }
    for (int offset = warp_size / 2; offset != 0; offset /= 2) {
#pragma unroll
        for (auto& value : result.observable) {
            value += __shfl_xor_sync(0xffffffffU, value, offset);
        }
    }
    return result;
}

template <typename S, int Rows>
__device__ void solve_drafts(const Lane& job, Cursor cursor, const DraftTile<Rows>& draft,
                             int feature, float (&held)[values_per_warp][key_parts]) {
    const int lane = static_cast<int>(threadIdx.x % warp_size);
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row >= job.rows)
            break;
        const int append = slot(cursor, cursor.committed + row);
        float target[values_per_warp];
#pragma unroll
        for (int pair = 0; pair < values_per_warp; ++pair) {
            const int val = feature + (pair * value_warps);
            target[pair] = __bfloat162float(value_row<S>(job, row)[val]);
        }
        const auto result = solve_row(held, draft.qk[row], lane, draft.gates[row], target);
        if (lane == 0) {
#pragma unroll
            for (int pair = 0; pair < values_per_warp; ++pair) {
                const int val = feature + (pair * value_warps);
                job.state.updates[append * S::value_width + val] = result.update[pair];
                job.observable[row * S::value_width + val] =
                    __float2bfloat16(result.observable[pair]);
            }
        }
    }
}

template <typename S, int Rows, typename Jobs>
__global__ void recurrent(const __grid_constant__ Jobs jobs, int max_rows, TextGdnWeights weights) {
    const auto& job = get_lane(jobs);
    const auto state = job.state;
    if (!valid_lane(job, max_rows) || !valid_state(state))
        return;
    const auto cursor = *state.cursor;
    if (!idle(cursor))
        return;
    const int head = static_cast<int>(blockIdx.x / tiles);
    const int tile = static_cast<int>(blockIdx.x % tiles);
    const int lane = static_cast<int>(threadIdx.x % warp_size);
    const int value = static_cast<int>(threadIdx.x / warp_size);
    const int feature = (head * head_dim) + (tile * tile_values);
    auto* anchor = state.checkpoint.recurrent + (feature + value) * head_dim;
    __shared__ HistoryTile history;
    __shared__ DraftTile<Rows> draft;
    float held[values_per_warp][key_parts];
    load_history_weights<S>(state, cursor, head, history, cursor.committed);
    load_history_keys<S>(state, cursor, head, history, cursor.committed);
    load_history_updates<S>(state, cursor, feature, history, cursor.committed);
    load_draft_gates<S>(job, cursor, weights, draft, head, tile == 0);
    load_draft_qk<S>(job, cursor, head / S::heads_per_qk * head_dim, draft);
    read_held(held, anchor, lane);
    __syncthreads();
    int first = 0;
    if (cursor.committed >= window) {
        replay_committed(held, value, history, 0, window);
        write_held(held, anchor, lane);
        first = window;
    }
    replay_committed(held, value, history, first, cursor.committed);
    solve_drafts<S, Rows>(job, cursor, draft, feature + value, held);
}

template <typename S, typename Jobs>
__global__ void finish(const __grid_constant__ Jobs jobs, int max_rows, const __nv_bfloat16* norm) {
    const auto& job = get_lane(jobs);
    if (!valid_lane(job, max_rows) || !valid_state(job.state))
        return;
    const int lane = static_cast<int>(threadIdx.x % warp_size);
    const int group = static_cast<int>((blockIdx.x * finish_warps) + (threadIdx.x / warp_size));
    if (group >= job.rows * S::value_heads)
        return;
    const int base = group * head_dim;
    float elements[key_parts];
    float squares = 0.0F;
#pragma unroll
    for (int part = 0; part < key_parts; ++part) {
        const int feature = (part * warp_size) + lane;
        elements[part] = __bfloat162float(job.observable[base + feature]);
        squares += warp_sum(elements[part] * elements[part]);
    }
    const float inverse_rms = rsqrtf((squares / head_dim) + epsilon);
#pragma unroll
    for (int part = 0; part < key_parts; ++part) {
        const int feature = (part * warp_size) + lane;
        const float gate = __bfloat162float(job.z[base + feature]);
        job.output[base + feature] =
            __float2bfloat16(elements[part] * inverse_rms * __bfloat162float(norm[feature]) *
                             (gate * sigmoid(gate)));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        auto cursor = *job.state.cursor;
        if (!idle(cursor))
            return;
        if (cursor.committed >= window) {
            cursor.first = slot(cursor, window);
            cursor.committed -= window;
        }
        if (job.commit_all) {
            cursor.committed += job.rows;
        } else {
            cursor.pending = job.rows;
        }
        *job.state.cursor = cursor;
    }
}

struct FixedAcceptance {
    int rows;
};

__device__ int accepted_count(FixedAcceptance accepted, [[maybe_unused]] int index) {
    return accepted.rows;
}
__device__ int accepted_count(const int* accepted, int index) {
    return accepted[index];
}

__device__ Cursor* get_cursor(Cursor* cursors, int index) {
    return cursors + index;
}

__device__ Cursor* get_cursor(const Lane* jobs, int index) {
    return jobs[index].state.cursor;
}

template <typename Jobs, typename Count>
__global__ void accept(Jobs jobs, Count accepted, int count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= count)
        return;
    auto* cursor = get_cursor(jobs, index);
    if (!cursor)
        return;
    const int rows = accepted_count(accepted, index);
    const auto current = *cursor;
    if (rows < 0 || rows > current.pending)
        return;
    cursor->committed = current.committed + rows;
    cursor->pending = 0;
}

__device__ int materialized_count(Cursor cursor, int boundary, bool inplace) {
    const int count = boundary < 0 ? cursor.committed : boundary;
    if (count > cursor.committed || count > max_committed ||
        (inplace && ((cursor.pending != 0) || count != cursor.committed)))
        return -1;
    return count;
}

template <typename S>
__global__ void materialize_recurrent(State state, TextGdnState destination, int boundary,
                                      bool inplace) {
    const auto cursor = *state.cursor;
    const int count = materialized_count(cursor, boundary, inplace);
    if (count < 0 || (count == 0 && inplace))
        return;
    const int head = static_cast<int>(blockIdx.x / tiles);
    const int tile = static_cast<int>(blockIdx.x % tiles);
    const int lane = static_cast<int>(threadIdx.x % warp_size);
    const int value = static_cast<int>(threadIdx.x / warp_size);
    const int feature = (head * head_dim) + (tile * tile_values);
    const int offset = (feature + value) * head_dim;
    __shared__ HistoryTile history;
    float held[values_per_warp][key_parts];
    load_history_weights<S>(state, cursor, head, history, count);
    load_history_keys<S>(state, cursor, head, history, count);
    read_held(held, state.checkpoint.recurrent + offset, lane);
    load_history_updates<S>(state, cursor, feature, history, count);
    __syncthreads();
    replay_committed(held, value, history, 0, count);
    write_held(held, destination.recurrent + offset, lane);
}

template <typename S>
__global__ void materialize_conv(State state, TextGdnState destination, int boundary,
                                 bool inplace) {
    const auto cursor = *state.cursor;
    const int count = materialized_count(cursor, boundary, inplace);
    if (count < 0 || (count == 0 && inplace))
        return;
    const int channel = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    __nv_bfloat16 samples[taps];
#pragma unroll
    for (int tap = 0; tap < taps; ++tap) {
        samples[tap] = raw_sample<S>(state, cursor, count - taps + tap, channel);
    }
#pragma unroll
    for (int tap = 0; tap < taps; ++tap) {
        destination.history[(channel * taps) + tap] = samples[tap];
    }
}

__global__ void reset_materialized(Cursor* cursor, int boundary) {
    if (materialized_count(*cursor, boundary, true) >= 0)
        *cursor = {};
}

template <typename S, int Rows, typename Jobs>
cudaError_t launch_kernels(Jobs jobs, int count, int max_rows, TextGdnWeights weights,
                           cudaStream_t stream) {
    const int conv_blocks = (S::qk_width + S::value_width) / head_dim;
    convolve<S, Rows><<<dim3(conv_blocks, 1, count), 2 * head_dim, 0, stream>>>(
        jobs, max_rows, weights.convolution);
    auto status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return status;
    recurrent<S, Rows>
        <<<dim3(S::value_heads * tiles, 1, count), value_warps * warp_size, 0, stream>>>(
            jobs, max_rows, weights);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return status;
    const int finish_blocks = max_rows * S::value_heads / finish_warps;
    finish<S><<<dim3(finish_blocks, 1, count), finish_warps * warp_size, 0, stream>>>(
        jobs, max_rows, weights.norm);
    return cudaPeekAtLastError();
}

template <typename Jobs>
cudaError_t launch_for_shape(bool rank_local, Jobs jobs, int count, int max_rows,
                             TextGdnWeights weights, cudaStream_t stream) {
    if (!weights.convolution || !weights.a_log || !weights.dt_bias || !weights.norm) {
        return cudaErrorInvalidValue;
    }
    if (max_rows == 1) {
        return rank_local ? launch_kernels<GdnLocal, 1>(jobs, count, max_rows, weights, stream)
                          : launch_kernels<GdnFull, 1>(jobs, count, max_rows, weights, stream);
    }
    return rank_local ? launch_kernels<GdnLocal, drafts>(jobs, count, max_rows, weights, stream)
                      : launch_kernels<GdnFull, drafts>(jobs, count, max_rows, weights, stream);
}

template <typename Jobs, typename Count>
cudaError_t enqueue_accept(Jobs jobs, Count accepted, int count, cudaStream_t stream) {
    constexpr int threads = 128;
    accept<<<1 + ((count - 1) / threads), threads, 0, stream>>>(jobs, accepted, count);
    return cudaPeekAtLastError();
}

template <typename S>
cudaError_t enqueue_materialize(State state, TextGdnState destination, int boundary, bool inplace,
                                cudaStream_t stream) {
    materialize_recurrent<S><<<S::value_heads * tiles, value_warps * warp_size, 0, stream>>>(
        state, destination, boundary, inplace);
    auto status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return status;
    materialize_conv<S>
        <<<S::qkv_width / head_dim, head_dim, 0, stream>>>(state, destination, boundary, inplace);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess || !inplace)
        return status;
    reset_materialized<<<1, 1, 0, stream>>>(state.cursor, boundary);
    return cudaPeekAtLastError();
}

}

cudaError_t launch(bool rank_local, Lane lane, TextGdnWeights weights, cudaStream_t stream,
                   const State* device_state) {
    if (!valid_lane(lane, drafts) || ((device_state == nullptr) && !valid_state(lane.state))) {
        return cudaErrorInvalidValue;
    }
    return launch_for_shape(rank_local, SingleLane{lane, device_state}, 1, lane.rows, weights,
                            stream);
}

cudaError_t launch(bool rank_local, const Lanes& lanes, int lane_count, int max_rows,
                   TextGdnWeights weights, cudaStream_t stream) {
    if (!valid_rows(lane_count, max_packed_lanes) || !valid_rows(max_rows, drafts)) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < lane_count; ++index) {
        if (!valid_lane(lanes.lanes[index], max_rows) || !valid_state(lanes.lanes[index].state)) {
            return cudaErrorInvalidValue;
        }
    }
    return launch_for_shape(rank_local, lanes, lane_count, max_rows, weights, stream);
}

cudaError_t launch(bool rank_local, const Lane* device_lanes, int lane_count, int max_rows,
                   TextGdnWeights weights, cudaStream_t stream) {
    if ((device_lanes == nullptr) || !valid_rows(lane_count, max_device_lanes) ||
        !valid_rows(max_rows, drafts)) {
        return cudaErrorInvalidValue;
    }
    return launch_for_shape(rank_local, device_lanes, lane_count, max_rows, weights, stream);
}

cudaError_t commit(Cursor* cursors, int layers, int accepted, cudaStream_t stream) {
    if ((cursors == nullptr) || layers <= 0 || accepted < 0 || accepted > drafts) {
        return cudaErrorInvalidValue;
    }
    return enqueue_accept(cursors, FixedAcceptance{accepted}, layers, stream);
}

cudaError_t commit(const Lane* device_lanes, const int* device_accepted, int lane_count,
                   cudaStream_t stream) {
    if ((device_lanes == nullptr) || (device_accepted == nullptr) ||
        !valid_rows(lane_count, max_device_lanes)) {
        return cudaErrorInvalidValue;
    }
    return enqueue_accept(device_lanes, device_accepted, lane_count, stream);
}

cudaError_t materialize(bool rank_local, State state, TextGdnState destination, cudaStream_t stream,
                        int boundary) {
    if (!valid_state(state) || (destination.history == nullptr) ||
        (destination.recurrent == nullptr) || boundary < -1 || boundary > max_committed) {
        return cudaErrorInvalidValue;
    }
    const bool inplace = state.checkpoint.recurrent == destination.recurrent;
    if (inplace != (state.checkpoint.history == destination.history))
        return cudaErrorInvalidValue;
    return rank_local ? enqueue_materialize<GdnLocal>(state, destination, boundary, inplace, stream)
                      : enqueue_materialize<GdnFull>(state, destination, boundary, inplace, stream);
}

}
