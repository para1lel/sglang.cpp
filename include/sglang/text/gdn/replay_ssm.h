#pragma once

#include "sglang/text_attention.h"

namespace sglang::gdn_replay_ssm {

inline constexpr int window = 16;
inline constexpr int capacity = window + (2 * text_gdn_max_verify_rows) - 1;
inline constexpr int max_device_lanes = 32;
inline constexpr int max_packed_lanes = 16;

struct Cursor {
    int first;
    int committed;
    int pending;
};

struct State {
    TextGdnState checkpoint;
    float* keys;
    float* updates;
    float* decay;
    __nv_bfloat16* raw;
    Cursor* cursor;
};

struct Lane {
    State state{};
    TextGdnReplayRows inputs{};
    const __nv_bfloat16* z = nullptr;
    __nv_bfloat16* observable = nullptr;
    __nv_bfloat16* output = nullptr;
    int rows = 0;
    bool commit_all = false;
};

struct Lanes {
    Lane lanes[max_packed_lanes];
};

cudaError_t launch(bool rank_local, Lane lane, TextGdnWeights weights, cudaStream_t stream,
                   const State* device_state = nullptr);
cudaError_t launch(bool rank_local, const Lanes& lanes, int lane_count, int max_rows,
                   TextGdnWeights weights, cudaStream_t stream);
cudaError_t launch(bool rank_local, const Lane* device_lanes, int lane_count, int max_rows,
                   TextGdnWeights weights, cudaStream_t stream);

cudaError_t commit(Cursor* cursors, int layers, int accepted, cudaStream_t stream);
cudaError_t commit(const Lane* device_lanes, const int* device_accepted, int lane_count,
                   cudaStream_t stream);

cudaError_t materialize(bool rank_local, State state, TextGdnState destination, cudaStream_t stream,
                        int boundary = -1);

}
