#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/runtime/model/qwen_layers.h"
#include "sglang/text/decode_proj.h"
#include <optional>

namespace sglang::decoder {

inline constexpr int draft_depth_winner_rows = 16;
inline constexpr int draft_depth_winner_tile_rows = 128;
inline constexpr std::size_t draft_depth_winner_bytes =
    draft_depth_winner_rows * (vocab_size / draft_depth_winner_tile_rows) * sizeof(unsigned);

struct DraftDepthRows {
    DraftInt8Projection row[decoder_batch_max_lanes];
};

struct DraftDepthScratch {
    CUtensorMap weights{};
    CUtensorMap activation16{};
    CUtensorMap activation32{};
    std::uint8_t* activation = nullptr;
    float* maxima = nullptr;
    bool ready = false;
    bool fused_winner = false;
};

std::optional<cudaError_t> prepare_draft_depth(DraftDepthScratch& scratch,
                                             const std::uint8_t* weights, bool tensor_parallel,
                                             cudaStream_t stream);
cudaError_t project_draft_depth(const DraftDepthScratch& scratch, const float* maxima,
                                DraftDepthRows rows, int count, cudaStream_t stream);
cudaError_t score_draft_depth16(const DraftDepthScratch& scratch, const float* maxima,
                                 DraftDepthRows rows, TextLogitCandidate* const* winners,
                                 cudaStream_t stream);
}
