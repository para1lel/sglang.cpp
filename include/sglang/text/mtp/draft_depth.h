#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/text/decode_proj.h"
#include <optional>

namespace sglang::decoder {

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
};

std::optional<cudaError_t> prepare_draft_depth(DraftDepthScratch& scratch,
                                             const std::uint8_t* weights, cudaStream_t stream);
cudaError_t project_draft_depth(const DraftDepthScratch& scratch, const float* maxima,
                                DraftDepthRows rows, int count, cudaStream_t stream);
}
