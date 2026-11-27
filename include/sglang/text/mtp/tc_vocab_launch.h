#pragma once

#include "sglang/text/decode_proj_sm100.cuh"
#include "sglang/text/mtp/tc_vocab.h"

namespace sglang::decoder::tc_vocab {

using Bf16 = __nv_bfloat16;
inline constexpr int vocabulary = 248320;
inline constexpr int width = 5120;
inline constexpr int rows = tc_vocab_wide_rows;
inline constexpr int tile = 1024;
inline constexpr int tiles = (vocabulary + tile - 1) / tile;
inline constexpr int refine_items = 64;
inline constexpr int refine_tiles = 256;
inline constexpr int slots = refine_items * refine_tiles;
static_assert(width * 128 * 128 < (1LL << 31));

struct Stats {
    float scale;
    float approx_norm;
    float error_norm;
    float value_norm;
};

struct Winner {
    float score;
    int token;
};

struct Inputs {
    const Bf16* rows[tc_vocab::rows];
};

struct Outputs {
    TextLogitCandidate* rows[tc_vocab::rows];
};

inline constexpr int short_rows = tc_vocab_short_rows;
inline constexpr int short_columns = decode_proj::bf16_small_columns;
inline constexpr int short_tiles = vocabulary / decode_proj::tile_rows;
static_assert(short_rows > 8 && short_rows <= 16);

struct ShortCandidates {
    int tile_count;
    int row_count[short_rows];
    int tiles[short_tiles];
};

inline constexpr int score_columns = 32;
inline constexpr int wide_columns = 64;
inline constexpr int narrow_columns = 16;

cudaError_t configure_packet_scores(int count);
cudaError_t configure_short_screen();
cudaError_t configure_short_seeds();
cudaError_t configure_short_replay();
cudaError_t launch_pack_inputs(int count, const Inputs& inputs, Bf16* output, cudaStream_t stream);
cudaError_t launch_shared_stats(int count, const Bf16* original, const std::uint8_t* quantized,
                               const float* maxima, Stats* output, cudaStream_t stream);
cudaError_t launch_vocabulary_scores(const CUtensorMap& weight, const CUtensorMap& input,
                                    int* scores, int count, cudaStream_t stream);
cudaError_t launch_lower_bounds(int count, int* scores, const Stats* weight, const Stats* inputs,
                               float* lower, cudaStream_t stream);
cudaError_t launch_thresholds(int count, const float* lower, float* threshold, cudaStream_t stream);
cudaError_t launch_candidates(int count, const int* bounds, const float* threshold,
                              int* count_per_row, int* ids, cudaStream_t stream);
cudaError_t launch_refine(int count, const Bf16* weight, const Inputs& inputs,
                          const int* count_per_row, const int* ids, Winner* partial,
                          cudaStream_t stream);
cudaError_t launch_finish(int count, const Winner* partial, const Outputs& outputs,
                          cudaStream_t stream);
cudaError_t launch_screen_short(const CUtensorMap& weight, const CUtensorMap& input,
                               const Stats* weight_stats, const Stats* input_stats,
                               int* upper, float* lower, cudaStream_t stream);
cudaError_t launch_select_short(const float* lower, float* threshold, int* seeds,
                                cudaStream_t stream);
cudaError_t launch_improve_short(const CUtensorMap& weight, const CUtensorMap& input,
                                 const int* seeds, float* threshold, cudaStream_t stream);
cudaError_t launch_collect_short(const int* upper, const float* threshold,
                                 ShortCandidates* candidates, cudaStream_t stream);
cudaError_t launch_replay_short(const CUtensorMap& weight, const CUtensorMap& input,
                                const int* upper, const float* threshold,
                                const ShortCandidates* candidates, Winner* partial,
                                cudaStream_t stream);
cudaError_t launch_finish_short(const ShortCandidates* candidates, const Winner* partial,
                               const Outputs& outputs, cudaStream_t stream);

}
