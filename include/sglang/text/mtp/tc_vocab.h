#pragma once

#include "sglang/text_ops.h"

#include <cstdint>
#include <memory>

namespace sglang::decoder {

class Allocations;
struct TcVocabWeights;
struct TcVocabScratch;

inline constexpr int tc_vocab_rows = 20;
inline constexpr int tc_vocab_wide_rows = 2 * tc_vocab_rows;
inline constexpr int tc_vocab_narrow_rows = 5;

struct TcVocabRow {
    const __nv_bfloat16* input;
    TextLogitCandidate* winner;
};

struct TcVocabRmsRows {
    struct Row {
        const __nv_bfloat16* branch;
        const __nv_bfloat16* residual;
        __nv_bfloat16* normalized;
    };
    Row rows[tc_vocab_wide_rows];
};

cudaError_t pack_tc_vocab_rms(const TcVocabRmsRows& rows, const __nv_bfloat16* weight,
                             __nv_bfloat16* packed, cudaStream_t stream);

std::shared_ptr<const TcVocabWeights> prepare_tc_vocab(const __nv_bfloat16* original,
                                                       const std::uint8_t* quantized,
                                                       const float* maxima, Allocations& memory);
std::shared_ptr<const TcVocabScratch> prepare_tc_vocab_scratch(Allocations& memory, bool wide,
                                                               bool resident);
bool try_tc_vocab(const TcVocabWeights& weights, const TcVocabScratch& scratch,
                  const TcVocabRow* batch, int count, cudaStream_t stream);
bool try_tc_vocab_final(const TcVocabWeights& weights, const TcVocabScratch& scratch,
                        const TcVocabRow* batch, const TcVocabRmsRows& rms,
                        const __nv_bfloat16* gain, cudaStream_t stream);

}
