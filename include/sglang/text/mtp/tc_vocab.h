#pragma once

#include "sglang/text_ops.h"

#include <array>
#include <cstdint>
#include <memory>

namespace sglang::decoder {

class Allocations;
struct TcVocabWeights;
struct TcVocabScratch;

inline constexpr int tc_vocab_rows = 20;

struct TcVocabRow {
    const __nv_bfloat16* input;
    TextLogitCandidate* winner;
};

std::shared_ptr<const TcVocabWeights> prepare_tc_vocab(const __nv_bfloat16* original,
                                                       const std::uint8_t* quantized,
                                                       const float* maxima, Allocations& memory);
std::shared_ptr<const TcVocabScratch> prepare_tc_vocab_scratch(Allocations& memory);
bool try_tc_vocab(const TcVocabWeights& weights, const TcVocabScratch& scratch,
                  const std::array<TcVocabRow, tc_vocab_rows>& batch, cudaStream_t stream);

}
