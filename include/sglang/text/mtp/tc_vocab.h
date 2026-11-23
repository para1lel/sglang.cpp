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
inline constexpr int tc_vocab_narrow_rows = 5;

struct TcVocabRow {
    const __nv_bfloat16* input;
    TextLogitCandidate* winner;
};

std::shared_ptr<const TcVocabWeights> prepare_tc_vocab(const __nv_bfloat16* original,
                                                       const std::uint8_t* quantized,
                                                       const float* maxima, Allocations& memory);
std::shared_ptr<const TcVocabScratch> prepare_tc_vocab_scratch(Allocations& memory, bool wide);
bool try_tc_vocab(const TcVocabWeights& weights, const TcVocabScratch& scratch,
                  const TcVocabRow* batch, int count, cudaStream_t stream);

}
