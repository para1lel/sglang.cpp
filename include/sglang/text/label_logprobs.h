#pragma once

#include "sglang/decoder_batch.h"

#include <cuda_bf16.h>

namespace sglang::decoder {

std::vector<double> label_logprobs(const std::vector<float>& logits,
                                   const std::vector<int>& labels);
std::vector<double> label_logprobs(const std::vector<__nv_bfloat16>& logits,
                                   const std::vector<int>& labels);

void collect_label_score(const std::vector<float>& logits, const std::vector<int>& labels,
                         DecoderBatchStep& step);
void collect_label_score(const std::vector<__nv_bfloat16>& logits, const std::vector<int>& labels,
                         DecoderBatchStep& step);
}
