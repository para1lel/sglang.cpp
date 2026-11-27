#pragma once

#include "sglang/decoder_batch.h"

namespace sglang::decoder {

template <class Logit>
std::vector<double> label_logprobs(const std::vector<Logit>& logits,
                                  const std::vector<int>& labels);

template <class Logit>
void collect_label_score(const std::vector<Logit>& logits, const std::vector<int>& labels,
                         DecoderBatchStep& step);

}
