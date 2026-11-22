#pragma once

#include "sglang/decoder_batch.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sglang::decoder {

template <class Logit>
std::vector<double> label_logprobs(const std::vector<Logit>& logits,
                                   const std::vector<int>& labels) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto logit : logits) {
        const float value = static_cast<float>(logit);
        if (std::isnan(value) || value == std::numeric_limits<float>::infinity()) {
            throw DecoderSamplingError();
        }
        maximum = std::max(maximum, double(value));
    }
    if (!std::isfinite(maximum)) {
        throw DecoderSamplingError();
    }
    double mass = 0;
    for (const auto logit : logits) {
        mass += std::exp(static_cast<float>(logit) - maximum);
    }
    const double log_mass = std::log(mass);
    std::vector<double> output;
    output.reserve(labels.size());
    for (int label : labels) {
        output.push_back(std::max(-double(std::numeric_limits<float>::max()),
                                  (static_cast<float>(logits[label]) - maximum) - log_mass));
    }
    return output;
}

template <class Logit>
void collect_label_score(const std::vector<Logit>& logits, const std::vector<int>& labels,
                         DecoderBatchStep& step) {
    try {
        step.label_logprobs = std::make_shared<const std::vector<double>>(
            label_logprobs(logits, labels));
        step.finish_reason = DecoderFinishReason::length;
    } catch (const DecoderSamplingError&) {
        step.sampling_failed = true;
    }
}

}
