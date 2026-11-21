#pragma once

#include "sglang/decoder.h"
#include "sglang/text_ops.h"

#include <cstddef>

namespace sglang::decoder {

struct GeneratedLogprobReadback {
    DecoderLogprob selected;
    DecoderLogprob alternatives[decoder_max_generated_top_logprobs];
};

std::size_t generated_logprob_workspace_bytes();
std::shared_ptr<const DecoderTokenLogprobs>
collect_generated_logprobs(const GeneratedLogprobReadback& row, int top);
cudaError_t enqueue_generated_logprobs(const __nv_bfloat16* logits, int vocabulary,
                                       const TextLogitCandidate* winner, int top, void* workspace,
                                       GeneratedLogprobReadback* host, cudaStream_t stream);

}
