#include "sglang/runtime/execution/decoder_execution_policy.h"

#include "sglang/decoder_cuda.cuh"

namespace sglang::decoder {

DecoderExecutionPolicy DecoderExecutionPolicy::for_capabilities(bool ordered_attention,
                                                                const Limits& limits) {
    return {
        ordered_attention,
        !ordered_attention,
        limits.tensor_prefill_rows,
        limits.fallback_prefill_rows,
        limits.ordered_graph_lanes,
        limits.target_split_end,
    };
}

bool DecoderExecutionPolicy::regular_decode_graph(bool text_decode, bool contains_mtp,
                                                  int count) const {
    return text_decode && !contains_mtp && (!ordered_attention || count == ordered_graph_lanes);
}

int DecoderExecutionPolicy::prefill_limit(bool media, bool draft, bool prefix_cache) const {
    return media || (draft && (!ordered_attention || prefix_cache)) ? fallback_prefill_rows
                                                                    : tensor_prefill_rows;
}

int DecoderExecutionPolicy::label_scoring_limit(bool media) const {
    return media ? fallback_prefill_rows : tensor_prefill_rows;
}

bool DecoderExecutionPolicy::split_decode(int prompt_count, int limit) const {
    return target_split_end != 0 && limit > target_split_first_position &&
           prompt_count < target_split_end;
}

}
