#pragma once

namespace sglang::decoder {

struct CudaBackendPolicy;

struct DecoderExecutionPolicy {
    struct Limits {
        int ordered_graph_lanes;
        int tensor_prefill_rows;
        int fallback_prefill_rows;
    };

    static DecoderExecutionPolicy for_backend(const CudaBackendPolicy& backend,
                                              const Limits& limits);

    bool regular_decode_graph(bool text_decode, bool contains_mtp, int count) const;
    int prefill_limit(bool media, bool draft, bool prefix_cache) const;
    int label_scoring_limit(bool media) const;
    bool split_decode(int prompt_count, int limit) const;

    bool ordered_attention;
    bool narrow_eager_decode;
    bool tensor_core_prefill;
    int tensor_prefill_rows;
    int fallback_prefill_rows;
    int ordered_graph_lanes;
    int target_split_end;
};

}
