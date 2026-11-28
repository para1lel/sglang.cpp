#pragma once

namespace sglang::decoder {

struct CudaBackendPolicy;

struct DecoderExecutionPolicy {
    struct Limits {
        int tensor_prefill_rows;
        int fallback_prefill_rows;
    };

    static DecoderExecutionPolicy for_backend(const CudaBackendPolicy& backend,
                                              const Limits& limits);

    bool regular_decode_graph(int count) const;
    int prefill_limit() const;
    bool split_decode(int prompt_count, int limit) const;

    bool ordered_attention;
    bool narrow_eager_decode;
    bool tensor_core_prefill;
    int tensor_prefill_rows;
    int fallback_prefill_rows;
    int target_split_end;
};

}
