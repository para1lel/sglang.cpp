#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/decoder_internal.h"

#include <string_view>

namespace sglang::decoder {

class PrefixCache {
public:
    PrefixCache(int device, int capacity);

    static int checkpoint(int tokens);
    int restore(Request& request, const std::vector<int>& prompt, std::string_view salt,
                cudaStream_t stream);
    void store(const Request& request, const std::vector<int>& prompt, std::string_view salt,
               int tokens, cudaStream_t stream);
    void count_prefill(int tokens);
    DecoderPrefixCacheStats stats() const;

private:
    struct Entry {
        explicit Entry(Allocations& memory);

        std::array<int, decoder_batch_prefix_max_tokens> ids;
        std::array<char, decoder_prefix_max_salt_bytes> salt;
        std::size_t salt_size = 0;
        int tokens = 0;
        Mxfp8KvCache kv;
        TextGdnState gdn;
        Bf16* hidden;
        Bf16* residual;
    };

    int find(const std::vector<int>& prompt, std::string_view salt, int limit) const;
    void touch(int index);

    Allocations memory_;
    std::vector<std::unique_ptr<Entry>> entries_;
    DecoderPrefixCacheStats counters_;
};

}
