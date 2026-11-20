#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/decoder_internal.h"

#include <string_view>

namespace sglang::decoder {

class PrefixCache {
public:
    PrefixCache(int device, int capacity, bool with_draft = false);

    static int checkpoint(int tokens);
    int restore(Request& request, const std::vector<int>& prompt, std::string_view salt,
                cudaStream_t stream, MtpDraftRequest* draft = nullptr);
    void store(const Request& request, const std::vector<int>& prompt, std::string_view salt,
               int tokens, cudaStream_t stream, const MtpDraftRequest* draft = nullptr);
    void count_prefill(int tokens, int draft_rows = 0);
    DecoderPrefixCacheStats stats() const;
    void reset();

private:
    struct Entry {
        Entry(Allocations& memory, bool with_draft);

        std::array<int, decoder_batch_prefix_max_tokens> ids;
        std::array<char, decoder_prefix_max_salt_bytes> salt;
        std::size_t salt_size = 0;
        int tokens = 0;
        Mxfp8KvCache kv;
        TextGdnState gdn;
        Bf16* hidden;
        Bf16* residual;
        Mxfp8KvCache draft;
        int draft_rows = 0;
    };

    int find(const std::vector<int>& prompt, std::string_view salt, int limit,
             bool require_draft = false) const;
    void touch(int index);

    Allocations memory_;
    std::vector<std::unique_ptr<Entry>> entries_;
    DecoderPrefixCacheStats counters_;
};

}
