#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/runtime/cache/gpu_prefix_snapshot_store.h"
#include "sglang/runtime/cache/prefix_index.h"

#include <string_view>

namespace sglang::decoder {

class PrefixCache {
public:
    PrefixCache(int device, int capacity, bool with_draft = false);

    static int checkpoint(int tokens);
    int restore(Request& request, const std::vector<int>& prompt, std::string_view salt,
                cudaStream_t stream, MtpDraftRequest* draft = nullptr,
                int max_tokens = decoder_batch_prefix_max_tokens);
    void store(const Request& request, const std::vector<int>& prompt, std::string_view salt,
               int tokens, cudaStream_t stream, const MtpDraftRequest* draft = nullptr);
    void count_prefill(int tokens, int draft_rows = 0);
    DecoderPrefixCacheStats stats() const;
    void reset();

private:
    GpuPrefixSnapshotStore snapshots_;
    PrefixIndex index_;
    DecoderPrefixCacheStats counters_;
};

}
