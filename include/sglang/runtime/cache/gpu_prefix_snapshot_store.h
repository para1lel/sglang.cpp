#pragma once

#include "sglang/decoder_internal.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

namespace sglang::decoder {

class GpuPrefixSnapshotStore {
    static constexpr int max_request_pages = text_attention_max_tokens / mxfp8_page_size;

    struct KvAllocation {
        KvAllocation(int device, int layers, int pages);

        Allocations memory;
        Mxfp8KvCache kv;
    };

    struct Target : KvAllocation {
        Target(int device, int pages);

        TextGdnState gdn;
        Bf16* hidden;
        Bf16* residual;
    };

    struct Snapshot {
        std::unique_ptr<Target> target;
        std::unique_ptr<KvAllocation> draft;
    };

public:
    struct PageMapping {
        std::array<std::int32_t, max_request_pages> physical;
        int count;
    };

    GpuPrefixSnapshotStore(int device, int capacity, bool with_draft);

    bool supports_draft() const { return with_draft_; }

    std::size_t device_bytes() const;
    std::size_t draft_device_bytes() const;

    std::optional<Snapshot> prepare(int slot, int target_pages, int draft_pages) const;
    void release(int slot);
    void clear();

    // Validate the live mapping before the caller invalidates an overwrite victim.
    static PageMapping read_request_pages(const Request& request, int pages, cudaStream_t stream);
    void store(int slot, Snapshot prepared, const Request& request, const PageMapping& mapping,
               int tokens, cudaStream_t stream, const MtpDraftRequest* draft, int draft_rows);
    void restore(int slot, Request& request, int tokens, bool restore_workspace,
                 cudaStream_t stream, MtpDraftRequest* draft, int draft_rows) const;

private:
    const int device_;
    const bool with_draft_;
    std::array<Snapshot, decoder_batch_prefix_max_entries> slots_{};
};

}
