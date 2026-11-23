#pragma once

#include "sglang/decoder_internal.h"

#include <array>
#include <cstdint>
#include <vector>

namespace sglang::decoder {

class GpuPrefixSnapshotStore {
    static constexpr int max_request_pages = text_attention_max_tokens / mxfp8_page_size;

public:
    struct PageMapping {
        std::array<std::int32_t, max_request_pages> physical;
        int count;
    };

    GpuPrefixSnapshotStore(int device, int capacity, bool with_draft);

    std::size_t device_bytes() const;
    std::size_t draft_device_bytes() const;

    // Validate the live mapping before the caller invalidates an overwrite victim.
    static PageMapping read_request_pages(const Request& request, int pages, cudaStream_t stream);
    void store(int slot, const Request& request, const PageMapping& mapping, cudaStream_t stream,
               const MtpDraftRequest* draft, int draft_rows);
    void restore(int slot, Request& request, int pages, bool restore_workspace, cudaStream_t stream,
                 MtpDraftRequest* draft, int draft_rows) const;

private:
    struct Snapshot {
        explicit Snapshot(Allocations& memory);

        Mxfp8KvCache kv;
        TextGdnState gdn;
        Bf16* hidden;
        Bf16* residual;
        Mxfp8KvCache draft{};
    };

    Allocations memory_;
    std::vector<Snapshot> slots_;
    std::size_t draft_device_bytes_ = 0;
};

}
