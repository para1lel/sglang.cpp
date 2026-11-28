#pragma once

#include "sglang/runtime/cache/prefix_payload.h"
#include <cuda_runtime_api.h>
#include <vector>

namespace sglang {
struct TextGdnState;
}

namespace sglang::decoder {

struct Request;
struct Workspace;
class MtpDraftRequest;

class PrefixPrefillSnapshot {
  public:
    explicit PrefixPrefillSnapshot(const PrefixCacheLayout* layout);
    ~PrefixPrefillSnapshot();
    PrefixPrefillSnapshot(const PrefixPrefillSnapshot&) = delete;
    PrefixPrefillSnapshot& operator=(const PrefixPrefillSnapshot&) = delete;
    PrefixPrefillSnapshot(PrefixPrefillSnapshot&&) = delete;
    PrefixPrefillSnapshot& operator=(PrefixPrefillSnapshot&&) = delete;
    TextGdnState state(int bank) const;
    void capture_row(const void* hidden, const void* residual, cudaStream_t stream) const;
    void capture_state(TextGdnState state, cudaStream_t stream) const;
    std::byte* data() const;
    std::size_t bytes() const;

  private:
    int device_ = 0;
    std::byte* data_ = nullptr;
    std::array<std::size_t, 4> sizes_;
};

class PrefixPrefillPool {
  public:
    PrefixPrefillPool();
    std::size_t available(std::size_t capacity) const;
    void wait_available(std::size_t capacity, std::size_t count) const;
    std::shared_ptr<PrefixPrefillSnapshot> acquire(std::size_t capacity,
                                                   const PrefixCacheLayout* layout);

  private:
    struct Release;
    std::shared_ptr<Release> release_;
    std::vector<std::shared_ptr<PrefixPrefillSnapshot>> slots_;
};

PrefixDeviceTransfer restore_prefix_view(Request& request, const PrefixCacheRead& read,
                                         cudaStream_t initialized,
                                         MtpDraftRequest* draft = nullptr,
                                         const PrefixCacheLayout* layout = nullptr);
PrefixDeviceTransfer
export_prefix_view(const Request& request, int gqa_begin, int draft_begin, int end,
                   cudaStream_t source_ready, const MtpDraftRequest* draft = nullptr,
                   const Workspace* row = nullptr, bool terminal = false,
                   const PrefixCacheLayout* layout = nullptr,
                   const std::shared_ptr<PrefixPrefillSnapshot>& snapshot = nullptr);
int commit_restored_prefix(Request& request, const PrefixDeviceTransfer& view,
                           MtpDraftRequest* draft = nullptr);

}
