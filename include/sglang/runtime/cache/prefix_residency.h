#pragma once

#include "sglang/mxfp8_gqa.h"
#include "sglang/runtime/cache/prefix_payload.h"

#include <array>
#include <memory>
#include <vector>

namespace sglang {
struct DecoderPrefixCacheStats;
}

namespace sglang::decoder {

struct PrefixCacheLayout;
class LmCacheGroup;
class PrefixPrefillSnapshot;
class PrefixSlab;

using PrefixRankKeys = std::vector<std::shared_ptr<PrefixCacheKeys>>;
using PrefixPage = std::shared_ptr<const int>;
using PrefixPages = std::vector<PrefixPage>;
using PrefixSnapshots = std::array<std::shared_ptr<PrefixPrefillSnapshot>, 2>;

struct ResidentPrefixRead {
    PrefixPages gqa;
    PrefixPages draft;
    PrefixSnapshots gdn;
    int gqa_tokens = 0;
    int draft_rows = 0;
    int gdn_tokens = 0;
    int matched_gdn = 0;
    int gqa_private_from = 0;
    int draft_private_from = 0;
};

struct ResidentPrefixPageRange {
    const PrefixPage* pages = nullptr;
    int begin = 0;
    int end = 0;
};

class PrefixSlab {
  public:
    struct CacheLocation {
        int rank = 0;
        int layer = 0;
    };

    struct TailCopy {
        int destination;
        int source;
        int rows;
    };

    PrefixSlab(const std::vector<int>& devices, int slots, int layers, int heads);
    ~PrefixSlab();
    PrefixSlab(const PrefixSlab&) = delete;
    PrefixSlab& operator=(const PrefixSlab&) = delete;
    PrefixSlab(PrefixSlab&&) = delete;
    PrefixSlab& operator=(PrefixSlab&&) = delete;

    int capacity() const;
    int available() const;
    int layers() const;
    int devices() const;
    std::size_t bytes_per_page() const;
    Mxfp8KvCache cache(CacheLocation location) const;
    PrefixPage acquire();
    void copy_tail(int rank, TailCopy tail, cudaStream_t stream) const;
    void quarantine() noexcept;

  private:
    struct State;
    std::shared_ptr<State> state_;
};

class PrefixResidency {
  public:
    PrefixResidency(const std::vector<int>& devices, bool enabled, bool draft, int lanes,
                    int max_pages, int max_checkpoints);
    ~PrefixResidency();
    PrefixResidency(const PrefixResidency&) = delete;
    PrefixResidency& operator=(const PrefixResidency&) = delete;
    PrefixResidency(PrefixResidency&&) = delete;
    PrefixResidency& operator=(PrefixResidency&&) = delete;

    const std::shared_ptr<PrefixSlab>& target() const;
    const std::shared_ptr<PrefixSlab>& draft() const;
    ResidentPrefixRead lookup(const PrefixRankKeys& keys, int max_restore, bool draft);
    PrefixPages reserve_pages(const ResidentPrefixRead& read, int capacity, bool draft);
    PrefixSnapshots lookup_checkpoint(const PrefixRankKeys& keys, int maximum, int& end);
    void commit_lookup(const ResidentPrefixRead& result);
    void notify_released();
    PrefixSnapshots reserve_checkpoint(bool* transfer_pending = nullptr);
    void wait_for_checkpoint();
    void publish_pages(const PrefixRankKeys& keys, const std::shared_ptr<const LmCacheGroup>& io,
                       ResidentPrefixPageRange range, bool draft, bool backed = false);
    void publish_checkpoint(const PrefixRankKeys& keys,
                            const std::shared_ptr<const LmCacheGroup>& io, int end,
                            const PrefixSnapshots& snapshots, bool backed = false);
    void stats(DecoderPrefixCacheStats& result) const;
    bool backup_pending() const;
    void reset();
    void quarantine() noexcept;

  private:
    struct State;
    std::unique_ptr<State> state_;
};

}