#include "sglang/runtime/cache/lmcache_group.h"
#include "sglang/runtime/cache/prefix_cache_layout.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace sglang::decoder {
namespace {

int common_kv(const LmCacheClient& client, const std::vector<const PrefixCacheKeys*>& keys,
              const std::vector<LmCachePrefixDetail>& details, bool draft) {
    const auto coverage = [draft](const LmCachePrefixDetail& detail) {
        return draft ? detail.draft : detail.gqa;
    };
    const auto [lowest, highest] = std::minmax_element(
        details.begin(), details.end(),
        [&](const auto& left, const auto& right) { return coverage(left) < coverage(right); });
    const int limit = coverage(*lowest);
    const int tail = limit % prefix_gqa_chunk_tokens;
    if (tail == 0 || limit == coverage(*highest))
        return limit;

    const int base = limit - tail;
    const auto family = draft ? LmCacheFamily::draft : LmCacheFamily::gqa;
    std::vector<bool> common(tail, true);
    std::vector<LmCacheKey> candidates(tail);
    for (const auto* rank : keys) {
        for (int index = 0; index < tail; ++index) {
            candidates[index] = {family, rank->key(limit - index + static_cast<int>(draft))};
        }
        const auto present = client.lookup(candidates);
        for (int index = 0; index < tail; ++index)
            common[index] = common[index] && present[index];
    }
    const auto first = std::find(common.begin(), common.end(), true);
    return first == common.end() ? base : limit - static_cast<int>(first - common.begin());
}

bool valid_draft_geometry(const PrefixDeviceLayout& layout, int world, int rank) {
    return world != 2 || layout.draft.end == 0 ||
           layout.draft.kv_heads == tp2_prefix_cache_layout(rank).kv_heads;
}

bool same_rank_extent(const PrefixDeviceLayout& layout, const PrefixDeviceLayout& lead) {
    return layout.gqa.begin == lead.gqa.begin && layout.gqa.end == lead.gqa.end &&
           layout.draft.begin == lead.draft.begin && layout.draft.end == lead.draft.end &&
           layout.gdn.end == lead.gdn.end;
}

const LmCacheEndpointCoverage* find_endpoint(const LmCachePrefixDetail& detail, int end) {
    const auto& endpoints = detail.endpoints;
    const auto found =
        std::lower_bound(endpoints.begin(), endpoints.end(), end,
                         [](const auto& item, int value) { return item.end > value; });
    return found != endpoints.end() && found->end == end ? &*found : nullptr;
}

struct RankTransferResult {
    LmCacheStatus status = LmCacheStatus::declined;
    LmCacheReceipt receipt;
    std::exception_ptr error;
};

template <typename Run>
std::vector<RankTransferResult> run_ranks(std::size_t world, const Run& run) {
    std::vector<RankTransferResult> results(world);
    const auto run_rank = [&](std::size_t rank) noexcept {
        auto& result = results[rank];
        try {
            result.status = run(rank, &result.receipt);
        } catch (...) {
            result.error = std::current_exception();
        }
    };
    std::vector<std::thread> workers(world - 1);
    for (std::size_t rank = 1; rank < world; ++rank) {
        try {
            workers[rank - 1] = std::thread(run_rank, rank);
        } catch (const std::system_error&) {
            run_rank(rank);
        }
    }
    run_rank(0);
    for (std::size_t rank = 1; rank < world; ++rank) {
        if (workers[rank - 1].joinable()) {
            workers[rank - 1].join();
        }
    }
    for (const auto& result : results) {
        if (result.error != nullptr)
            std::rethrow_exception(result.error);
    }
    return results;
}

LmCacheStatus finish_ranks(const std::vector<RankTransferResult>& results, LmCacheReceipt* receipt,
                           std::vector<LmCacheReceipt>* per_rank) {
    LmCacheStatus status = LmCacheStatus::complete;
    LmCacheReceipt total;
    for (std::size_t rank = 0; rank < results.size(); ++rank) {
        const auto& result = results[rank];
        if (result.status != LmCacheStatus::complete && status != LmCacheStatus::miss) {
            status = result.status;
        }
        if (per_rank != nullptr)
            (*per_rank)[rank] = result.receipt;
        for (std::size_t family = 0; family < total.bytes.size(); ++family) {
            total.bytes[family] += result.receipt.bytes[family];
        }
        total.gqa_written_tokens =
            std::max(total.gqa_written_tokens, result.receipt.gqa_written_tokens);
    }
    if (receipt != nullptr)
        *receipt = total;
    return status;
}
}

LmCacheGroup::LmCacheGroup(LmCacheClient client) : client_(std::move(client)), rank_stores_(1) {
}

LmCacheGroup::LmCacheGroup(LmCacheConfig config, int world_size)
    : client_(std::move(config)), world_size_(world_size) {
    if (world_size_ <= 0) {
        throw std::invalid_argument("Invalid LMCache world size");
    }
    rank_stores_ = std::vector<RankStoreCounters>(static_cast<std::size_t>(world_size_));
}

bool LmCacheGroup::ready() const {
    return client_.ready();
}

bool LmCacheGroup::ready(const std::vector<std::array<std::uint8_t, 16>>& devices) const {
    return client_.ready(devices);
}

PrefixCacheRead LmCacheGroup::lookup_prefix(const std::vector<const PrefixCacheKeys*>& keys,
                                            int max_restore, bool draft) const {
    const auto world = static_cast<std::size_t>(world_size());
    if (keys.size() != world) {
        throw std::invalid_argument("LMCache rank key count mismatch");
    }
    for (int rank = 0; rank < world_size_; ++rank) {
        if (keys[rank] == nullptr) {
            throw std::invalid_argument("Missing LMCache rank keys");
        }
        if (!keys[rank]->matches_rank(world_size_, rank)) {
            return {};
        }
    }
    std::vector<LmCachePrefixDetail> details(world);
    for (std::size_t rank = 0; rank < world; ++rank) {
        details[rank] = client_.lookup_prefix(*keys[rank], draft);
    }
    PrefixCacheRead read;
    read.matched_gqa = common_kv(client_, keys, details, false);
    if (draft)
        read.matched_draft = common_kv(client_, keys, details, true);
    read.usable_gqa = read.matched_gqa;
    const int maximum = std::min(max_restore, read.usable_gqa);
    for (const auto& candidate : details.front().endpoints) {
        bool draft_ready = candidate.draft;
        const auto contains = [&](const LmCachePrefixDetail& detail) {
            const auto* endpoint = find_endpoint(detail, candidate.end);
            draft_ready = draft_ready && endpoint != nullptr && endpoint->draft;
            return endpoint != nullptr;
        };
        if (!std::all_of(details.begin() + 1, details.end(), contains))
            continue;
        if (read.matched_gdn == 0)
            read.matched_gdn = candidate.end;
        read.matched_gdn_endpoints.push_back(candidate.end);
        if (draft_ready)
            read.draft_ready_gdn_endpoints.push_back(candidate.end);
        if (read.usable_gdn == 0 && candidate.end <= maximum && (!draft || draft_ready)) {
            read.usable_gdn = candidate.end;
            read.usable_draft = draft ? candidate.end - 1 : 0;
        }
    }
    std::reverse(read.matched_gdn_endpoints.begin(), read.matched_gdn_endpoints.end());
    std::reverse(read.draft_ready_gdn_endpoints.begin(), read.draft_ready_gdn_endpoints.end());
    return read;
}

std::string LmCacheGroup::prepare_store(int rank, const PrefixCacheKeys& keys,
                                        const PrefixDeviceTransfer& transfer) const {
    if (rank < 0 || rank >= world_size_ || !keys.matches_rank(world_size_, rank) ||
        !valid_draft_geometry(transfer.layout, world_size_, rank)) {
        throw std::invalid_argument("Invalid LMCache rank store");
    }
    return client_.prepare_store(keys, transfer);
}

LmCacheStatus LmCacheGroup::store(int rank, const PrefixCacheKeys& keys,
                                  const PrefixDeviceTransfer& transfer,
                                  LmCacheReceipt* receipt) const {
    return store_prepared(prepare_store(rank, keys, transfer), receipt);
}

LmCacheStatus LmCacheGroup::store_prepared(std::string_view metadata,
                                           LmCacheReceipt* receipt) const {
    return client_.store_prepared(metadata, receipt);
}

LmCacheStatus LmCacheGroup::store(const std::vector<const PrefixCacheKeys*>& keys,
                                  const std::vector<const PrefixDeviceTransfer*>& transfers,
                                  LmCacheReceipt* receipt, std::vector<LmCacheReceipt>* per_rank,
                                  int stop_fd) const {
    const auto start = std::chrono::steady_clock::now();
    ++pending_;
    const auto finish = [&] {
        const auto elapsed = std::chrono::steady_clock::now() - start;
        write_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        --pending_;
    };
    LmCacheReceipt aggregate;
    auto* sink = receipt != nullptr ? receipt : &aggregate;
    try {
        const auto status = transfer_ranked(keys, transfers, sink, per_rank, stop_fd, true);
        if (status == LmCacheStatus::complete) {
            ++stores_;
            gqa_written_ += sink->gqa_written_tokens;
            for (std::size_t family = 0; family < sink->bytes.size(); ++family) {
                written_bytes_[family] += sink->bytes[family];
            }
        } else {
            ++declined_;
        }
        finish();
        return status;
    } catch (...) {
        ++errors_;
        finish();
        throw;
    }
}

LmCacheGroupStoreMetrics LmCacheGroup::store_metrics() const {
    LmCacheGroupStoreMetrics result;
    result.pending = pending_;
    result.stores = stores_;
    result.declined = declined_;
    result.errors = errors_;
    result.elapsed_ns = write_ns_;
    result.gqa_written_tokens = gqa_written_;
    for (std::size_t family = 0; family < written_bytes_.size(); ++family) {
        result.bytes[family] = written_bytes_[family];
    }
    result.rank_stores.reserve(rank_stores_.size());
    result.rank_bytes.reserve(rank_stores_.size());
    for (const auto& rank : rank_stores_) {
        result.rank_stores.push_back(rank.stores);
        result.rank_bytes.push_back({rank.bytes[0], rank.bytes[1], rank.bytes[2]});
    }
    return result;
}

LmCacheStatus LmCacheGroup::retrieve(const std::vector<const PrefixCacheKeys*>& keys,
                                     const std::vector<const PrefixDeviceTransfer*>& transfers,
                                     LmCacheReceipt* receipt,
                                     std::vector<LmCacheReceipt>* per_rank,
                                     int stop_fd) const {
    return transfer_ranked(keys, transfers, receipt, per_rank, stop_fd, false);
}

LmCacheStatus
LmCacheGroup::transfer_ranked(const std::vector<const PrefixCacheKeys*>& keys,
                              const std::vector<const PrefixDeviceTransfer*>& transfers,
                              LmCacheReceipt* receipt, std::vector<LmCacheReceipt>* per_rank,
                              int stop_fd, bool store) const {
    const auto world = static_cast<std::size_t>(world_size());
    if (keys.size() != world || transfers.size() != world) {
        throw std::invalid_argument("LMCache rank transfer count mismatch");
    }
    if (transfers.front() == nullptr)
        throw std::invalid_argument("Missing LMCache lead transfer");
    const auto& layout = transfers.front()->layout;
    if (receipt != nullptr)
        *receipt = {};
    if (per_rank != nullptr)
        per_rank->assign(world, LmCacheReceipt{});
    const auto incompatible = store ? LmCacheStatus::declined : LmCacheStatus::miss;
    for (int rank = 0; rank < world_size_; ++rank) {
        if (keys[rank] == nullptr)
            throw std::invalid_argument("Missing LMCache rank keys");
        if (!keys[rank]->matches_rank(world_size_, rank)) {
            return incompatible;
        }
        if (transfers[rank] == nullptr || !same_rank_extent(transfers[rank]->layout, layout) ||
            !valid_draft_geometry(transfers[rank]->layout, world_size_, rank)) {
            throw std::invalid_argument("LMCache transfer extent or draft geometry mismatch");
        }
    }
    const auto transfer_rank = [&](std::size_t rank, LmCacheReceipt* sink) {
        return transfer_one(rank, *keys[rank], *transfers[rank], *sink, stop_fd, store);
    };
    if (world == 1) {
        LmCacheReceipt local;
        const auto status = transfer_rank(0, &local);
        if (receipt != nullptr)
            *receipt = local;
        if (per_rank != nullptr)
            per_rank->front() = local;
        return status;
    }
    return finish_ranks(run_ranks(world, transfer_rank), receipt, per_rank);
}

LmCacheStatus LmCacheGroup::transfer_one(std::size_t rank, const PrefixCacheKeys& keys,
                                         const PrefixDeviceTransfer& transfer,
                                         LmCacheReceipt& receipt, int stop_fd, bool store) const {
    if (!store)
        return client_.retrieve(keys, transfer, &receipt, stop_fd);
    const auto status =
        client_.store_prepared(client_.prepare_store(keys, transfer), &receipt, stop_fd);
    if (status == LmCacheStatus::complete) {
        auto& totals = rank_stores_[rank];
        ++totals.stores;
        for (std::size_t family = 0; family < totals.bytes.size(); ++family) {
            totals.bytes[family] += receipt.bytes[family];
        }
    }
    return status;
}

void LmCacheGroup::reset() const {
    client_.reset();
}

int LmCacheGroup::world_size() const {
    return world_size_;
}
}
