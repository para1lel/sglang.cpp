#include "sglang/runtime/cache/lmcache_group.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace sglang::decoder {
namespace {

constexpr int draft_rank = 0;

int common_gqa(const LmCacheClient& client, const std::vector<const PrefixCacheKeys*>& keys,
               const std::vector<LmCachePrefixDetail>& details) {
    auto [lowest, highest] = std::minmax_element(
        details.begin(), details.end(),
        [](const auto& left, const auto& right) { return left.gqa < right.gqa; });
    const int limit = lowest->gqa;
    const int tail = limit % prefix_gqa_chunk_tokens;
    if (tail == 0 || limit == highest->gqa)
        return limit;

    const int base = limit - tail;
    std::vector<bool> common(tail, true);
    std::vector<LmCacheKey> candidates(tail);
    for (const auto* rank : keys) {
        for (int index = 0; index < tail; ++index) {
            candidates[index] = {LmCacheFamily::gqa, rank->key(limit - index)};
        }
        const auto present = client.lookup(candidates);
        for (int index = 0; index < tail; ++index)
            common[index] = common[index] && present[index];
    }
    const auto first = std::find(common.begin(), common.end(), true);
    return first == common.end() ? base : limit - static_cast<int>(first - common.begin());
}
}

LmCacheGroup::LmCacheGroup(LmCacheClient client) : client_(std::move(client)) {
}

LmCacheGroup::LmCacheGroup(LmCacheConfig config, int world_size)
    : client_(std::move(config)), world_size_(world_size) {
    if (world_size_ <= 0) {
        throw std::invalid_argument("Invalid LMCache world size");
    }
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
        details[rank] = client_.lookup_prefix(*keys[rank], draft && rank == draft_rank);
    }
    PrefixCacheRead read;
    read.matched_gqa = common_gqa(client_, keys, details);
    read.usable_gqa = read.matched_gqa;
    const int maximum = std::min(max_restore, read.usable_gqa);
    for (const auto& candidate : details[draft_rank].endpoints) {
        const auto contains = [&](const LmCachePrefixDetail& detail) {
            const auto& endpoints = detail.endpoints;
            const auto found =
                std::lower_bound(endpoints.begin(), endpoints.end(), candidate.end,
                                 [](const auto& item, int end) { return item.end > end; });
            return found != endpoints.end() && found->end == candidate.end;
        };
        if (!std::all_of(details.begin() + 1, details.end(), contains))
            continue;
        if (read.matched_gdn == 0)
            read.matched_gdn = candidate.end;
        read.matched_gdn_endpoints.push_back(candidate.end);
        if (candidate.draft)
            read.draft_ready_gdn_endpoints.push_back(candidate.end);
        if (read.usable_gdn == 0 && candidate.end <= maximum && (!draft || candidate.draft)) {
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
        (rank != draft_rank && transfer.layout.draft.end != 0)) {
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

LmCacheStatus LmCacheGroup::retrieve(const std::vector<const PrefixCacheKeys*>& keys,
                                     const std::vector<const PrefixDeviceTransfer*>& transfers,
                                     LmCacheReceipt* receipt,
                                     std::vector<LmCacheReceipt>* per_rank,
                                     int stop_fd) const {
    const auto world = static_cast<std::size_t>(world_size());
    if (keys.size() != world || transfers.size() != world) {
        throw std::invalid_argument("LMCache rank transfer count mismatch");
    }
    for (int rank = 0; rank < world_size_; ++rank) {
        if (!keys[rank] || !transfers[rank]) {
            throw std::invalid_argument("Missing LMCache rank transfer");
        }
    }
    if (receipt) *receipt = {};
    if (per_rank) per_rank->assign(world, LmCacheReceipt{});
    for (int rank = 0; rank < world_size_; ++rank) {
        if (!keys[rank]->matches_rank(world_size_, rank)) {
            return LmCacheStatus::miss;
        }
        if (rank != draft_rank && transfers[rank]->layout.draft.end != 0) {
            throw std::invalid_argument("Invalid LMCache rank draft transfer");
        }
    }
    if (world == 1) {
        auto* sink = per_rank ? &(*per_rank)[0] : receipt;
        const auto status = client_.retrieve(*keys.front(), *transfers.front(), sink, stop_fd);
        if (receipt && per_rank) *receipt = (*per_rank)[0];
        return status;
    }

    std::vector<LmCacheStatus> statuses(world, LmCacheStatus::declined);
    std::vector<LmCacheReceipt> receipts(world);
    std::vector<std::exception_ptr> errors(world);
    const auto restore_rank = [&](std::size_t rank) noexcept {
        try {
            statuses[rank] =
                client_.retrieve(*keys[rank], *transfers[rank], &receipts[rank], stop_fd);
        } catch (...) {
            errors[rank] = std::current_exception();
        }
    };
    std::vector<std::thread> workers(world - 1);
    for (std::size_t rank = 1; rank < world; ++rank) {
        try {
            workers[rank - 1] = std::thread(restore_rank, rank);
        } catch (const std::system_error&) {
        }
    }
    restore_rank(0);
    for (std::size_t rank = 1; rank < world; ++rank) {
        if (workers[rank - 1].joinable()) {
            workers[rank - 1].join();
        } else {
            restore_rank(rank);
        }
    }
    for (const auto& error : errors) {
        if (error) std::rethrow_exception(error);
    }
    if (per_rank) *per_rank = receipts;
    LmCacheStatus result = LmCacheStatus::complete;
    for (const auto status : statuses) {
        if (status == LmCacheStatus::complete) continue;
        result = status;
        if (status == LmCacheStatus::miss) break;
    }
    if (receipt) {
        for (const auto& rank_receipt : receipts) {
            for (std::size_t family = 0; family < receipt->bytes.size(); ++family) {
                receipt->bytes[family] += rank_receipt.bytes[family];
            }
            receipt->gqa_written_tokens =
                std::max(receipt->gqa_written_tokens, rank_receipt.gqa_written_tokens);
        }
    }
    return result;
}

void LmCacheGroup::reset() const {
    client_.reset();
}

int LmCacheGroup::world_size() const {
    return world_size_;
}
}
