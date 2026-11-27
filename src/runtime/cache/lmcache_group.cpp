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

LmCacheConfig cache_config(LmCacheConfig config, int world_size) {
    if (world_size <= 0) {
        throw std::invalid_argument("Invalid LMCache world size");
    }
    config.native_v4 |= world_size > 1;
    return config;
}
}

LmCacheGroup::LmCacheGroup(LmCacheClient client) : client_(std::move(client)) {
}

LmCacheGroup::LmCacheGroup(LmCacheConfig config, int world_size)
    : client_(cache_config(std::move(config), world_size)), world_size_(world_size) {
}

bool LmCacheGroup::ready() const {
    return client_.ready();
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
    if (world == 1) {
        const auto match = client_.lookup_prefix(*keys.front(), max_restore, draft);
        return {match.gqa, match.matched_gdn, match.gqa, match.gdn, match.draft};
    }
    std::vector<LmCachePrefixDetail> details(world);
    for (std::size_t rank = 0; rank < world; ++rank) {
        details[rank] =
            client_.lookup_prefix_detail(*keys[rank], max_restore, draft && rank == draft_rank);
    }
    PrefixCacheRead read;
    read.matched_gqa = details.front().gqa;
    read.matched_gdn = details.front().matched_gdn;
    for (std::size_t rank = 1; rank < world; ++rank) {
        read.matched_gqa = std::min(read.matched_gqa, details[rank].gqa);
        read.matched_gdn = std::min(read.matched_gdn, details[rank].matched_gdn);
    }
    int endpoint = 0;
    for (const auto& candidate : details[draft_rank].endpoints) {
        if (draft && !candidate.draft) continue;
        bool common = true;
        for (std::size_t rank = 0; rank < world; ++rank) {
            if (static_cast<int>(rank) == draft_rank) continue;
            const auto& endpoints = details[rank].endpoints;
            common = std::any_of(endpoints.begin(), endpoints.end(),
                                 [end = candidate.end](const LmCacheEndpointCoverage& item) {
                                     return item.end == end;
                                 });
            if (!common) break;
        }
        if (common) {
            endpoint = candidate.end;
            break;   
        }
    }
    read.usable_gqa = read.matched_gqa;
    read.usable_gdn = endpoint;
    read.usable_draft = draft && endpoint ? endpoint - 1 : 0;
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
