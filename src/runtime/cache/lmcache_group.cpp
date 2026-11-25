#include "sglang/runtime/cache/lmcache_group.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>

namespace sglang::decoder {
namespace {

constexpr int draft_rank = 0;

}  // namespace

std::vector<LmCacheConfig> ranked_lmcache_configs(int world_size, int timeout_ms) {
    if (world_size <= 0) throw std::invalid_argument("Invalid LMCache world size");
    std::vector<LmCacheConfig> configs;
    configs.reserve(static_cast<std::size_t>(world_size));
    for (int rank = 0; rank < world_size; ++rank) {
        if (world_size == 1) {
            configs.push_back(LmCacheConfig{default_lmcache_endpoint(), "sglang", timeout_ms});
            continue;
        }
        configs.push_back(LmCacheConfig{rank_lmcache_endpoint(rank),
                                        "sglang-r" + std::to_string(rank), timeout_ms, true});
    }
    return configs;
}

std::vector<LmCacheConfig> ranked_lmcache_configs(const LmCacheConfig& base, int world_size) {
    if (world_size <= 0) throw std::invalid_argument("Invalid LMCache world size");
    std::vector<LmCacheConfig> configs;
    configs.reserve(static_cast<std::size_t>(world_size));
    for (int rank = 0; rank < world_size; ++rank) {
        if (world_size == 1) {
            configs.push_back(base);
            continue;
        }
        configs.push_back(LmCacheConfig{rank_lmcache_endpoint(base.endpoint, rank),
                                        base.name_space + "-r" + std::to_string(rank),
                                        base.timeout_ms, true});
    }
    return configs;
}

namespace {

std::vector<std::unique_ptr<LmCacheClient>> connect(std::vector<LmCacheConfig> configs) {
    if (configs.empty()) throw std::invalid_argument("Empty LMCache rank configuration");
    std::vector<std::unique_ptr<LmCacheClient>> clients;
    clients.reserve(configs.size());
    for (auto& config : configs) {
        clients.push_back(std::make_unique<LmCacheClient>(std::move(config)));
    }
    return clients;
}

}  // namespace

LmCacheGroup::LmCacheGroup(std::vector<LmCacheConfig> configs)
    : LmCacheGroup(connect(std::move(configs))) {}

LmCacheGroup::LmCacheGroup(std::vector<std::unique_ptr<LmCacheClient>> clients) {
    if (clients.empty()) throw std::invalid_argument("Empty LMCache rank configuration");
    for (std::size_t rank = 0; rank < clients.size(); ++rank) {
        if (!clients[rank]) throw std::invalid_argument("Missing LMCache rank client");
        for (std::size_t other = 0; other < rank; ++other) {
            if (clients[rank]->config().endpoint == clients[other]->config().endpoint) {
                throw std::invalid_argument("Duplicate LMCache rank endpoint");
            }
        }
    }
    clients_ = std::move(clients);
}


bool LmCacheGroup::ready() const {
    for (const auto& client : clients_) {
        if (!client->ready()) return false;
    }
    return true;
}

PrefixCacheRead LmCacheGroup::lookup_prefix(const std::vector<const PrefixCacheKeys*>& keys,
                                            int max_restore, bool draft) const {
    const auto world = static_cast<std::size_t>(world_size());
    if (keys.size() != world) {
        throw std::invalid_argument("LMCache rank key count mismatch");
    }
    for (const auto* rank_keys : keys) {
        if (!rank_keys) throw std::invalid_argument("Missing LMCache rank keys");
    }
    if (world == 1) {
        const auto match = clients_.front()->lookup_prefix(*keys.front(), max_restore, draft);
        return {match.gqa, match.matched_gdn, match.gqa, match.gdn, match.draft};
    }
    std::vector<LmCachePrefixDetail> details(world);
    for (std::size_t rank = 0; rank < world; ++rank) {
        details[rank] = clients_[rank]->lookup_prefix_detail(
            *keys[rank], max_restore, draft && static_cast<int>(rank) == draft_rank);
    }
    PrefixCacheRead read;
    read.matched_gqa = details.front().gqa;
    read.matched_gdn = details.front().matched_gdn;
    for (std::size_t rank = 1; rank < world; ++rank) {
        read.matched_gqa = std::min(read.matched_gqa, details[rank].gqa);
        read.matched_gdn = std::min(read.matched_gdn, details[rank].matched_gdn);
    }
    // Largest GDN endpoint present on every rank, draft-eligible on the draft
    // lane when a draft is requested. Endpoint intersection, not a min of
    // per-rank maxima: a rank with no record at the other's endpoint must not
    // silently truncate the restore.
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
            break;  // endpoints arrive in descending order
        }
    }
    read.usable_gqa = read.matched_gqa;
    read.usable_gdn = endpoint;
    read.usable_draft = draft && endpoint ? endpoint - 1 : 0;
    return read;
}

std::string LmCacheGroup::prepare_store(int rank, const PrefixCacheKeys& keys,
                                        const PrefixDeviceTransfer& transfer) const {
    return client(rank).prepare_store(keys, transfer);
}

LmCacheStatus LmCacheGroup::store(int rank, const PrefixCacheKeys& keys,
                                  const PrefixDeviceTransfer& transfer,
                                  LmCacheReceipt* receipt) const {
    return client(rank).store(keys, transfer, receipt);
}

LmCacheStatus LmCacheGroup::store_prepared(int rank, std::string_view metadata,
                                           LmCacheReceipt* receipt) const {
    return client(rank).store_prepared(metadata, receipt);
}

LmCacheStatus LmCacheGroup::retrieve(const std::vector<const PrefixCacheKeys*>& keys,
                                     const std::vector<const PrefixDeviceTransfer*>& transfers,
                                     LmCacheReceipt* receipt, int stop_fd) const {
    const auto world = static_cast<std::size_t>(world_size());
    if (keys.size() != world || transfers.size() != world) {
        throw std::invalid_argument("LMCache rank transfer count mismatch");
    }
    for (std::size_t rank = 0; rank < world; ++rank) {
        if (!keys[rank] || !transfers[rank]) {
            throw std::invalid_argument("Missing LMCache rank transfer");
        }
    }
    if (receipt) *receipt = {};
    if (world == 1) {
        LmCacheReceipt rank_receipt;
        auto* sink = receipt ? &rank_receipt : nullptr;
        const auto status =
            clients_.front()->retrieve(*keys.front(), *transfers.front(), sink, stop_fd);
        if (receipt) *receipt = rank_receipt;
        return status;
    }

    std::vector<LmCacheStatus> statuses(world, LmCacheStatus::declined);
    std::vector<LmCacheReceipt> receipts(world);
    std::vector<std::exception_ptr> errors(world);
    std::vector<std::thread> workers;
    workers.reserve(world);
    for (std::size_t rank = 0; rank < world; ++rank) {
        workers.emplace_back([this, rank, &keys, &transfers, &statuses, &receipts, &errors,
                              stop_fd]() noexcept {
            try {
                statuses[rank] = clients_[rank]->retrieve(*keys[rank], *transfers[rank],
                                                          &receipts[rank], stop_fd);
            } catch (...) {
                errors[rank] = std::current_exception();
            }
        });
    }
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) {
        if (error) std::rethrow_exception(error);
    }
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
            // The ranks restore the same tokens; count them once.
            receipt->gqa_written_tokens =
                std::max(receipt->gqa_written_tokens, rank_receipt.gqa_written_tokens);
        }
    }
    return result;
}

void LmCacheGroup::reset() const {
    for (const auto& client : clients_) client->reset();
}

}  // namespace sglang::decoder
