#include "sglang/runtime/cache/prefix_residency.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_prefix_cache.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/backend/device_arena.h"
#include "sglang/runtime/cache/lmcache_group.h"
#include "sglang/runtime/cache/prefix_cache_layout.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <utility>

namespace sglang::decoder {
namespace {

constexpr int ranks_max = 2;
constexpr int families = 3;
constexpr auto target_family = std::size_t{0};
constexpr auto state_family = std::size_t{1};
constexpr auto draft_family = std::size_t{2};
constexpr int default_checkpoints = 128;

int pages_for(int rows) {
    return (rows + mxfp8_page_size - 1) / mxfp8_page_size;
}

std::size_t checkpoint_stride(const PrefixCacheLayout& layout) {
    const auto parts = prefix_gdn_state_bytes(layout);
    const auto bytes = std::accumulate(parts.begin(), parts.end(), std::size_t{0});
    return (bytes + 255) / 256 * 256;
}

struct CheckpointSlab {
    struct Rank {
        std::shared_ptr<DeviceArena> arena;
        std::byte* base = nullptr;
        std::size_t stride = 0;
        PrefixCacheLayout layout;
    };
    CheckpointSlab(const std::vector<int>& devices, int count);
    ~CheckpointSlab();
    CheckpointSlab(const CheckpointSlab&) = delete;
    CheckpointSlab& operator=(const CheckpointSlab&) = delete;
    CheckpointSlab(CheckpointSlab&&) = delete;
    CheckpointSlab& operator=(CheckpointSlab&&) = delete;
    std::mutex mutex;
    std::array<Rank, ranks_max> ranks;
    std::vector<int> free;
    int world;
    bool quarantined = false;
};

CheckpointSlab::CheckpointSlab(const std::vector<int>& devices, int count)
    : free(count), world(static_cast<int>(devices.size())) {
    std::iota(free.rbegin(), free.rend(), 0);
    try {
        for (int rank = 0; rank < world; ++rank) {
            const DeviceScope selected(devices[rank]);
            auto& slot = ranks[rank];
            slot.layout = world == 1 ? tp1_prefix_cache_layout : tp2_prefix_cache_layout(rank);
            slot.stride = checkpoint_stride(slot.layout);
            slot.arena = DeviceArena::current(devices[rank]);
            slot.base = static_cast<std::byte*>(slot.arena->allocate_cache(slot.stride * count));
        }
    } catch (...) {
        for (const auto& rank : ranks) {
            if (rank.base != nullptr)
                rank.arena->release(rank.base);
        }
        throw;
    }
}

CheckpointSlab::~CheckpointSlab() {
    for (const auto& rank : ranks) {
        if (rank.base != nullptr && !quarantined)
            rank.arena->release(rank.base);
    }
}

struct CheckpointLease {
    CheckpointLease(std::shared_ptr<CheckpointSlab> owner, int index);
    ~CheckpointLease();
    CheckpointLease(const CheckpointLease&) = delete;
    CheckpointLease& operator=(const CheckpointLease&) = delete;
    CheckpointLease(CheckpointLease&&) = delete;
    CheckpointLease& operator=(CheckpointLease&&) = delete;
    std::shared_ptr<CheckpointSlab> slab;
    int slot;
};

CheckpointLease::CheckpointLease(std::shared_ptr<CheckpointSlab> owner, int index)
    : slab(std::move(owner)), slot(index) {
}

CheckpointLease::~CheckpointLease() {
    const std::scoped_lock lock(slab->mutex);
    slab->free.push_back(slot);
}

PrefixSnapshots take_checkpoint(const std::shared_ptr<CheckpointSlab>& slab,
                                const std::vector<int>& devices) {
    if (!slab)
        return {};
    std::shared_ptr<CheckpointLease> lease;
    {
        const std::scoped_lock lock(slab->mutex);
        if (slab->free.empty())
            return {};
        lease = std::make_shared<CheckpointLease>(slab, slab->free.back());
        slab->free.pop_back();
    }
    PrefixSnapshots result;
    for (int rank = 0; rank < slab->world; ++rank) {
        const DeviceScope selected(devices[rank]);
        const auto& storage = slab->ranks[rank];
        result[rank] = std::make_shared<PrefixPrefillSnapshot>(
            storage.layout, lease, storage.base + (lease->slot * storage.stride));
    }
    return result;
}

struct PoolLimits {
    int target = 0;
    int draft = 0;
    int checkpoints = 0;
};

struct PoolOptions {
    int lanes;
    int max_pages;
    int max_checkpoints;
};

PoolLimits pool_limits(const std::vector<int>& devices, bool enabled, bool draft,
                       PoolOptions opts) {
    std::array<std::size_t, ranks_max> bytes{};
    std::array<std::size_t, ranks_max> checkpoint{};
    int checkpoints = opts.max_checkpoints > 0 ? opts.max_checkpoints : default_checkpoints;
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        const auto layout = devices.size() == 1 ? tp1_prefix_cache_layout
                                                : tp2_prefix_cache_layout(static_cast<int>(rank));
        bytes[rank] = DeviceArena::current(devices[rank])->cache_available_bytes();
        checkpoint[rank] = checkpoint_stride(layout);
        const auto budget = opts.max_checkpoints != 0 ? bytes[rank] / 2 : bytes[rank] / 12;
        checkpoints =
            static_cast<int>(std::min<std::size_t>(checkpoints, budget / checkpoint[rank]));
    }
    if (!enabled)
        checkpoints = 0;
    const auto target_bytes = std::size_t{2} * full_attention_layers *
                              (mxfp8_payload_bytes_per_page + mxfp8_scale_bytes_per_page) /
                              devices.size();
    const auto draft_bytes =
        2 * (mxfp8_payload_bytes_per_page + mxfp8_scale_bytes_per_page) / devices.size();
    const auto page_cost = target_bytes + (draft ? draft_bytes : 0);
    const auto token_pages = decoder_batch_prefix_max_tokens / mxfp8_page_size;
    int pages = opts.max_pages > 0 ? opts.max_pages : token_pages * opts.lanes;
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        const auto available = bytes[rank] - (checkpoint[rank] * checkpoints);
        pages = static_cast<int>(std::min<std::size_t>(pages, available / page_cost));
    }
    if (pages < 1 || (enabled && checkpoints < 1)) {
        throw DecoderCapacityError("Device cache arena cannot hold resident prefix pools", false);
    }
    if (enabled && opts.max_checkpoints == 0) {
        const auto page_bytes = pages * page_cost;
        auto fit = static_cast<std::size_t>(std::numeric_limits<int>::max());
        for (std::size_t rank = 0; rank < devices.size(); ++rank) {
            const auto remaining = (bytes[rank] - page_bytes) / checkpoint[rank];
            fit = std::min(fit, remaining);
        }
        checkpoints = std::max(checkpoints, static_cast<int>(fit) - opts.lanes);
    }
    return {pages, draft ? pages : 0, checkpoints};
}

int ranked_key_tokens(const PrefixRankKeys& keys, int world) {
    if (keys.size() != static_cast<std::size_t>(world))
        return -1;
    int tokens = std::numeric_limits<int>::max();
    for (int rank = 0; rank < world; ++rank) {
        if (!keys[rank] || !keys[rank]->matches_rank(world, rank))
            return -1;
        tokens = std::min(tokens, keys[rank]->cacheable_tokens());
    }
    return tokens;
}

struct Entry {
    std::string key;
    PrefixRankKeys keys;
    std::shared_ptr<const LmCacheGroup> io;
    PrefixPages pages;
    PrefixSnapshots snapshots;
    int begin = 0;
    int end = 0;
    bool backed = false;
    bool attempted = false;
};

using EntryHandle = std::shared_ptr<Entry>;
using EntryQueue = std::list<EntryHandle>;

enum class Reclaim { reclaimed, backup_pending, exhausted };

bool matches_peers(const Entry& entry, const PrefixRankKeys& keys, int end) {
    for (std::size_t rank = 1; rank < keys.size(); ++rank) {
        if (!keys[rank]->same_prefix(*entry.keys[rank], end))
            return false;
    }
    return true;
}

struct Family {
    EntryQueue lru;
    std::unordered_map<std::string, EntryQueue::iterator> index;
    std::map<int, int, std::greater<>> endpoints;
    std::unordered_map<int, int> page_references;
    bool unpinned(const Entry& entry) const;
    const Entry* get(const PrefixRankKeys& keys, int end, bool touch = false);
    const Entry* deepest(const PrefixRankKeys& keys, int maximum);
    EntryHandle backup_candidate(bool checkpoint) const;
    void insert(EntryHandle entry, int key_end);
    void erase(EntryQueue::iterator position);
};

bool Family::unpinned(const Entry& entry) const {
    return std::all_of(entry.pages.begin(), entry.pages.end(),
                       [&](const PrefixPage& page) {
                           return page.use_count() == page_references.at(*page);
                       }) &&
           std::all_of(entry.snapshots.begin(), entry.snapshots.end(),
                       [](const auto& snapshot) { return !snapshot || snapshot.use_count() == 1; });
}

const Entry* Family::get(const PrefixRankKeys& keys, int end, bool touch) {
    const auto found = index.find(keys.front()->key(end));
    if (found == index.end() || !matches_peers(**found->second, keys, end))
        return nullptr;
    if (touch)
        lru.splice(lru.end(), lru, found->second);
    return found->second->get();
}

const Entry* Family::deepest(const PrefixRankKeys& keys, int maximum) {
    for (auto at = endpoints.lower_bound(maximum); at != endpoints.end(); ++at) {
        if (const auto* entry = get(keys, at->first); entry != nullptr)
            return entry;
    }
    return nullptr;
}

EntryHandle Family::backup_candidate(bool checkpoint) const {
    EntryHandle candidate;
    for (const auto& entry : lru) {
        const bool eligible = entry->io && !entry->attempted;
        if ((!checkpoint && !eligible) || !unpinned(*entry))
            continue;
        if (checkpoint && (entry->backed || entry->attempted))
            return {};
        if (eligible && !candidate) {
            if (!checkpoint)
                return entry;
            candidate = entry;
        }
    }
    return candidate;
}

void Family::insert(EntryHandle entry, int key_end) {
    if (const auto duplicate = index.find(entry->key); duplicate != index.end()) {
        auto& previous = **duplicate->second;
        if (!matches_peers(previous, entry->keys, key_end))
            return;
        previous.backed |= entry->backed;
        previous.attempted |= entry->backed;
        if (!entry->pages.empty())
            lru.splice(lru.end(), lru, duplicate->second);
        return;
    }
    const auto inserted = lru.insert(lru.end(), std::move(entry));
    const auto& item = **inserted;
    std::size_t referenced = 0;
    bool counted = false;
    try {
        index.emplace(item.key, inserted);
        ++endpoints[item.end];
        counted = true;
        for (const auto& page : item.pages) {
            ++page_references[*page];
            ++referenced;
        }
    } catch (...) {
        index.erase(item.key);
        if (counted) {
            const auto end = endpoints.find(item.end);
            if (--end->second == 0)
                endpoints.erase(end);
        }
        while (referenced != 0) {
            const auto page = page_references.find(*item.pages[--referenced]);
            if (--page->second == 0)
                page_references.erase(page);
        }
        lru.erase(inserted);
        throw;
    }
}

void Family::erase(EntryQueue::iterator position) {
    index.erase((*position)->key);
    if (const auto end = endpoints.find((*position)->end); --end->second == 0)
        endpoints.erase(end);
    for (const auto& page : (*position)->pages) {
        if (const auto ref = page_references.find(*page); --ref->second == 0) {
            page_references.erase(ref);
        }
    }
    lru.erase(position);
}

bool load_chunk(Family& family, const PrefixRankKeys& keys, PrefixPages& output,
                ResidentPrefixPageRange range, bool draft, int& coverage) {
    for (auto at = family.endpoints.lower_bound(range.end);
         at != family.endpoints.end() && at->first > range.begin; ++at) {
        const int end = at->first;
        const auto* entry = family.get(keys, end + static_cast<int>(draft), true);
        if (entry == nullptr || entry->begin != range.begin)
            continue;
        output.insert(output.end(), entry->pages.begin(), entry->pages.end());
        coverage = end;
        return end - range.begin == prefix_gqa_chunk_tokens;
    }
    return false;
}

void lookup_pages(Family& family, const PrefixRankKeys& keys, PrefixPages& output, int maximum,
                  bool draft, int& coverage) {
    for (int begin = 0; begin < maximum; begin += prefix_gqa_chunk_tokens) {
        const int end = std::min(maximum, begin + prefix_gqa_chunk_tokens);
        if (!load_chunk(family, keys, output, {nullptr, begin, end}, draft, coverage))
            break;
    }
}
}

struct PrefixResidency::State {
    State(const std::vector<int>& selected, bool enabled, bool has_draft, int lanes, int max_pages,
          int max_checkpoints);
    ~State();
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;

    void run();
    std::pair<std::size_t, EntryHandle> backup_candidate() const;
    bool backup_pending() const;
    bool should_back_up(std::size_t family) const;
    LmCacheStatus backup(std::size_t family, const Entry& entry) const;
    Reclaim reclaim(std::size_t family);
    bool evict(Family& family, bool backed);
    int free_slots(std::size_t family) const;

    const std::vector<int> devices;
    const PoolLimits limits;
    const std::shared_ptr<PrefixSlab> target;
    const std::shared_ptr<PrefixSlab> draft;
    const std::shared_ptr<CheckpointSlab> checkpoints;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::array<Family, families> resident;
    std::array<bool, families> pressure{};
    EntryHandle transferring;
    std::size_t transferring_family = families;
    DecoderPrefixCacheStats counters;
    bool stopping = false;
    bool resetting = false;
    std::thread worker;
};

PrefixResidency::State::State(const std::vector<int>& selected, bool enabled, bool has_draft,
                              int lanes, int max_pages, int max_checkpoints)
    : devices(selected),
      limits(pool_limits(selected, enabled, has_draft, {lanes, max_pages, max_checkpoints})),
      target(std::make_shared<PrefixSlab>(selected, limits.target, full_attention_layers,
                                          mxfp8_kv_heads / static_cast<int>(selected.size()))),
      draft(has_draft
                ? std::make_shared<PrefixSlab>(selected, limits.draft, 1,
                                               mxfp8_kv_heads / static_cast<int>(selected.size()))
                : nullptr),
      checkpoints(enabled ? std::make_shared<CheckpointSlab>(selected, limits.checkpoints)
                          : nullptr) {
    if (enabled)
        worker = std::thread(&State::run, this);
}

PrefixResidency::State::~State() {
    {
        const std::scoped_lock lock(mutex);
        stopping = true;
    }
    changed.notify_all();
    if (worker.joinable())
        worker.join();
}

int PrefixResidency::State::free_slots(std::size_t family) const {
    if (family == target_family)
        return target->available();
    if (family == draft_family)
        return draft ? draft->available() : 0;
    if (!checkpoints)
        return 0;
    const std::scoped_lock lock(checkpoints->mutex);
    return static_cast<int>(checkpoints->free.size());
}

bool PrefixResidency::State::should_back_up(std::size_t family) const {
    const std::array capacities{limits.target, limits.checkpoints, limits.draft};
    const int capacity = capacities[family];
    return pressure[family] || (capacity > 0 && free_slots(family) < std::max(1, capacity / 8));
}

std::pair<std::size_t, EntryHandle> PrefixResidency::State::backup_candidate() const {
    if (resetting)
        return {};
    const std::array order = pressure[state_family]
                                 ? std::array{state_family, target_family, draft_family}
                                 : std::array{target_family, state_family, draft_family};
    for (const auto family : order) {
        if (!should_back_up(family))
            continue;
        if (auto entry = resident[family].backup_candidate(family == state_family))
            return {family, std::move(entry)};
    }
    return {};
}

LmCacheStatus PrefixResidency::State::backup(std::size_t family, const Entry& entry) const {
    std::array<PrefixDeviceTransfer, ranks_max> views;
    std::vector<const PrefixCacheKeys*> ranked_keys;
    std::vector<const PrefixDeviceTransfer*> ranked_views;
    ranked_keys.reserve(devices.size());
    ranked_views.reserve(devices.size());
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        ranked_keys.push_back(entry.keys.at(rank).get());
        const DeviceScope selected(devices[rank]);
        const std::array<PrefixSlab*, families> slabs{target.get(), nullptr, draft.get()};
        const auto* slab = slabs[family];
        views[rank] = pooled_prefix_view(
            slab, static_cast<int>(rank), {entry.pages.data(), entry.begin, entry.end},
            family == draft_family, entry.snapshots[rank], family == state_family ? entry.end : 0,
            cudaStreamPerThread);
        ranked_views.push_back(&views[rank]);
    }
    return entry.io->store(ranked_keys, ranked_views);
}

void PrefixResidency::State::run() {
    std::unique_lock lock(mutex);
    while (!stopping) {
        auto [family, entry] =
            transferring ? std::make_pair(transferring_family, transferring) : backup_candidate();
        if (!entry) {
            changed.wait(lock);
            continue;
        }
        transferring = entry;
        transferring_family = family;
        entry->attempted = true;
        ++counters.backups_started;
        lock.unlock();
        bool complete = false;
        try {
            complete = backup(family, *entry) == LmCacheStatus::complete;
        } catch (const std::exception&) {
            complete = false;
        }
        lock.lock();
        entry->backed |= complete;
        if (complete)
            ++counters.backups_completed;
        else
            ++counters.backups_failed;
        transferring.reset();
        transferring_family = families;
        changed.notify_all();
    }
}

bool PrefixResidency::State::evict(Family& family, bool backed) {
    for (auto entry = family.lru.begin(); entry != family.lru.end(); ++entry) {
        const auto& victim = **entry;
        if (*entry == transferring || victim.backed != backed ||
            (!backed && victim.io && !victim.attempted) || !family.unpinned(victim)) {
            continue;
        }
        family.erase(entry);
        if (backed)
            ++counters.backed_evictions;
        else
            ++counters.unbacked_evictions;
        return true;
    }
    return false;
}

Reclaim PrefixResidency::State::reclaim(std::size_t family) {
    auto& entries = resident[family];
    if (evict(entries, true))
        return Reclaim::reclaimed;
    pressure[family] = true;
    changed.notify_all();
    if (transferring_family == family)
        return Reclaim::backup_pending;
    if (auto candidate = entries.backup_candidate(family == state_family)) {
        if (!transferring) {
            transferring = std::move(candidate);
            transferring_family = family;
            transferring->attempted = true;
        }
        return Reclaim::backup_pending;
    }
    return evict(entries, false) ? Reclaim::reclaimed : Reclaim::exhausted;
}

PrefixResidency::PrefixResidency(const std::vector<int>& devices, bool enabled, bool draft,
                                 int lanes, int max_pages, int max_checkpoints)
    : state_(std::make_unique<State>(devices, enabled, draft, lanes, max_pages, max_checkpoints)) {
}

PrefixResidency::~PrefixResidency() = default;

const std::shared_ptr<PrefixSlab>& PrefixResidency::target() const {
    return state_->target;
}

const std::shared_ptr<PrefixSlab>& PrefixResidency::draft() const {
    return state_->draft;
}

ResidentPrefixRead PrefixResidency::lookup(const PrefixRankKeys& keys, int max_restore,
                                           bool draft_wanted) {
    ResidentPrefixRead result;
    const int maximum = ranked_key_tokens(keys, static_cast<int>(state_->devices.size()));
    if (maximum <= 0)
        return result;
    const int max_checkpoint = std::min(maximum, max_restore);
    const std::scoped_lock lock(state_->mutex);
    auto& resident = state_->resident;
    lookup_pages(resident[target_family], keys, result.gqa, maximum, false, result.gqa_tokens);
    if (draft_wanted && maximum > 1) {
        lookup_pages(resident[draft_family], keys, result.draft, maximum - 1, true,
                     result.draft_rows);
    }
    result.gqa_private_from = result.gqa_tokens;
    result.draft_private_from = result.draft_rows;
    int usable = std::min(result.gqa_tokens, max_checkpoint);
    if (draft_wanted)
        usable = std::min(usable, result.draft_rows + 1);
    auto& checkpoints = resident[state_family];
    const auto* matched = checkpoints.deepest(keys, max_checkpoint);
    if (matched != nullptr) {
        result.matched_gdn = matched->end;
        const auto* hit = matched->end <= usable ? matched : checkpoints.deepest(keys, usable);
        if (hit != nullptr) {
            result.gdn = hit->snapshots;
            result.gdn_tokens = hit->end;
        }
    }
    return result;
}

PrefixSnapshots PrefixResidency::lookup_checkpoint(const PrefixRankKeys& keys, int maximum,
                                                   int& end) {
    end = 0;
    maximum = std::min(maximum, ranked_key_tokens(keys, static_cast<int>(state_->devices.size())));
    if (maximum <= 0)
        return {};
    const std::scoped_lock lock(state_->mutex);
    const auto* entry = state_->resident[state_family].deepest(keys, maximum);
    if (entry != nullptr) {
        end = entry->end;
        return entry->snapshots;
    }
    return {};
}

PrefixSnapshots PrefixResidency::find_checkpoint(const PrefixRankKeys& keys, int end) {
    if (end <= 0 || end > ranked_key_tokens(keys, static_cast<int>(state_->devices.size())))
        return {};
    const std::scoped_lock lock(state_->mutex);
    if (const auto* entry = state_->resident[state_family].get(keys, end); entry != nullptr)
        return entry->snapshots;
    return {};
}

void PrefixResidency::commit_lookup(const ResidentPrefixRead& result, const PrefixRankKeys& keys,
                                    int resident_gdn) {
    const std::scoped_lock lock(state_->mutex);
    if (resident_gdn > 0 &&
        resident_gdn <= ranked_key_tokens(keys, static_cast<int>(state_->devices.size()))) {
        state_->resident[state_family].get(keys, resident_gdn, true);
    }
    auto& stats = state_->counters;
    stats.gqa_resident_hits += result.gqa_tokens > 0 ? 1 : 0;
    stats.gqa_resident_tokens += result.gqa_tokens;
    stats.gdn_resident_hits += result.matched_gdn > 0 ? 1 : 0;
    stats.gdn_resident_tokens += result.matched_gdn;
}

PrefixPages PrefixResidency::reserve_pages(const ResidentPrefixRead& read, int capacity,
                                           bool draft_wanted) {
    const auto& slab = draft_wanted ? state_->draft : state_->target;
    const auto& hits = draft_wanted ? read.draft : read.gqa;
    const int hit_rows = draft_wanted ? read.draft_private_from : read.gqa_private_from;
    const int count = pages_for(capacity);
    if (!slab || count > slab->capacity()) {
        throw DecoderCapacityError("Request exceeds resident KV page capacity", false);
    }
    const int shared = std::min(hit_rows / mxfp8_page_size, count);
    PrefixPages result;
    result.reserve(count);
    result.insert(result.end(), hits.begin(), hits.begin() + shared);
    const auto family = draft_wanted ? draft_family : target_family;
    const std::scoped_lock lock(state_->mutex);
    while (static_cast<int>(result.size()) < count) {
        auto page = slab->acquire();
        if (page) {
            result.push_back(std::move(page));
        } else if (state_->reclaim(family) != Reclaim::reclaimed) {
            throw DecoderCapacityError("Resident KV pages are busy", true);
        }
    }
    state_->pressure[family] = false;
    return result;
}

void PrefixResidency::notify_released() {
    const std::scoped_lock lock(state_->mutex);
    state_->changed.notify_all();
}

PrefixSnapshots PrefixResidency::reserve_checkpoint(bool* transfer_pending) {
    if (transfer_pending != nullptr)
        *transfer_pending = false;
    if (!state_->checkpoints)
        return {};
    const std::scoped_lock lock(state_->mutex);
    for (;;) {
        auto snapshots = take_checkpoint(state_->checkpoints, state_->devices);
        if (snapshots.front()) {
            state_->pressure[state_family] = false;
            return snapshots;
        }
        const auto reclaimed = state_->reclaim(state_family);
        if (reclaimed != Reclaim::reclaimed) {
            if (transfer_pending != nullptr)
                *transfer_pending = reclaimed == Reclaim::backup_pending;
            return {};
        }
    }
}

void PrefixResidency::wait_for_checkpoint() {
    std::unique_lock lock(state_->mutex);
    state_->changed.wait_for(lock, std::chrono::milliseconds(1),
                             [&] { return state_->stopping || !state_->transferring; });
}

void PrefixResidency::publish_pages(const PrefixRankKeys& keys,
                                    const std::shared_ptr<const LmCacheGroup>& io,
                                    ResidentPrefixPageRange range, bool draft_wanted, bool backed) {
    const int key_end = range.end + static_cast<int>(draft_wanted);
    if (range.pages == nullptr || range.end <= range.begin ||
        key_end > ranked_key_tokens(keys, static_cast<int>(state_->devices.size())))
        return;
    const auto family = draft_wanted ? draft_family : target_family;
    auto key = keys.front()->key(key_end);
    if (key.empty())
        return;
    auto entry = std::make_shared<Entry>();
    entry->key = std::move(key);
    entry->keys = keys;
    entry->io = io;
    const auto count = pages_for(range.end) - (range.begin / mxfp8_page_size);
    entry->pages.assign(range.pages, range.pages + count);
    entry->begin = range.begin;
    entry->end = range.end;
    entry->backed = backed;
    entry->attempted = backed;
    {
        const std::scoped_lock lock(state_->mutex);
        state_->resident[family].insert(std::move(entry), key_end);
    }
    state_->changed.notify_one();
}

void PrefixResidency::publish_checkpoint(const PrefixRankKeys& keys,
                                         const std::shared_ptr<const LmCacheGroup>& io, int end,
                                         const PrefixSnapshots& snapshots, bool backed) {
    const auto world = state_->devices.size();
    if (end <= 0 || end > ranked_key_tokens(keys, static_cast<int>(world)) ||
        std::any_of(snapshots.begin(), snapshots.begin() + world,
                    [](const auto& snapshot) { return !snapshot; }))
        return;
    std::string key = keys.front()->key(end);
    auto entry = std::make_shared<Entry>();
    entry->key = std::move(key);
    entry->keys = keys;
    entry->io = io;
    entry->snapshots = snapshots;
    entry->end = end;
    entry->backed = backed;
    entry->attempted = backed;
    {
        const std::scoped_lock lock(state_->mutex);
        state_->resident[state_family].insert(std::move(entry), end);
    }
    state_->changed.notify_one();
}

bool PrefixResidency::State::backup_pending() const {
    return transferring != nullptr || backup_candidate().second != nullptr;
}

bool PrefixResidency::backup_pending() const {
    const std::scoped_lock lock(state_->mutex);
    return state_->backup_pending();
}

void PrefixResidency::stats(DecoderPrefixCacheStats& result) const {
    const std::scoped_lock lock(state_->mutex);
    const auto& stats = state_->counters;
    result.gqa_page_capacity = state_->limits.target;
    result.gdn_slot_capacity = state_->limits.checkpoints;
    result.entries = static_cast<int>(state_->resident[state_family].lru.size());
    result.pages = static_cast<int>(state_->resident[target_family].page_references.size());
    result.draft_pages = static_cast<int>(state_->resident[draft_family].page_references.size());
    result.device_bytes = result.pages * target()->bytes_per_page() * state_->devices.size();
    if (state_->checkpoints) {
        for (const auto& rank : state_->checkpoints->ranks) {
            result.device_bytes += result.entries * rank.stride;
        }
    }
    result.draft_device_bytes =
        draft() ? result.draft_pages * draft()->bytes_per_page() * draft()->devices() : 0;
    result.pending_backups = static_cast<int>(state_->backup_pending());
    result.gqa_resident_hits = stats.gqa_resident_hits;
    result.gqa_resident_tokens = stats.gqa_resident_tokens;
    result.gdn_resident_hits = stats.gdn_resident_hits;
    result.gdn_resident_tokens = stats.gdn_resident_tokens;
    result.backups_started = stats.backups_started;
    result.backups_completed = stats.backups_completed;
    result.backups_failed = stats.backups_failed;
    result.backed_evictions = stats.backed_evictions;
    result.unbacked_evictions = stats.unbacked_evictions;
    result.evictions += stats.backed_evictions + stats.unbacked_evictions;
    result.stores += stats.backups_completed;
}

void PrefixResidency::reset() {
    std::unique_lock lock(state_->mutex);
    state_->resetting = true;
    state_->changed.wait(lock, [&] { return !state_->transferring; });
    state_->resident = {};
    state_->pressure = {};
    state_->resetting = false;
}

void PrefixResidency::quarantine() noexcept {
    const std::scoped_lock lock(state_->mutex);
    state_->stopping = true;
    state_->target->quarantine();
    if (state_->draft)
        state_->draft->quarantine();
    if (const auto& checkpoints = state_->checkpoints) {
        checkpoints->quarantined = true;
        for (const auto& rank : checkpoints->ranks) {
            if (rank.arena != nullptr)
                rank.arena->quarantine();
        }
    }
    state_->changed.notify_all();
}

}