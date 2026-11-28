#include "sglang/runtime/cache/prefix_residency.h"

#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/backend/device_arena.h"

#include <mutex>
#include <numeric>
#include <stdexcept>

namespace sglang::decoder {

struct PrefixSlab::State {
    State() = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;
    struct Rank {
        std::shared_ptr<DeviceArena> arena;
        std::uint8_t* base = nullptr;
    };

    ~State() {
        for (const auto& rank : ranks) {
            if (rank.base != nullptr && !quarantined)
                rank.arena->release(rank.base);
        }
    }

    std::mutex mutex;
    std::vector<Rank> ranks;
    std::vector<int> free;
    int slots = 0;
    int layers = 0;
    int heads = 0;
    std::size_t payload = 0;
    std::size_t scales = 0;
    bool quarantined = false;
};

PrefixSlab::PrefixSlab(const std::vector<int>& devices, int slots, int layers, int heads)
    : state_(std::make_shared<State>()) {
    if (devices.empty() || devices.size() > 2 || slots < 1 || layers < 1 || heads < 1) {
        throw std::invalid_argument("Invalid resident prefix slab geometry");
    }
    state_->slots = slots;
    state_->layers = layers;
    state_->heads = heads;
    state_->payload = static_cast<std::size_t>(heads) * mxfp8_head_dim * mxfp8_page_size;
    state_->scales = static_cast<std::size_t>(heads) * mxfp8_groups * mxfp8_page_size;
    state_->free.resize(slots);
    std::iota(state_->free.rbegin(), state_->free.rend(), 0);
    state_->ranks.reserve(devices.size());
    for (const int device : devices) {
        const DeviceScope scope(device);
        auto& rank = state_->ranks.emplace_back();
        rank.arena = DeviceArena::current(device);
        rank.base = static_cast<std::uint8_t*>(
            rank.arena->allocate_cache(bytes_per_page() * static_cast<std::size_t>(slots)));
    }
}

PrefixSlab::~PrefixSlab() = default;

void PrefixSlab::quarantine() noexcept {
    const std::scoped_lock lock(state_->mutex);
    state_->quarantined = true;
    for (const auto& rank : state_->ranks) {
        rank.arena->quarantine();
    }
}

int PrefixSlab::capacity() const {
    return state_->slots;
}

int PrefixSlab::available() const {
    const std::scoped_lock lock(state_->mutex);
    return static_cast<int>(state_->free.size());
}

int PrefixSlab::layers() const {
    return state_->layers;
}

int PrefixSlab::devices() const {
    return static_cast<int>(state_->ranks.size());
}

std::size_t PrefixSlab::bytes_per_page() const {
    return std::size_t{2} * state_->layers * (state_->payload + state_->scales);
}

Mxfp8KvCache PrefixSlab::cache(CacheLocation location) const {
    const auto [rank, layer] = location;
    auto* base = state_->ranks.at(rank).base;
    const auto payload_pitch = state_->payload * state_->slots;
    const auto scales_pitch = state_->scales * state_->slots;
    const auto payload = payload_pitch * state_->layers;
    const auto scales = scales_pitch * state_->layers;
    auto* keys = base + (layer * payload_pitch);
    auto* key_scales = base + (2 * payload) + (layer * scales_pitch);
    return {keys, keys + payload, key_scales, key_scales + scales, state_->slots, state_->heads};
}

PrefixPage PrefixSlab::acquire() {
    int id = 0;
    {
        const std::scoped_lock lock(state_->mutex);
        if (state_->free.empty())
            return {};
        id = state_->free.back();
        state_->free.pop_back();
    }
    struct Lease {
        Lease(std::shared_ptr<State> state, int id) : state(std::move(state)), id(id) {
        }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&&) = delete;
        Lease& operator=(Lease&&) = delete;
        ~Lease() {
            const std::scoped_lock lock(state->mutex);
            state->free.push_back(id);
        }
        std::shared_ptr<State> state;
        int id;
    };
    try {
        const auto lease = std::make_shared<Lease>(state_, id);
        return {lease, &lease->id};
    } catch (...) {
        const std::scoped_lock lock(state_->mutex);
        state_->free.push_back(id);
        throw;
    }
}

void PrefixSlab::copy_tail(int rank, TailCopy tail, cudaStream_t stream) const {
    const int destination = tail.destination;
    const int source = tail.source;
    const int rows = tail.rows;
    if (rows <= 0 || rows > mxfp8_page_size || destination == source) {
        throw std::logic_error("Invalid resident MXFP8 tail copy");
    }
    for (int layer = 0; layer < state_->layers; ++layer) {
        const auto data = cache({rank, layer});
        const auto copy = [&](std::uint8_t* base, std::size_t stride) {
            copy_cuda_async(base + (destination * stride), base + (source * stride),
                            (stride / mxfp8_page_size) * rows, cudaMemcpyDeviceToDevice, stream,
                            "Copy shared MXFP8 tail into private page");
        };
        copy(data.key_payload, state_->payload);
        copy(data.value_payload, state_->payload);
        const auto copy_scales = [&](std::uint8_t* base) {
            auto* to = base + (destination * state_->scales);
            const auto* from = base + (source * state_->scales);
            constexpr auto pitch = (mxfp8_page_size / mxfp8_group_size) * mxfp8_groups;
            const int lanes = rows % mxfp8_group_size;
            const auto bytes = (rows / mxfp8_group_size) * mxfp8_groups;
            const auto transfer = [&](int offset, int width, int count) {
                if (width == 0 || count == 0)
                    return;
                check(cudaMemcpy2DAsync(to + offset, pitch, from + offset, pitch, width, count,
                                        cudaMemcpyDeviceToDevice, stream),
                      "Copy shared MXFP8 tail scales into private page");
            };
            if (lanes == 0) {
                transfer(0, bytes, state_->heads * mxfp8_group_size);
                return;
            }
            for (int head = 0; head < state_->heads; ++head) {
                const int offset = head * mxfp8_group_size * pitch;
                transfer(offset, bytes + mxfp8_groups, lanes);
                transfer(offset + (lanes * pitch), bytes, mxfp8_group_size - lanes);
            }
        };
        copy_scales(data.key_scales);
        copy_scales(data.value_scales);
    }
}

}