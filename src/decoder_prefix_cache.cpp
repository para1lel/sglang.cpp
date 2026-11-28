#include "sglang/decoder_prefix_cache.h"

#include "sglang/decoder_mtp.h"

#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/backend/device_arena.h"
#include "sglang/runtime/model/qwen_request.h"

#include "sglang/runtime/cache/prefix_cache_layout.h"

#include <algorithm>
#include <condition_variable>
#include <numeric>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace sglang::decoder {
namespace {

std::array<std::size_t, 4> gdn_state_bytes(const PrefixCacheLayout* layout) {
    return prefix_gdn_state_bytes(layout ? *layout : tp1_prefix_cache_layout);
}

cudaError_t consume_prefix_error(cudaError_t direct) noexcept {
    if (direct == cudaSuccess) return direct;
    const auto recorded = cudaGetLastError();
    return !CudaError::terminal(direct) && CudaError::terminal(recorded) ? recorded : direct;
}

void check_prefix_cuda(cudaError_t status, const char* operation) {
    check(consume_prefix_error(status), operation);
}

int prefix_device(const char* operation) {
    int device = 0;
    check(cudaGetDevice(&device), operation);
    return device;
}

class PrefixStateVersion {
  public:
    PrefixStateVersion(std::array<std::shared_ptr<const void>, 4> sources,
                       const std::array<std::size_t, 4>& bytes)
        : device_(prefix_device("Identify decoder prefix version device")),
          arena_(DeviceArena::current(device_)),
          producer_(std::make_unique<Producer>(std::move(sources))), bytes_(bytes),
          data_(static_cast<std::byte*>(arena_->allocate_cache(version_bytes()))) {
    }

    ~PrefixStateVersion() {
        if (!producer_) return;
        try {
            DeviceScope device(device_);
            const auto status = complete_producer();
            if (status == cudaSuccess) {
                arena_->release(data_);
            } else {
                report(status);
            }
        } catch (const CudaError& error) {
            if (producer_->started) quarantine_producer();
            report(error.status);
        }
    }

    std::byte* data() const { return data_; }

    void submit(const std::array<const void*, 4>& state, cudaStream_t stream,
                const std::shared_ptr<void>& event) {
        producer_->stream = stream;
        producer_->started = true;
        try {
            auto* data = data_;
            for (std::size_t index = 0; index < bytes_.size(); ++index) {
                check_prefix_cuda(cudaMemcpyAsync(data, state[index], bytes_[index],
                                                  cudaMemcpyDeviceToDevice, stream),
                                  "Version decoder prefix state");
                data += bytes_[index];
            }
            check_prefix_cuda(cudaEventRecord(static_cast<cudaEvent_t>(event.get()), stream),
                              "Record decoder prefix readiness");
            producer_->ready = event;
        } catch (const CudaError& error) {
            const auto drained = complete_producer();
            if (CudaError::terminal(drained) && !CudaError::terminal(error.status)) {
                check(drained, "Drain failed decoder prefix copy");
            }
            report(drained);
            throw;
        }
    }

private:
    std::size_t version_bytes() const {
        return bytes_[0] + bytes_[1] + bytes_[2] + bytes_[3];
    }

    struct Producer {
        explicit Producer(std::array<std::shared_ptr<const void>, 4> retained)
            : sources(std::move(retained)) {}
        std::array<std::shared_ptr<const void>, 4> sources;
        std::shared_ptr<void> ready;
        cudaStream_t stream = nullptr;
        bool started = false;
    };

    cudaError_t complete_producer() noexcept {
        if (!producer_->started) return cudaSuccess;
        const auto ready = static_cast<cudaEvent_t>(producer_->ready.get());
        auto status = consume_prefix_error(ready ? cudaEventQuery(ready)
                                                 : cudaStreamSynchronize(producer_->stream));
        if (ready && status == cudaErrorNotReady) {
            status = consume_prefix_error(cudaEventSynchronize(ready));
        }
        if (status == cudaSuccess) producer_->started = false;
        else quarantine_producer();
        return status;
    }

    void quarantine_producer() noexcept {
        (void)producer_.release();
    }

    static void report(cudaError_t status) noexcept {
        if (status != cudaSuccess) {
            std::fprintf(stderr, "sglang decoder: Release prefix state version: %s\n",
                         cudaGetErrorString(status));
        }
    }

    int device_ = 0;
    std::shared_ptr<DeviceArena> arena_;
    std::unique_ptr<Producer> producer_;
    std::array<std::size_t, 4> bytes_{};
    std::byte* data_;
};

class PrefixViewBuilder {
public:
    explicit PrefixViewBuilder(const PrefixCacheLayout* layout, bool store = false)
        : state_bytes_(gdn_state_bytes(layout)) {
        if (store) {
            owners_ = std::make_shared<Owners>();
            view_.store_owner = owners_;
        }
    }

    PrefixDeviceAddress allocation(const void* base, std::size_t bytes,
                                   const Allocations* memory = nullptr) {
        if (memory != nullptr)
            owners_->blocks.push_back(memory->retain_block(base));
        const int device = prefix_device("Identify decoder prefix allocation device");
        const auto exported = DeviceArena::current(device)->export_address(base, bytes);
        const auto existing = std::find(bases_.begin(), bases_.end(), exported.base);
        if (existing != bases_.end()) {
            return {static_cast<std::uint32_t>(existing - bases_.begin()), exported.offset};
        }
        cudaIpcMemHandle_t handle;
        check_prefix_cuda(cudaIpcGetMemHandle(&handle, exported.base),
                          "Export decoder prefix allocation handle");
        auto& allocations = view_.transfer.allocations;
        const auto index = static_cast<std::uint32_t>(allocations.size());
        auto& allocation = allocations.emplace_back();
        static_assert(sizeof(handle) == sizeof(allocation.handle));
        std::memcpy(allocation.handle.data(), &handle, sizeof(handle));
        allocation.bytes = exported.bytes;
        bases_.push_back(exported.base);
        return {index, exported.offset};
    }

    PrefixDeviceMxfp8 mxfp8(Mxfp8KvCache cache, int layers, int begin, int end, int kv_heads,
                            const Allocations* memory = nullptr,
                            const std::int32_t* mapped = nullptr) {
        if (begin == end)
            return {};
        PrefixDeviceMxfp8 result{begin, end, layers, cache.page_count};
        result.kv_heads = kv_heads;
        const auto page_count = static_cast<std::size_t>(layers) * cache.page_count;
        const auto payload = page_count * gqa_payload_bytes_per_page(kv_heads);
        const auto scales = page_count * gqa_scale_bytes_per_page(kv_heads);
        result.planes = {
            allocation(cache.key_payload, payload, memory),
            allocation(cache.value_payload, payload),
            allocation(cache.key_scales, scales),
            allocation(cache.value_scales, scales),
        };
        if (mapped != nullptr) {
            const int last_page = (end + mxfp8_page_size - 1) / mxfp8_page_size;
            const int mapped_pages = last_page - (begin / mxfp8_page_size);
            result.physical_pages.assign(mapped, mapped + mapped_pages);
        }
        return result;
    }

    PrefixDeviceGdn gdn(TextGdnState state, const Workspace& row, int end,
                        const Allocations* memory) {
        state_ = {state.history, state.recurrent, row.hidden, row.residual};
        PrefixDeviceGdn result{end, {}};
        std::array<std::shared_ptr<const void>, 4> sources;
        for (std::size_t index = 0; index < state_.size(); ++index) {
            sources[index] = memory->retain_block(state_[index]);
        }
        owners_->version = std::make_shared<PrefixStateVersion>(std::move(sources), state_bytes_);
        const auto bytes =
            std::accumulate(state_bytes_.begin(), state_bytes_.end(), std::size_t{0});
        auto address = allocation(owners_->version->data(), bytes);
        for (std::size_t index = 0; index < state_bytes_.size(); ++index) {
            result.planes[index] = {address, state_bytes_[index]};
            address.offset += state_bytes_[index];
        }
        return result;
    }
    PrefixDeviceGdn gdn(const std::shared_ptr<PrefixPrefillSnapshot>& snapshot, int end,
                        bool storing = true) {
        if (storing) {
            owners_->blocks.push_back(snapshot);
        } else {
            view_.restore_snapshot = snapshot;
        }
        auto address = allocation(snapshot->data(), snapshot->bytes());
        PrefixDeviceGdn result{end, {}};
        const auto& sizes = snapshot->sizes();
        for (std::size_t index = 0; index < sizes.size(); ++index) {
            result.planes[index] = {address, sizes[index]};
            address.offset += sizes[index];
        }
        view_.prefill_checkpoint = true;
        return result;
    }

    PrefixDeviceMxfp8 pooled(const PrefixSlab& slab, int rank, ResidentPrefixPageRange range) {
        if (range.pages == nullptr || range.end <= range.begin) {
            return {};
        }
        const auto cache = slab.cache({rank});
        auto result = mxfp8(cache, slab.layers(), range.begin, range.end, cache.kv_heads);
        const int end = (range.end + mxfp8_page_size - 1) / mxfp8_page_size;
        const int count = end - (range.begin / mxfp8_page_size);
        result.physical_pages.reserve(count);
        for (int page = 0; page < count; ++page) {
            result.physical_pages.push_back(*range.pages[page]);
        }
        return result;
    }

    void require_metadata() {
        metadata_only_ = true;
    }

    PrefixDeviceTransfer finish(PrefixDeviceLayout layout, cudaStream_t stream) {
        view_.layout = std::move(layout);
        if (view_.transfer.allocations.empty() && !metadata_only_) {
            return std::move(view_);
        }
        const int device = prefix_device("Identify decoder prefix CUDA device");
        cudaDeviceProp properties;
        check(cudaGetDeviceProperties(&properties, device), "Read decoder prefix CUDA UUID");
        static_assert(sizeof(properties.uuid.bytes) == sizeof(view_.transfer.device));
        std::memcpy(view_.transfer.device.data(), properties.uuid.bytes,
                    sizeof(properties.uuid.bytes));
        if (view_.transfer.allocations.empty()) {
            return std::move(view_);
        }
        cudaEvent_t ready;
        check_prefix_cuda(cudaEventCreateWithFlags(&ready, cudaEventInterprocess |
                                                   cudaEventDisableTiming),
                          "Create decoder prefix IPC event");
        view_.ready = std::shared_ptr<void>(ready, [](void* event) noexcept {
            cudaEventDestroy(static_cast<cudaEvent_t>(event));
        });
        cudaIpcEventHandle_t handle;
        check_prefix_cuda(cudaIpcGetEventHandle(&handle, ready),
                          "Export decoder prefix readiness handle");
        static_assert(sizeof(handle) == sizeof(view_.transfer.ready_event));
        std::memcpy(view_.transfer.ready_event.data(), &handle, sizeof(handle));
        if (owners_ && owners_->version) {
            owners_->version->submit(state_, stream, view_.ready);
        } else {
            check_prefix_cuda(cudaEventRecord(ready, stream), "Record decoder prefix readiness");
        }
        return std::move(view_);
    }

private:
    struct Owners {
        std::vector<std::shared_ptr<const void>> blocks;
        std::shared_ptr<PrefixStateVersion> version;
    };

    std::array<std::size_t, 4> state_bytes_{};
    std::array<const void*, 4> state_{};
    std::shared_ptr<Owners> owners_;
    std::vector<void*> bases_;
    bool metadata_only_ = false;
    PrefixDeviceTransfer view_;
};

int capacity(const Request& request) { return request.pages.page_count * mxfp8_page_size; }

}

PrefixPrefillSnapshot::PrefixPrefillSnapshot(const PrefixCacheLayout* layout)
    : sizes_(gdn_state_bytes(layout)) {
    check(cudaGetDevice(&device_), "Identify prefill checkpoint device");
    const auto arena = DeviceArena::current(device_);
    data_ = static_cast<std::byte*>(arena->allocate_cache(bytes()));
    storage_ = std::shared_ptr<void>(data_, [arena](void* pointer) { arena->release(pointer); });
}

PrefixPrefillSnapshot::PrefixPrefillSnapshot(const PrefixCacheLayout& layout,
                                             std::shared_ptr<void> storage, std::byte* data)
    : data_(data), sizes_(gdn_state_bytes(&layout)), storage_(std::move(storage)) {
}

PrefixPrefillSnapshot::~PrefixPrefillSnapshot() = default;

TextGdnState PrefixPrefillSnapshot::state(int bank) const {
    return {
        reinterpret_cast<Bf16*>(data_ + (bank * sizes_[0] / gdn_layers)),
        reinterpret_cast<float*>(data_ + sizes_[0] + (bank * sizes_[1] / gdn_layers)),
    };
}

void PrefixPrefillSnapshot::capture_row(const void* hidden, const void* residual,
                                        cudaStream_t stream) const {
    auto* destination = data_ + sizes_[0] + sizes_[1];
    check_prefix_cuda(
        cudaMemcpyAsync(destination, hidden, sizes_[2], cudaMemcpyDeviceToDevice, stream),
        "Capture prefill checkpoint hidden row");
    check_prefix_cuda(cudaMemcpyAsync(destination + sizes_[2], residual, sizes_[3],
                                      cudaMemcpyDeviceToDevice, stream),
                      "Capture prefill checkpoint residual row");
}

void PrefixPrefillSnapshot::capture_state(TextGdnState state, cudaStream_t stream) const {
    check_prefix_cuda(
        cudaMemcpyAsync(data_, state.history, sizes_[0], cudaMemcpyDeviceToDevice, stream),
        "Capture prefill checkpoint convolution");
    check_prefix_cuda(cudaMemcpyAsync(data_ + sizes_[0], state.recurrent, sizes_[1],
                                      cudaMemcpyDeviceToDevice, stream),
                      "Capture prefill checkpoint recurrent state");
}

void PrefixPrefillSnapshot::restore(Request& request, cudaStream_t stream) const {
    const std::array<void*, 4> destinations{
        request.states.front().history,
        request.states.front().recurrent,
        request.work.hidden,
        request.work.residual,
    };
    const auto* source = data_;
    for (std::size_t plane = 0; plane < destinations.size(); ++plane) {
        check_prefix_cuda(cudaMemcpyAsync(destinations[plane], source, sizes_[plane],
                                          cudaMemcpyDeviceToDevice, stream),
                          "Restore pooled decoder checkpoint state");
        source += sizes_[plane];
    }
}

std::byte* PrefixPrefillSnapshot::data() const {
    return data_;
}

std::size_t PrefixPrefillSnapshot::bytes() const {
    return sizes_[0] + sizes_[1] + sizes_[2] + sizes_[3];
}

const std::array<std::size_t, 4>& PrefixPrefillSnapshot::sizes() const {
    return sizes_;
}

struct PrefixPrefillPool::Release {
    std::mutex mutex;
    std::condition_variable completed;
};

PrefixPrefillPool::PrefixPrefillPool() : release_(std::make_shared<Release>()) {
}

std::size_t PrefixPrefillPool::available(std::size_t capacity) const {
    const auto existing = std::min(capacity, slots_.size());
    return capacity - existing +
           std::count_if(slots_.begin(), slots_.begin() + static_cast<std::ptrdiff_t>(existing),
                         [](const auto& slot) { return slot.use_count() == 1; });
}

void PrefixPrefillPool::wait_available(std::size_t capacity, std::size_t count) const {
    std::unique_lock lock(release_->mutex);
    release_->completed.wait(lock, [&] { return available(capacity) >= count; });
}

std::shared_ptr<PrefixPrefillSnapshot> PrefixPrefillPool::acquire(std::size_t capacity,
                                                                  const PrefixCacheLayout* layout) {
    const auto lease = [this](std::shared_ptr<PrefixPrefillSnapshot> owner) {
        auto* address = owner.get();
        return std::shared_ptr<PrefixPrefillSnapshot>(
            address,
            [owner = std::move(owner), release = release_](PrefixPrefillSnapshot*) mutable {
                const std::scoped_lock lock(release->mutex);
                owner.reset();
                release->completed.notify_one();
            });
    };
    for (std::size_t index = 0; index < std::min(capacity, slots_.size()); ++index) {
        if (slots_[index].use_count() == 1)
            return lease(slots_[index]);
    }
    if (slots_.size() >= capacity)
        return {};
    const auto sizes = gdn_state_bytes(layout);
    const auto bytes = sizes[0] + sizes[1] + sizes[2] + sizes[3];
    if (available_device_bytes() < decoder_shared_reserve + bytes)
        return {};
    try {
        return lease(slots_.emplace_back(std::make_shared<PrefixPrefillSnapshot>(layout)));
    } catch (const CudaError& error) {
        if (error.status != cudaErrorMemoryAllocation)
            throw;
        return {};
    }
}

PrefixDeviceTransfer restore_prefix_view(Request& request, const PrefixCacheRead& read,
                                         cudaStream_t initialized, MtpDraftRequest* draft,
                                         const PrefixCacheLayout* layout,
                                         const std::shared_ptr<PrefixPrefillSnapshot>& snapshot,
                                         const PrefixRestoreExtent* extent) {
    const bool has_extent = extent != nullptr;
    if (!has_extent &&
        (request.cached_gqa_tokens != 0 || (draft != nullptr && draft->valid_prefix() != 0))) {
        throw std::logic_error("Restore an already initialized decoder prefix");
    }
    if (draft && layout && !layout->draft_lane) {
        throw std::logic_error("Restore a draft decoder prefix on a non-draft lane");
    }
    const int kv_heads = layout != nullptr ? layout->kv_heads : mxfp8_kv_heads;
    const int coherent = has_extent ? extent->gdn_end : read.usable_gdn;
    const int gqa_begin = has_extent ? extent->gqa_begin : 0;
    const int gqa_end = has_extent ? extent->gqa_end : read.usable_gqa;
    const int draft_begin = has_extent ? extent->draft_begin : 0;
    int draft_rows = 0;
    if (draft != nullptr) {
        draft_rows = has_extent ? extent->draft_end : std::max(0, coherent - 1);
    }
    const bool draft_count_binding = draft != nullptr || layout == nullptr || layout->draft_lane;
    if (gqa_begin < 0 || gqa_begin > gqa_end || gqa_end > capacity(request) || coherent < 0 ||
        draft_begin < 0 || (draft != nullptr && draft_begin > draft_rows) ||
        (!has_extent &&
         (gqa_end > read.matched_gqa || coherent > gqa_end || coherent > read.matched_gdn ||
          (draft_count_binding && read.usable_draft != draft_rows)))) {
        throw std::logic_error("Invalid decoder prefix restore extent");
    }
    PrefixViewBuilder view(layout);
    if (has_extent) {
        view.require_metadata();
    }
    PrefixDeviceLayout payload;
    const auto* target_map = request.shared_prefix_pages.empty()
                                 ? nullptr
                                 : request.memory.transfers().logical_pages.data();
    payload.gqa = view.mxfp8(
        request.caches.front(), full_attention_layers, gqa_begin, gqa_end, kv_heads, nullptr,
        target_map != nullptr ? target_map + (gqa_begin / mxfp8_page_size) : nullptr);
    if (draft != nullptr && draft_rows > draft_begin) {
        const auto* draft_map =
            draft->resident_prefix_pages().empty()
                ? nullptr
                : draft->prefix_physical_pages().data() + (draft_begin / mxfp8_page_size);
        payload.draft = view.mxfp8(draft->prefix_cache_view(draft_rows, false), 1, draft_begin,
                                   draft_rows, mxfp8_kv_heads, nullptr, draft_map);
    }
    if (coherent != 0) {
        const auto storage = snapshot ? snapshot : std::make_shared<PrefixPrefillSnapshot>(layout);
        payload.gdn = view.gdn(storage, coherent, false);
    }
    auto transfer = view.finish(std::move(payload), initialized);
    if (coherent > 0 || (gqa_end > gqa_begin && draft_rows > draft_begin))
        transfer.attention_only_restore = std::make_shared<bool>(false);
    return transfer;
}

PrefixDeviceTransfer export_prefix_view(const Request& request, int gqa_begin, int draft_begin,
                                        int end, cudaStream_t source_ready,
                                        const MtpDraftRequest* draft, const Workspace* row,
                                        bool terminal, const PrefixCacheLayout* layout,
                                        const std::shared_ptr<PrefixPrefillSnapshot>& snapshot) {
    (void)terminal;
    if (draft && layout && !layout->draft_lane) {
        throw std::logic_error("Export a draft decoder prefix from a non-draft lane");
    }
    const int kv_heads = layout ? layout->kv_heads : mxfp8_kv_heads;
    if (end <= 0 || end > capacity(request) || gqa_begin < 0 || gqa_begin > end ||
        draft_begin < 0 || draft_begin > (draft ? end - 1 : 0)) {
        throw std::logic_error("Invalid decoder prefix checkpoint extent");
    }
    PrefixViewBuilder view(layout, true);
    PrefixDeviceLayout payload;
    payload.gqa = view.mxfp8(request.caches.front(), full_attention_layers, gqa_begin, end,
                             kv_heads, &request.memory);
    if (draft && draft_begin < end - 1) {
        payload.draft = view.mxfp8(draft->prefix_cache_view(end - 1, true), 1, draft_begin,
                                   end - 1, mxfp8_kv_heads, &draft->prefix_cache_memory());
    }
    payload.gdn = snapshot ? view.gdn(snapshot, end)
                           : view.gdn(request.states.front(), row != nullptr ? *row : request.work,
                                      end, &request.memory);
    return view.finish(std::move(payload), source_ready);
}

int commit_restored_prefix(Request& request, const PrefixDeviceTransfer& view,
                           MtpDraftRequest* draft, cudaStream_t stream) {
    const bool attention_only = view.attention_only_restore && *view.attention_only_restore;
    if (!attention_only && view.restore_snapshot) {
        view.restore_snapshot->restore(request, stream);
    }
    if (view.restore_snapshot) {
        synchronize_cuda(stream, "Complete decoder prefix checkpoint restore");
    }
    if (draft != nullptr)
        draft->commit_restored_prefix(attention_only ? 0 : view.layout.draft.end);
    request.cached_gqa_tokens = view.layout.gqa.end;
    return attention_only ? 0 : view.layout.gdn.end;
}

PrefixDeviceTransfer pooled_prefix_view(const PrefixSlab* slab, int rank,
                                        ResidentPrefixPageRange range, bool draft,
                                        const std::shared_ptr<PrefixPrefillSnapshot>& snapshot,
                                        int endpoint, cudaStream_t stream) {
    PrefixDeviceLayout payload;
    PrefixViewBuilder view(nullptr, true);
    if (slab != nullptr) {
        if (draft) {
            payload.draft = view.pooled(*slab, rank, range);
        } else {
            payload.gqa = view.pooled(*slab, rank, range);
        }
    }
    if (snapshot) {
        payload.gdn = view.gdn(snapshot, endpoint);
    }
    return view.finish(payload, stream);
}
}
