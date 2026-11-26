#include "sglang/decoder_prefix_cache.h"

#include "sglang/decoder_mtp.h"

#include "sglang/runtime/backend/cuda_resources.h"

#include "sglang/runtime/cache/prefix_cache_layout.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
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

class PrefixStateVersion {
public:
    PrefixStateVersion(std::array<std::shared_ptr<const void>, 4> sources,
                       const std::array<std::size_t, 4>& bytes)
        : producer_(std::make_unique<Producer>(std::move(sources))), bytes_(bytes) {
        check(cudaGetDevice(&device_), "Identify decoder prefix version device");
        void* data = nullptr;
        check_prefix_cuda(cudaMalloc(&data, version_bytes()),
                          "Allocate decoder prefix state version");
        data_ = static_cast<std::byte*>(data);
    }

    ~PrefixStateVersion() {
        if (!producer_) return;
        try {
            DeviceScope device(device_);
            const auto status = complete_producer();
            if (status == cudaSuccess) {
                const auto released = release_cuda_block(data_, cudaStreamPerThread, true);
                report(consume_prefix_error(released));
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
    std::byte* data_ = nullptr;
    std::unique_ptr<Producer> producer_;
    std::array<std::size_t, 4> bytes_{};
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
        if (memory) owners_->blocks.push_back(memory->retain_block(base));
        cudaIpcMemHandle_t handle;
        check_prefix_cuda(cudaIpcGetMemHandle(&handle, const_cast<void*>(base)),
                          "Export decoder prefix allocation handle");
        auto& allocations = view_.transfer.allocations;
        const auto index = static_cast<std::uint32_t>(allocations.size());
        auto& allocation = allocations.emplace_back();
        static_assert(sizeof(handle) == sizeof(allocation.handle));
        std::memcpy(allocation.handle.data(), &handle, sizeof(handle));
        allocation.bytes = bytes;
        return {index, 0};
    }

    PrefixDeviceMxfp8 mxfp8(Mxfp8KvCache cache, int layers, int begin, int end, int kv_heads,
                            const Allocations* memory = nullptr) {
        if (begin == end) return {};
        PrefixDeviceMxfp8 result{begin, end, layers, cache.page_count};
        result.kv_heads = kv_heads;
        const auto payload = std::size_t(layers) * cache.page_count * mxfp8_page_size *
                             kv_heads * mxfp8_head_dim;
        const auto scales = std::size_t(layers) * cache.page_count * mxfp8_page_size *
                            kv_heads * mxfp8_groups;
        if (layers == full_attention_layers) {
            const auto slab = allocation(cache.key_payload, 2 * (payload + scales), memory);
            result.planes = {{slab, {slab.allocation, payload}, {slab.allocation, 2 * payload},
                              {slab.allocation, 2 * payload + scales}}};
        } else {
            result.planes = {{allocation(cache.key_payload, payload, memory),
                              allocation(cache.value_payload, payload, memory),
                              allocation(cache.key_scales, scales, memory),
                              allocation(cache.value_scales, scales, memory)}};
        }
        return result;
    }

    PrefixDeviceGdn gdn(TextGdnState state, const Workspace& row, int end,
                        const Allocations* memory = nullptr, bool terminal = false) {
        state_ = {state.history, state.recurrent, row.hidden, row.residual};
        PrefixDeviceGdn result{end, {}};
        if (memory && !terminal) {
            std::array<std::shared_ptr<const void>, 4> sources;
            for (std::size_t index = 0; index < state_.size(); ++index) {
                sources[index] = memory->retain_block(state_[index]);
            }
            owners_->version =
                std::make_shared<PrefixStateVersion>(std::move(sources), state_bytes_);
            auto address = allocation(owners_->version->data(), version_bytes());
            for (std::size_t index = 0; index < state_bytes_.size(); ++index) {
                result.planes[index] = {address, state_bytes_[index]};
                address.offset += state_bytes_[index];
            }
        } else {
            for (std::size_t index = 0; index < state_bytes_.size(); ++index) {
                const auto bytes = state_bytes_[index];
                result.planes[index] = {allocation(state_[index], bytes, memory), bytes};
            }
        }
        return result;
    }

    std::size_t version_bytes() const {
        return state_bytes_[0] + state_bytes_[1] + state_bytes_[2] + state_bytes_[3];
    }

    PrefixDeviceTransfer finish(PrefixDeviceLayout layout, cudaStream_t stream) {
        view_.layout = layout;
        if (view_.transfer.allocations.empty()) return std::move(view_);
        int device;
        check(cudaGetDevice(&device), "Identify decoder prefix CUDA device");
        cudaDeviceProp properties;
        check(cudaGetDeviceProperties(&properties, device), "Read decoder prefix CUDA UUID");
        static_assert(sizeof(properties.uuid.bytes) == sizeof(view_.transfer.device));
        std::memcpy(view_.transfer.device.data(), properties.uuid.bytes,
                    sizeof(properties.uuid.bytes));
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
    PrefixDeviceTransfer view_;
};

int capacity(const Request& request) { return request.pages.page_count * mxfp8_page_size; }

}

PrefixDeviceTransfer restore_prefix_view(Request& request, const PrefixCacheRead& read,
                                         cudaStream_t initialized, MtpDraftRequest* draft,
                                         const PrefixCacheLayout* layout) {
    if (request.cached_gqa_tokens || (draft && draft->valid_prefix())) {
        throw std::logic_error("Restore an already initialized decoder prefix");
    }
    if (draft && layout && !layout->draft_lane) {
        throw std::logic_error("Restore a draft decoder prefix on a non-draft lane");
    }
    const int kv_heads = layout ? layout->kv_heads : mxfp8_kv_heads;
    const int coherent = read.usable_gdn;
    const int draft_rows = draft ? std::max(0, coherent - 1) : 0;
    const bool draft_count_binding =
        draft != nullptr || !layout || layout->draft_lane;
    if (read.usable_gqa < 0 || read.usable_gqa > capacity(request) ||
        read.usable_gqa > read.matched_gqa || coherent < 0 || coherent > read.usable_gqa ||
        coherent > read.matched_gdn ||
        (draft_count_binding && read.usable_draft != draft_rows)) {
        throw std::logic_error("Invalid decoder prefix restore extent");
    }
    PrefixViewBuilder view(layout);
    PrefixDeviceLayout payload;
    payload.gqa = view.mxfp8(request.caches.front(), full_attention_layers, 0, read.usable_gqa,
                             kv_heads);
    if (draft_rows) {
        payload.draft = view.mxfp8(draft->prefix_cache_view(draft_rows, false), 1, 0,
                                   draft_rows, mxfp8_kv_heads);
    }
    if (coherent) payload.gdn = view.gdn(request.states.front(), request.work, coherent);
    return view.finish(payload, initialized);
}

PrefixDeviceTransfer export_prefix_view(const Request& request, int gqa_begin, int draft_begin,
                                        int end, cudaStream_t source_ready,
                                        const MtpDraftRequest* draft, const Workspace* row,
                                        bool terminal, const PrefixCacheLayout* layout) {
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
    payload.gdn = view.gdn(request.states.front(), row ? *row : request.work, end,
                           &request.memory, terminal);
    return view.finish(payload, source_ready);
}

int commit_restored_prefix(Request& request, const PrefixDeviceTransfer& view,
                           MtpDraftRequest* draft) {
    if (draft) draft->commit_restored_prefix(view.layout.draft.end);
    request.cached_gqa_tokens = view.layout.gqa.end;
    return view.layout.gdn.end;
}

}
