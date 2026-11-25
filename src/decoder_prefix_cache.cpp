#include "sglang/decoder_prefix_cache.h"

#include "sglang/decoder_mtp.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace sglang::decoder {
namespace {

class PrefixViewBuilder {
public:
    PrefixDeviceAddress allocation(const void* base, std::size_t bytes) {
        cudaIpcMemHandle_t handle;
        check(cudaIpcGetMemHandle(&handle, const_cast<void*>(base)),
              "Export decoder prefix allocation handle");
        auto& allocations = view_.transfer.allocations;
        const auto index = static_cast<std::uint32_t>(allocations.size());
        auto& allocation = allocations.emplace_back();
        static_assert(sizeof(handle) == sizeof(allocation.handle));
        std::memcpy(allocation.handle.data(), &handle, sizeof(handle));
        allocation.bytes = bytes;
        return {index, 0};
    }

    PrefixDeviceMxfp8 mxfp8(Mxfp8KvCache cache, int layers, int begin, int end) {
        if (begin == end) return {};
        PrefixDeviceMxfp8 result{begin, end, layers, cache.page_count};
        const auto payload = layers * cache.page_count * mxfp8_payload_bytes_per_page;
        const auto scales = layers * cache.page_count * mxfp8_scale_bytes_per_page;
        if (layers == full_attention_layers) {
            const auto slab = allocation(cache.key_payload, 2 * (payload + scales));
            result.planes = {{slab, {slab.allocation, payload}, {slab.allocation, 2 * payload},
                              {slab.allocation, 2 * payload + scales}}};
        } else {
            result.planes = {{allocation(cache.key_payload, payload),
                              allocation(cache.value_payload, payload),
                              allocation(cache.key_scales, scales),
                              allocation(cache.value_scales, scales)}};
        }
        return result;
    }

    PrefixDeviceGdn gdn(TextGdnState state, const Workspace& row, int end) {
        return {end, {{span(state.history, gdn_layers * gdn_history_elements * sizeof(Bf16)),
                       span(state.recurrent, gdn_layers * gdn_state_elements * sizeof(float)),
                       span(row.hidden, hidden_size * sizeof(Bf16)),
                       span(row.residual, hidden_size * sizeof(Bf16))}}};
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
        check(cudaEventCreateWithFlags(&ready, cudaEventInterprocess | cudaEventDisableTiming),
              "Create decoder prefix IPC event");
        view_.ready = std::shared_ptr<void>(ready, [](void* event) noexcept {
            cudaEventDestroy(static_cast<cudaEvent_t>(event));
        });
        check(cudaEventRecord(ready, stream), "Record decoder prefix readiness");
        cudaIpcEventHandle_t handle;
        check(cudaIpcGetEventHandle(&handle, ready), "Export decoder prefix readiness handle");
        static_assert(sizeof(handle) == sizeof(view_.transfer.ready_event));
        std::memcpy(view_.transfer.ready_event.data(), &handle, sizeof(handle));
        return std::move(view_);
    }

private:
    PrefixDeviceSpan span(const void* base, std::size_t bytes) {
        return {allocation(base, bytes), bytes};
    }

    PrefixDeviceTransfer view_;
};

int capacity(const Request& request) { return request.pages.page_count * mxfp8_page_size; }

}

PrefixDeviceTransfer restore_prefix_view(Request& request, const PrefixCacheRead& read,
                                         cudaStream_t initialized, MtpDraftRequest* draft) {
    if (request.cached_gqa_tokens || (draft && draft->valid_prefix())) {
        throw std::logic_error("Restore an already initialized decoder prefix");
    }
    const int coherent = read.usable_gdn;
    const int draft_rows = draft ? std::max(0, coherent - 1) : 0;
    if (read.usable_gqa < 0 || read.usable_gqa > capacity(request) ||
        read.usable_gqa > read.matched_gqa || coherent < 0 || coherent > read.usable_gqa ||
        coherent > read.matched_gdn || read.usable_draft != draft_rows) {
        throw std::logic_error("Invalid decoder prefix restore extent");
    }
    PrefixViewBuilder view;
    PrefixDeviceLayout layout;
    layout.gqa = view.mxfp8(request.caches.front(), full_attention_layers, 0, read.usable_gqa);
    if (draft_rows) layout.draft = view.mxfp8(draft->prefix_cache_view(draft_rows, false),
                                             1, 0, draft_rows);
    if (coherent) layout.gdn = view.gdn(request.states.front(), request.work, coherent);
    return view.finish(layout, initialized);
}

PrefixDeviceTransfer export_prefix_view(const Request& request, int gqa_begin, int draft_begin,
                                        int end, cudaStream_t source_ready,
                                        const MtpDraftRequest* draft, const Workspace* row) {
    if (end <= 0 || end > capacity(request) || gqa_begin < 0 || gqa_begin > end ||
        draft_begin < 0 || draft_begin > (draft ? end - 1 : 0)) {
        throw std::logic_error("Invalid decoder prefix checkpoint extent");
    }
    PrefixViewBuilder view;
    PrefixDeviceLayout layout;
    layout.gqa = view.mxfp8(request.caches.front(), full_attention_layers, gqa_begin, end);
    if (draft && draft_begin < end - 1) {
        layout.draft = view.mxfp8(draft->prefix_cache_view(end - 1, true), 1, draft_begin, end - 1);
    }
    layout.gdn = view.gdn(request.states.front(), row ? *row : request.work, end);
    return view.finish(layout, source_ready);
}

int commit_restored_prefix(Request& request, const PrefixDeviceTransfer& view,
                           MtpDraftRequest* draft) {
    if (draft) draft->commit_restored_prefix(view.layout.draft.end);
    request.cached_gqa_tokens = view.layout.gqa.end;
    return view.layout.gdn.end;
}

}
