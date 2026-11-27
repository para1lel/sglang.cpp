#include "sglang/decoder_internal.h"

#include "sglang/text/decoder_logprobs.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <utility>

namespace sglang::decoder {
namespace {

template <typename Element>
void zero(Element* buffer, std::size_t count, cudaStream_t stream) {
    zero_cuda_async(buffer, count * sizeof(Element), stream, "Reset request state");
}

int pages_for(int capacity) {
    return (capacity + mxfp8_page_size - 1) / mxfp8_page_size;
}

std::size_t kv_bytes(int pages, int kv_heads) {
    return 2 * full_attention_layers * pages *
        (gqa_payload_bytes_per_page(kv_heads) + gqa_scale_bytes_per_page(kv_heads));
}

std::array<Mxfp8KvCache, full_attention_layers> allocate_kv(Allocations& memory, int page_count,
                                                            int kv_heads) {
    const auto payload_per_layer = page_count * gqa_payload_bytes_per_page(kv_heads);
    const auto scales_per_layer = page_count * gqa_scale_bytes_per_page(kv_heads);
    const auto scale_bytes = full_attention_layers * scales_per_layer;
    const auto payload_bytes = full_attention_layers * payload_per_layer;
    auto* keys = memory.allocate<std::uint8_t>(kv_bytes(page_count, kv_heads));
    auto* values = keys + payload_bytes;
    auto* key_scales = values + payload_bytes;
    auto* val_scales = key_scales + scale_bytes;
    std::array<Mxfp8KvCache, full_attention_layers> caches;
    for (int index = 0; index < full_attention_layers; ++index) {
        caches[index] = {
            keys + index * payload_per_layer, values + index * payload_per_layer,
            key_scales + index * scales_per_layer, val_scales + index * scales_per_layer,
            page_count, kv_heads,
        };
    }
    return caches;
}

void copy_kv_plane(std::uint8_t* destination, const std::uint8_t* source, std::size_t bytes,
                   cudaStream_t stream) {
    copy_cuda_async(destination, source, bytes, cudaMemcpyDeviceToDevice, stream,
                    "Copy MXFP8 cache during growth");
}

}

void validate_batch_options(const DecoderBatchOptions& options) {
    if (options.max_lanes < 1 || options.max_lanes > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid decoder batch capacity");
    }
    if (options.prefix_cache_entries != 0) {
        throw std::invalid_argument("Local prefix cache capacity is unsupported with LMCache");
    }
    if (options.max_num_batched_tokens < 0 ||
        options.max_num_batched_tokens > decoder_batch_max_step_tokens) {
        throw std::invalid_argument("Invalid decoder batch step budget");
    }
}

int reserved_verification_rows(int output, int mtp_steps) {
    return mtp_steps && output > 1 ? std::min(mtp_steps + 1, output - 1) : 0;
}

std::size_t request_resident_bytes(int capacity, int verification_rows, bool draft) {
    constexpr std::size_t gdn_bytes = gdn_layers *
        (gdn_history_elements * sizeof(Bf16) + gdn_state_elements * sizeof(float));
    constexpr std::size_t workspace_allowance = 16 * 1024 * 1024;
    const int pages = pages_for(capacity);
    const auto target = kv_bytes(pages, mxfp8_kv_heads) + pages * sizeof(std::int32_t) +
                        gdn_bytes;
    const auto draft_kv = kv_bytes(pages, mxfp8_kv_heads) / full_attention_layers +
                          pages * sizeof(std::int32_t);
    const auto verifier = verification_rows ? gdn_bytes + verification_rows * workspace_allowance
                                            : 0;
    return target + workspace_allowance + (draft ? draft_kv + workspace_allowance : 0) + verifier;
}

int limit_resident_output(int prompt, const DecoderOptions& options, std::size_t total_bytes,
                          std::size_t free_bytes, std::size_t extra_bytes) {
    const auto fits = [&](int output, std::size_t bytes) {
        const bool scoring = options.score_prompt || bool(options.label_token_ids);
        const auto resident = request_resident_bytes(
            prompt + std::max(output, int(scoring)) - 1,
            reserved_verification_rows(output, options.mtp_steps),
            options.mtp_steps != 0);
        return resident + extra_bytes <= bytes;
    };
    int output = options.max_new_tokens;
    if (options.incremental_kv) {
        int low = 0;
        while (low < output) {
            const int middle = low + (output - low + 1) / 2;
            if (fits(middle, total_bytes)) {
                low = middle;
            } else {
                output = middle - 1;
            }
        }
    }
    if ((output == 0 && ((!options.score_prompt && !options.label_token_ids) ||
                        options.max_new_tokens > 0)) ||
        !fits(output, total_bytes)) {
        throw DecoderCapacityError("Prompt and output budget exceed resident decoder capacity",
                                   false);
    }
    if (!fits(output, free_bytes)) {
        throw DecoderCapacityError("Insufficient free resident decoder memory. Retry later.", true);
    }
    return output;
}

std::size_t reserve_request_memory(RequestBlockCache& retired, DecoderOptions& request,
                                   int prompt, std::size_t capacity, std::size_t charged,
                                   std::size_t extra_bytes) {
    const auto headroom = [&] {
        return capacity - std::min(capacity, charged + retired.unavailable_bytes());
    };
    const auto after_reserve = [](std::size_t free) {
        return free - std::min(free, decoder_shared_reserve);
    };
    auto free = available_device_bytes();
    try {
        request.max_new_tokens = limit_resident_output(
            prompt, request, capacity, std::min(headroom(), after_reserve(free + retired.bytes())),
            extra_bytes);
    } catch (const DecoderCapacityError& error) {
        if (!error.retryable || !retired.pending_bytes()) throw;
        retired.clear();
        free = available_device_bytes();
        request.max_new_tokens = limit_resident_output(
            prompt, request, capacity, std::min(headroom(), after_reserve(free)), extra_bytes);
    }
    request.min_new_tokens = std::min(request.min_new_tokens, request.max_new_tokens);
    request.incremental_kv = false;
    const auto reserved = extra_bytes + request_resident_bytes(
        RequestProgress::capacity(prompt, request),
        reserved_verification_rows(request.max_new_tokens, request.mtp_steps),
        request.mtp_steps != 0);
    if (reserved > after_reserve(free) || retired.bytes() > headroom() - reserved) {
        retired.clear();
        if (reserved > headroom() || reserved > after_reserve(available_device_bytes())) {
            throw DecoderCapacityError("Insufficient free resident decoder memory. Retry later.",
                                       true);
        }
    }
    return reserved;
}

std::shared_ptr<const DecoderTokenLogprobs>
collect_generated_logprobs(const GeneratedLogprobReadback& row, int top) {
    if (row.selected.token_id < 0) {
        throw DecoderSamplingError();
    }
    return std::make_shared<DecoderTokenLogprobs>(
        DecoderTokenLogprobs{row.selected, {row.alternatives, row.alternatives + top}});
}

Workspace::Workspace(Allocations& allocations, Bf16* verification_attention,
                     TextLogitCandidate* verification_winner)
    : hidden(allocations.allocate<Bf16>(hidden_size)),
      normalized(allocations.allocate<Bf16>(hidden_size)),
      residual(allocations.allocate<Bf16>(hidden_size)),
      projected_attention(allocations.allocate<Bf16>(hidden_size)),
      attention(verification_attention ? verification_attention :
                allocations.allocate<Bf16>(text_gqa_q_width)),
      mlp_gate(allocations.allocate<Bf16>(intermediate_size)),
      mlp_up(allocations.allocate<Bf16>(intermediate_size)),
      q_gate(allocations.allocate<Bf16>(text_gqa_q_gate_width)),
      keys(allocations.allocate<Bf16>(text_gqa_kv_width)),
      values(allocations.allocate<Bf16>(text_gqa_kv_width)),
      rope_queries(allocations.allocate<Bf16>(text_gqa_q_width)),
      rope_keys(allocations.allocate<Bf16>(text_gqa_kv_width)),
      gdn_qkv(verification_attention ? nullptr : allocations.allocate<Bf16>(text_gdn_qkv_width)),
      gdn_z(verification_attention ? nullptr : allocations.allocate<Bf16>(text_gdn_value_width)),
      gdn_b(verification_attention ? nullptr : allocations.allocate<Bf16>(text_gdn_value_heads)),
      gdn_a(verification_attention ? nullptr : allocations.allocate<Bf16>(text_gdn_value_heads)),
      gdn_scratch(verification_attention ? nullptr : allocations.allocate<Bf16>(
          text_gdn_workspace_bytes(1) / sizeof(Bf16))),
      logits(reinterpret_cast<Bf16*>(allocations.allocate<std::byte>(vocab_size * sizeof(Bf16)))),
      activation{allocations.allocate<std::uint8_t>(intermediate_size),
                 allocations.allocate<float>(text_fp8_group_count(intermediate_size))},
      argmax_scratch(
          allocations.allocate<TextLogitCandidate>(text_argmax_scratch_count(vocab_size))),
      argmax(verification_winner ? verification_winner :
             allocations.allocate<TextLogitCandidate>(1)) {
    static_assert(text_gqa_q_width == text_gdn_value_width);
}

ExactGqaWorkspace Workspace::gqa_scores(int capacity) const {
    constexpr int bytes_per_page = mxfp8_page_size * mxfp8_query_heads * sizeof(float);
    constexpr int score_pages = vocab_size * sizeof(Bf16) / bytes_per_page;
    return {reinterpret_cast<float*>(logits), std::min(capacity, score_pages * mxfp8_page_size)};
}

template <>
BatchDecodeBinding* Allocations::allocate<BatchDecodeBinding>(std::size_t count) {
    return static_cast<BatchDecodeBinding*>(allocate_elements(count, sizeof(BatchDecodeBinding)));
}

BatchGraphScratch::BatchGraphScratch(int device)
    : memory(device), device_bindings(memory.allocate<BatchDecodeBinding>(work.size())) {
    for (auto& workspace : work) {
        workspace.emplace(memory);
    }
}

ExactGqaWorkspace Request::batch_graph_scores() {
    if (long_graph_scores) {
        return *long_graph_scores;
    }
    if (pages.page_count <= 0 ||
        pages.page_count > text_attention_max_tokens / mxfp8_page_size) {
        throw std::invalid_argument("Invalid batch graph score capacity");
    }
    const int capacity = pages.page_count * mxfp8_page_size;
    const auto count = static_cast<std::size_t>(capacity) * mxfp8_query_heads;
    try {
        long_graph_scores = {memory.allocate<float>(count), capacity};
    } catch (const CudaError& error) {
        if (error.status != cudaErrorMemoryAllocation) {
            throw;
        }
        const auto previous = cudaGetLastError();
        if (previous != cudaErrorMemoryAllocation) {
            check(previous, "Check CUDA error before eager batch score fallback");
        }
        std::fprintf(stderr, "sglang decoder: Allocate batch scores for %d positions: %s, "
                     "using eager decode for long positions\n", capacity,
                     cudaGetErrorString(error.status));
        long_graph_scores = ExactGqaWorkspace{};
    }
    return *long_graph_scores;
}

void allocate_gdn(Allocations& memory, GdnBank& states, std::size_t history_elements,
                  std::size_t state_elements) {
    auto* history = memory.allocate<Bf16>(gdn_layers * history_elements);
    auto* recurrent = memory.allocate<float>(gdn_layers * state_elements);
    for (int index = 0; index < gdn_layers; ++index) {
        states[index] = {history + index * history_elements,
                         recurrent + index * state_elements};
    }
}

Request::Request(int device, bool reusable, RequestBlockCache* cache)
    : memory(device, cache), work(memory),
      decode_position(reusable ? memory.allocate<int>(2) : nullptr) {}

Request::Request(int device, int capacity, std::shared_ptr<const void> owners, bool reusable,
                 int maximum_kv)
    : Request(device, reusable) {
    memory.open_stream(std::move(owners));
    initialize(capacity, memory.stream(), maximum_kv);
}

void Request::initialize(int capacity, cudaStream_t stream, int maximum_kv, int kv_heads) {
    cached_gqa_tokens = 0;
    kv_limit = maximum_kv;
    cache_kv_heads = kv_heads;
    const bool rank_local = kv_heads != mxfp8_kv_heads;
    gdn_history_stride = rank_local ? gdn_local_history_elements : gdn_history_elements;
    gdn_state_stride = rank_local ? gdn_local_state_elements : gdn_state_elements;
     const int page_count = pages_for(capacity);
     const int table_pages = std::max(page_count, pages_for(maximum_kv));
     auto* table = memory.allocate<std::int32_t>(table_pages);
     auto& logical_pages = memory.transfers().logical_pages;
     logical_pages.resize(table_pages);
     std::iota(logical_pages.begin(), logical_pages.end(), 0);
     copy_cuda_async(table, logical_pages.data(), table_pages * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice, stream, "Upload MXFP8 page table");
     pages = {table, page_count};
    caches = allocate_kv(memory, page_count, cache_kv_heads);
    allocate_gdn(memory, states, gdn_history_stride, gdn_state_stride);
 }

void Request::grow_kv(int positions, int written, cudaStream_t stream) {
    written = std::max(written, cached_gqa_tokens);
    const int needed = pages_for(positions);
    const int previous = pages.page_count;
    if (needed <= previous) {
        return;
    }
    if (positions > kv_limit || written > previous * mxfp8_page_size) {
        throw std::logic_error("Invalid decoder KV cache growth");
    }
    const int next = std::min(pages_for(kv_limit), std::max(needed, 2 * previous));
    const auto replacement = allocate_kv(memory, next, cache_kv_heads);
    if (written) {
        const auto payload = pages_for(written) * gqa_payload_bytes_per_page(cache_kv_heads);
        const auto scales = pages_for(written) * gqa_scale_bytes_per_page(cache_kv_heads);
        for (int layer = 0; layer < full_attention_layers; ++layer) {
            const auto& old = caches[layer];
            const auto& fresh = replacement[layer];
            copy_kv_plane(fresh.key_payload, old.key_payload, payload, stream);
            copy_kv_plane(fresh.value_payload, old.value_payload, payload, stream);
            copy_kv_plane(fresh.key_scales, old.key_scales, scales, stream);
            copy_kv_plane(fresh.value_scales, old.value_scales, scales, stream);
        }
    }
    synchronize_cuda(stream, "Complete MXFP8 cache growth");
    memory.discard_graph();
    memory.free_block(caches.front().key_payload, kv_bytes(previous, cache_kv_heads));
    if (long_graph_scores && long_graph_scores->scores) {
        memory.free_block(long_graph_scores->scores,
                          long_graph_scores->capacity * mxfp8_query_heads * sizeof(float));
    }
    long_graph_scores.reset();
    caches = replacement;
    pages.page_count = next;
}

void Request::reset(cudaStream_t stream) {
    zero(states.front().history, gdn_layers * gdn_history_stride, stream);
    zero(states.front().recurrent, gdn_layers * gdn_state_stride, stream);
}

void Request::reset() {
    reset(memory.stream());
}

}

namespace sglang::decoder {

Workspace::Workspace() noexcept = default;

}
