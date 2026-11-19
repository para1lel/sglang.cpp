#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_prefill.h"

#include <algorithm>

namespace sglang::decoder {
namespace {

auto first_span(const MultimodalPrompt& prompt, int position) {
    return std::lower_bound(prompt.spans.begin(), prompt.spans.end(), position,
        [](const DecoderMediaSpan& span, int token) {
            return span.token_begin + span.rows <= token;
        });
}

const Bf16* feature_at(const MultimodalInput& input, const DecoderMediaSpan& span, int position) {
    const auto row = std::size_t(span.feature_begin + position - span.token_begin);
    return input.features[span.media_index].get() + row * hidden_size;
}

}

const Bf16* MediaPrefill::feature(int token_position) const {
    const auto span = first_span(input->prompt, token_position);
    if (span == input->prompt.spans.end() || token_position < span->token_begin) {
        return nullptr;
    }
    return feature_at(*input, *span, token_position);
}

void MediaPrefill::substitute(Bf16* embeddings, int position, int rows, cudaStream_t stream) const {
    const auto& prompt = input->prompt;
    const int end = position + rows;
    constexpr auto row_bytes = hidden_size * sizeof(Bf16);
    for (auto span = first_span(prompt, position);
         span != prompt.spans.end() && span->token_begin < end; ++span) {
        const int start = std::max(position, span->token_begin);
        const int count = std::min(end, span->token_begin + span->rows) - start;
        auto* destination = embeddings + std::size_t(start - position) * hidden_size;
        check(cudaMemcpyAsync(destination, feature_at(*input, *span, start), count * row_bytes,
                              cudaMemcpyDeviceToDevice, stream), "Substitute visual embeddings");
    }
}

MediaPrefill upload_media_positions(Request& request, const MultimodalInput& input,
                                     cudaStream_t stream) {
    const auto& positions = input.prompt.positions;
    auto* device_positions = request.memory.allocate<MropePosition>(positions.size());
    const auto bytes = positions.size() * sizeof(MropePosition);
    check(cudaMemcpyAsync(device_positions, positions.data(), bytes,
                          cudaMemcpyHostToDevice, stream), "Upload multimodal rotary positions");
    return {&input, device_positions};
}

std::shared_ptr<MultimodalInput> prepare_input(
    const std::vector<int>& template_ids, const std::vector<DecoderMedia>& media,
    std::shared_ptr<const Allocations> model) {
    auto input = std::make_shared<MultimodalInput>();
    input->model = std::move(model);
    std::vector<DecoderMediaLayout> layouts;
    layouts.reserve(media.size());
    input->features.reserve(media.size());
    for (const auto& item : media) {
        const auto rows = decoder_media_feature_rows(item.layout);
        const auto address = reinterpret_cast<std::uintptr_t>(item.features.get());
        if (!address || item.features.use_count() == 0 || address % alignof(Bf16) ||
            item.feature_elements != std::size_t(rows) * hidden_size) {
            throw std::invalid_argument("Visual features must match the grid and model width");
        }
        layouts.push_back(item.layout);
        input->features.push_back(item.features);
    }
    input->prompt = prepare_multimodal_prompt(template_ids, layouts);
    return input;
}

void validate_feature_device(const MultimodalInput& input, int device) {
    for (const auto& features : input.features) {
        cudaPointerAttributes attributes{};
        check(cudaPointerGetAttributes(&attributes, features.get()), "Inspect visual features");
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
            throw std::invalid_argument("Visual features must reside on the decoder CUDA device");
        }
    }
}

}

namespace sglang {

using namespace decoder;

DecoderResult Decoder::generate_multimodal(const std::vector<int>& template_token_ids,
                                            const std::vector<DecoderMedia>& media,
                                            const DecoderOptions& options,
                                            const DecoderCallbacks& callbacks) const {
    return generate_multimodal_impl(template_token_ids, media, options, callbacks, nullptr);
}

DecoderResult Decoder::generate_multimodal_impl(const std::vector<int>& template_token_ids,
                                                 const std::vector<DecoderMedia>& media,
                                                 const DecoderOptions& options,
                                                 const DecoderCallbacks& callbacks,
                                                 const std::vector<int>* teacher) const {
    if (options.prefix_cache_salt.size() > decoder_prefix_max_salt_bytes) {
        throw std::invalid_argument("Prefix cache salt is too long");
    }
    const auto input = prepare_input(template_token_ids, media, model_->memory);
    validate_request(input->prompt.token_ids, options);
    if (teacher &&
        (options.max_new_tokens < 1 || teacher->size() != std::size_t(options.max_new_tokens - 1) ||
         !std::all_of(teacher->begin(), teacher->end(),
                      [](int token) { return token >= 0 && token < vocab_size; }))) {
        throw std::invalid_argument("Invalid teacher-forced continuation");
    }
    DecoderResult result;
    result.peak_device_allocation_bytes = model_device_allocation_bytes();
    if (callbacks.is_cancelled && callbacks.is_cancelled()) {
        result.finish_reason = DecoderFinishReason::cancelled;
    } else if (options.max_new_tokens != 0) {
        DeviceScope selected_device(model_->device);
        validate_feature_device(*input, model_->device);
        reclaim_prefill_workspaces(model_->device);
        result.output_token_ids.reserve(options.max_new_tokens);
        model_->generate_serial(
            input->prompt.token_ids, options, callbacks, result, input, teacher);
        selected_device.restore();
    }
    return result;
}

}
