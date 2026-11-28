#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/decoder_media_internal.h"
#include "sglang/testing/media/decoder_fixture.h"
#include "sglang/testing/media/decoder_trace.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <limits>

namespace {

using namespace sglang;
using namespace media_test;
using Bf16 = __nv_bfloat16;

template <class Function>
void rejects(Function request, std::string_view diagnostic = "") {
    try {
        request();
    } catch (const std::invalid_argument& error) {
        require(std::string_view(error.what()).find(diagnostic) != std::string_view::npos,
                error.what());
        return;
    }
    throw std::runtime_error("Malformed media request was accepted");
}

MultimodalPrompt check_plan() {
    const auto plan = prepare_multimodal_prompt(prompt(), layouts());
    require(plan.token_ids.size() == physical && plan.positions.size() == physical &&
                plan.next_position == rotary && plan.spans.size() == spans.size(),
            "Expanded physical and rotary lengths differ from the mixed fixture");
    std::vector<int> expected(125, 8149);
    expected.push_back(begin);
    expected.insert(expected.end(), 12, image);
    expected.insert(expected.end(), {end, 17, begin, 111, 112, begin});
    expected.insert(expected.end(), 6, video);
    expected.insert(expected.end(), {end, 113, begin});
    expected.insert(expected.end(), 6, video);
    expected.insert(expected.end(), {end, end, 18, begin});
    expected.insert(expected.end(), 96, image);
    expected.insert(expected.end(), {end, 19, 20});
    require(plan.token_ids == expected, "Mixed prompt order or video timestamps changed");
    for (std::size_t index = 0; index < spans.size(); ++index) {
        const auto& wanted = spans[index];
        const auto& actual = plan.spans[index];
        require(actual.token_begin == wanted.start && actual.rows == wanted.height * wanted.width &&
                    actual.media_index == std::size_t(wanted.item) &&
                    actual.feature_begin == wanted.first,
                "Mixed media span or feature order changed");
    }
    for (int row = 0; row < physical; ++row) {
        require(equal(plan.positions[row], axes(row)), "Unexpected mixed rotary vector");
    }
    return plan;
}

std::shared_ptr<Bf16> allocate(std::size_t size, bool managed = false) {
    void* allocation = nullptr;
    cuda(managed ? cudaMallocManaged(&allocation, size * sizeof(Bf16))
                 : cudaMalloc(&allocation, size * sizeof(Bf16)));
    return {static_cast<Bf16*>(allocation), [](Bf16* address) { cudaFree(address); }};
}

std::vector<DecoderMedia> fixtures() {
    std::vector<DecoderMedia> output;
    for (const auto& layout : layouts()) {
        const int media = static_cast<int>(output.size());
        const int rows = decoder_media_feature_rows(layout);
        std::vector<std::uint16_t> host(std::size_t(rows) * width);
        for (int row = 0; row < rows; ++row) {
            for (int channel = 0; channel < width; ++channel) {
                host[std::size_t(row) * width + channel] = sample(media, row, channel);
            }
        }
        auto device = allocate(host.size());
        cuda(cudaMemcpy(device.get(), host.data(), host.size() * sizeof(host[0]),
                        cudaMemcpyHostToDevice));
        output.push_back({layout, std::move(device), host.size()});
    }
    return output;
}

void check_embedding(const MultimodalPrompt& plan, const std::vector<DecoderMedia>& media) {
    decoder::MultimodalInput input{nullptr, plan, {}};
    for (const auto& item : media) {
        input.features.push_back(item.features);
    }
    const decoder::MediaPrefill view{&input, nullptr};
    cudaStream_t stream;
    cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    constexpr std::uint16_t sentinel = 0x400a;
    constexpr int tile = 128;
    const auto scratch = allocate((tile + 2) * width);
    std::vector<std::uint16_t> before((tile + 2) * width, sentinel);
    auto after = before;
    for (int start = 0; start < physical; start += tile) {
        const int count = std::min(tile, physical - start);
        cuda(cudaMemcpyAsync(scratch.get(), before.data(), before.size() * sizeof(before[0]),
                             cudaMemcpyHostToDevice, stream));
        view.substitute(scratch.get() + width, start, count, stream);
        cuda(cudaMemcpyAsync(after.data(), scratch.get(), after.size() * sizeof(after[0]),
                             cudaMemcpyDeviceToHost, stream));
        cuda(cudaStreamSynchronize(stream));
        for (int row = -1; row <= tile; ++row) {
            const auto [item, source_row] = row >= 0 && row < count
                ? feature(start + row) : std::pair{-1, -1};
            if (row >= 0 && row < count) {
                const auto* expected = item < 0 ? nullptr
                    : media[item].features.get() + std::size_t(source_row) * width;
                require(view.feature(start + row) == expected, "Wrong single-row feature address");
            }
            for (int channel = 0; channel < width; ++channel) {
                const auto expected = item < 0 ? sentinel : sample(item, source_row, channel);
                require(after[std::size_t(row + 1) * width + channel] == expected,
                        "Wrong BF16 replacement or surrounding embedding byte modified");
            }
        }
    }
    cuda(cudaStreamDestroy(stream));
}

void check_public_guards(const Decoder& engine, const std::vector<DecoderMedia>& media) {
    const auto template_tokens = prompt();
    auto items = media;
    --items.back().feature_elements;
    rejects([&] { engine.generate_multimodal(template_tokens, items); });
    items = media;
    std::swap(items[0], items[1]);
    rejects([&] { engine.generate_multimodal(template_tokens, items); });
    items = media;
    items[1].layout.video_timestamp_tokens[1] = {image};
    rejects([&] { engine.generate_multimodal(template_tokens, items); });
    items = media;
    items[0].layout.patch_height = 5;
    rejects([&] { engine.generate_multimodal(template_tokens, items); });
    items = media;
    items[0].features = allocate(items[0].feature_elements, true);
    rejects([&] { engine.generate_multimodal(template_tokens, items); });
    items = media;
    const auto* odd = reinterpret_cast<const std::byte*>(items[0].features.get()) + 1;
    items[0].features = {items[0].features, reinterpret_cast<const Bf16*>(odd)};
    rejects([&] { engine.generate_multimodal(template_tokens, items); });
    items = media;
    items.pop_back();
    rejects([&] { engine.generate_multimodal(template_tokens, items); });
    auto invalid = template_tokens;
    invalid[125] = 19;
    rejects([&] { engine.generate_multimodal(invalid, media); });
    DecoderOptions options;
    options.mtp_steps = decoder_max_mtp_steps + 1;
    rejects([&] { engine.generate_multimodal(template_tokens, media, options); }, "MTP");
    options.mtp_steps = 0;
    options.max_new_tokens = 262145 - physical;
    rejects([&] { engine.generate_multimodal(template_tokens, media, options); });
    options.mtp_steps = 4;
    options.max_new_tokens = std::numeric_limits<int>::max();
    rejects([&] { engine.generate_multimodal(template_tokens, media, options); },
            "Prompt and output budget must fit");
    const auto cancelled = engine.generate_multimodal(template_tokens, media, {},
        DecoderCallbacks{{}, [] { return true; }});
    require(cancelled.output_token_ids.empty() &&
                cancelled.finish_reason == DecoderFinishReason::cancelled,
            "Pre-admission cancellation generated a token");
}

void print_result(std::string_view name, const DecoderResult& result) {
    std::cout << name;
    for (std::size_t row = 0; row < result.output_token_ids.size(); ++row) {
        std::cout << ' ' << result.output_token_ids[row]
                  << '/' << result.step_top_logits[row].front().logit;
    }
    std::cout << '\n';
}

void check_dispatch(const Decoder& engine, const std::vector<DecoderMedia>& media, bool fallback) {
    DecoderOptions options;
    options.max_new_tokens = 4;
    options.top_logits = 8;
    options.eager = true;
    options.max_num_batched_tokens = mxfp8_page_size;
    options.prefix_cache_salt = "mixed-media";
    std::vector<int> received;
    start_trace(fallback);
    const auto result = engine.generate_multimodal(prompt(), media, options,
        DecoderCallbacks{[&](int token) { received.push_back(token); return true; }, {}});
    finish_trace();
    require(result.cached_prompt_tokens == 0 && result.mtp_rounds == 0 &&
                result.finish_reason == DecoderFinishReason::length &&
                result.output_token_ids.size() == 4 && result.step_top_logits.size() == 4 &&
                result.output_token_ids == received,
            "Unexpected media generation, callbacks, cache hit, or speculative draft");
    print_result(fallback ? "one-row fallback" : "three prefill chunks", result);
}

bool equal_scores(const DecoderResult& left, const DecoderResult& right) {
    const auto same_row = [](const auto& first, const auto& second) {
        return std::equal(first.begin(), first.end(), second.begin(), second.end(),
            [](const DecoderLogit& a, const DecoderLogit& b) {
                return a.token_id == b.token_id &&
                    std::memcmp(&a.logit, &b.logit, sizeof(a.logit)) == 0;
            });
    };
    return std::equal(left.step_top_logits.begin(), left.step_top_logits.end(),
                      right.step_top_logits.begin(), right.step_top_logits.end(), same_row);
}

void check_media_identity(const Decoder& engine, std::vector<DecoderMedia> media) {
    require(media[0].feature_elements == media[1].feature_elements,
            "Identity fixture needs interchangeable feature shapes");
    DecoderOptions options;
    options.top_logits = 16;
    options.temperature = 0.8;
    options.top_k = 4;
    options.seed = -421;
    options.prefix_cache_salt = "same-template-and-salt";
    const auto before = engine.generate_multimodal(prompt(), media, options);
    std::swap(media[0].features, media[1].features);
    const auto changed = engine.generate_multimodal(prompt(), media, options);
    std::swap(media[0].features, media[1].features);
    const auto replay = engine.generate_multimodal(prompt(), media, options);
    require(before.cached_prompt_tokens == 0 && changed.cached_prompt_tokens == 0 &&
                replay.cached_prompt_tokens == 0 &&
                before.output_token_ids.size() == 1 && changed.output_token_ids.size() == 1 &&
                before.output_token_ids == replay.output_token_ids &&
                equal_scores(before, replay) && !equal_scores(before, changed),
            "Same template and salt reused another media owner's scores");
    print_result("original media owners", before);
    print_result("swapped media owners", changed);
    print_result("replayed media owners", replay);
}

void check_chunked_media(const Decoder& engine, const std::vector<DecoderMedia>& media) {
    DecoderOptions options;
    options.max_new_tokens = 4;
    options.top_logits = 16;
    options.eager = true;
    const auto expected = engine.generate_multimodal(prompt(), media, options);
    require(expected.output_token_ids.size() == 4 && expected.step_top_logits.size() == 4 &&
                expected.finish_reason == DecoderFinishReason::length,
            "Default media generation did not return four scored IDs");
    options.max_num_batched_tokens = 64;
    const auto actual = engine.generate_multimodal(prompt(), media, options);
    const auto repeated = engine.generate_multimodal(prompt(), media, options);
    require(actual.output_token_ids == expected.output_token_ids &&
                actual.finish_reason == expected.finish_reason &&
                actual.cached_prompt_tokens == 0 && actual.mtp_rounds == 0 &&
                actual.output_token_ids == repeated.output_token_ids &&
                equal_scores(actual, repeated),
            "Chunked media prefill changed greedy IDs or same-budget replay scores");
    print_result("media with smaller prefill budget", actual);
}

void check_speculative_media(const Decoder& engine, const std::vector<DecoderMedia>& media) {
    auto tokens = prompt();
    constexpr int prompt_rows = 3 * mxfp8_page_size - 2;
    tokens.insert(tokens.end(), prompt_rows - physical, 8149);
    DecoderOptions options;
    options.top_logits = 8;
    options.eager = true;
    for (int budget : {3, 14}) {
        options.max_new_tokens = budget;
        options.mtp_steps = 0;
        const auto expected = engine.generate_multimodal(tokens, media, options);
        options.mtp_steps = 4;
        std::vector<int> emitted;
        start_mtp_trace(prompt_rows);
        const auto actual = engine.generate_multimodal(tokens, media, options,
            DecoderCallbacks{[&](int token) { emitted.push_back(token); return true; }, {}});
        finish_mtp_trace();
        require(std::size_t(budget) == expected.output_token_ids.size() &&
                    expected.step_top_logits.size() == expected.output_token_ids.size() &&
                    actual.output_token_ids == expected.output_token_ids &&
                    actual.output_token_ids == emitted && equal_scores(actual, expected) &&
                    actual.finish_reason == DecoderFinishReason::length &&
                    actual.finish_reason == expected.finish_reason && actual.mtp_rounds > 0 &&
                    actual.mtp_drafted_tokens > 0,
                "Media K4 changed greedy tokens, exact target logits, or final-tail behavior");
        std::cout << "media K4 cap=" << budget << " drafted=" << actual.mtp_drafted_tokens
                  << " accepted=" << actual.mtp_accepted_tokens << ' ';
        print_result("target logits", actual);
        if (budget == 3) continue;

        options.top_logits = 0;
        DecoderBatch batch(engine, DecoderBatchOptions{false, 1, 0,
                                                        options.max_num_batched_tokens});
        require(batch.admit_multimodal(0, tokens, options, media) == 0,
                "Unexpected cache hit in the media K4 page test");
        std::vector<int> committed;
        int discarded = 0;
        bool verified_across_page = false;
        for (;;) {
            const auto step = batch.advance_chunked({true})[0];
            require(!step.sampling_failed && !step.constraint_failed, "Media K4 step failed");
            if (!committed.empty() && step.mtp_verified_rows > 0) {
                const int start = prompt_rows + static_cast<int>(committed.size()) - 1;
                const int end = start + step.mtp_verified_rows - 1;
                verified_across_page |= start / mxfp8_page_size != end / mxfp8_page_size;
            }
            require(step.mtp_replayed_rows == 0, "Media MTP replayed committed GDN rows");
            discarded +=
                std::max(0, step.mtp_verified_rows - static_cast<int>(step.token.has_value()) -
                                step.extra_token_count);
            if (step.token) committed.push_back(*step.token);
            committed.insert(committed.end(), step.extra_tokens.begin(),
                step.extra_tokens.begin() + step.extra_token_count);
            if (step.finish_reason) {
                require(*step.finish_reason == expected.finish_reason, "Media K4 finish changed");
                break;
            }
        }
        require(committed == expected.output_token_ids && verified_across_page && discarded > 0,
                "Media K4 changed after cross-page verification and discarded proposals");
        std::cout << "media K4 cross-page verified, discarded=" << discarded << '\n';
    }
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 2 || argc == 3,
                "Usage: media-contract CHECKPOINT [--embedding-only]");
        const auto plan = check_plan();
        const auto media = fixtures();
        check_embedding(plan, media);
        std::cout << "Mixed planner and BF16 row substitution passed\n";
        if (argc == 2) {
            const Decoder engine(argv[1], 0, true);
            check_public_guards(engine, media);
            check_dispatch(engine, media, false);
            check_dispatch(engine, media, true);
            check_media_identity(engine, media);
            check_chunked_media(engine, media);
            check_speculative_media(engine, media);
        } else {
            require(std::string_view(argv[2]) == "--embedding-only", "Unknown option");
        }
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
