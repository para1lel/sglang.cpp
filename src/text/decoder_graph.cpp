#include "sglang/decoder_internal.h"
#include "sglang/text/gqa_verify_batch.h"

#include <algorithm>

namespace sglang::decoder {
namespace {

ExactGqaWorkspace singleton_scores(Allocations& execution, const ForwardLane& lane) {
    if (lane.position < singleton_score_min_visible - 1 ||
        lane.position >= singleton_score_max_visible) {
        return {};
    }
    const auto& request = *lane.request;
    const int pages = lane.position / mxfp8_page_size + 1;
    if (pages > request.pages.page_count || !request.pages.device_pages ||
        !std::all_of(request.caches.begin(), request.caches.end(), [pages](const auto& cache) {
            return pages <= cache.page_count && cache.key_payload && cache.value_payload &&
                cache.key_scales && cache.value_scales;
        })) {
        return {};
    }
    auto scores = execution.prepare_singleton_scores();
    if (scores.scores) {
        scores.capacity = pages * mxfp8_page_size;
    }
    return scores;
}

std::optional<DecodeGraphs::Slot> decode_graph_shape(const DecoderModel& model,
                                                     const ForwardLane* lanes, int count) {
    DecodeGraphs::Slot shape;
    shape.width = count;
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        if (lane.mode != TextGdnMode::decode || lane.position < 0 || lane.token_id < 0 ||
            lane.token_id >= vocab_size) {
            return std::nullopt;
        }
        const auto& request = *lane.request;
        if (!std::all_of(request.caches.begin(), request.caches.end(), [&](const auto& cache) {
                return gqa_geometry_matches(gqa_full_geometry, cache) &&
                       cache_covers(cache, request.pages, lane.position + 1);
            })) {
            return std::nullopt;
        }
        const bool split = model.backend.vector_decode_attention &&
                           target_split_position(lane.position, model.backend.target_split_end);
        const bool paged = !model.backend.ordered_attention && count > 1 &&
                           count <= paged_decode_lanes && paged_decode_position(lane.position);
        if (!split && !paged && lane.singleton_scores.scores == nullptr &&
            lane.position >= request.work.gqa_scores(text_attention_max_tokens).capacity) {
            return std::nullopt;
        }
        const auto bit = std::uint32_t{1} << static_cast<unsigned>(index);
        if (split)
            shape.split |= bit;
        if (paged)
            shape.paged |= bit;
    }
    return shape;
}

Workspace& graph_workspace(BatchGraphScratch& scratch, int index) {
    auto& work = scratch.work[index];
    if (!work.has_value()) {
        throw std::bad_optional_access{};
    }
    return *work;
}

BatchGraphAttention graph_attention(const DecodeGraphs::Slot& shape, int index) {
    const auto bit = std::uint32_t{1} << static_cast<unsigned>(index);
    if ((shape.split & bit) != 0U) {
        return BatchGraphAttention::split;
    }
    return (shape.paged & bit) != 0U ? BatchGraphAttention::paged : BatchGraphAttention::direct;
}

void capture_decode_graph(const DecoderModel& model, BatchGraphScratch& scratch,
                          const DecodeGraphs::Slot& shape, cudaStream_t stream) {
    std::array<BatchGraphLane, decoder_batch_max_lanes> capture;
    for (int index = 0; index < shape.width; ++index) {
        capture[index] = {
            &graph_workspace(scratch, index),
            &scratch.device_bindings->lanes[index],
            graph_attention(shape, index),
            scratch.device_bindings,
        };
    }
    model.capture_batch_decode(capture.data(), shape.width, stream);
}

void refresh_decode_bindings(BatchGraphScratch& scratch, const ForwardLane* lanes, int count,
                             cudaStream_t stream) {
    auto& host = *scratch.host_bindings;
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        const auto& request = *lane.request;
        auto& work = graph_workspace(scratch, index);
        auto scores = work.gqa_scores(request.pages.page_count * mxfp8_page_size);
        if (lane.singleton_scores.scores != nullptr) {
            scores = lane.singleton_scores;
        }
        const int rotary = lane.rotary_position < 0 ? lane.position : lane.rotary_position;
        const TextModelStateRow state =
            request.prefix_checkpoint_outputs
                ? TextModelStateRow{request.work.hidden, request.work.residual}
                : TextModelStateRow{};
        host.lanes[index] = {
            {0.0F, lane.token_id}, lane.position,      rotary, scores, request.pages,
            request.caches,        request.gdn_replay, state,
        };
        host.tokens[index] = lane.token_id;
        host.rotary[index] = rotary;
        for (int layer = 0; layer < full_attention_layers; ++layer) {
            host.reads[layer][index] = {
                request.caches[layer],
                request.pages,
                lane.position,
                paged_gqa_buffers(work),
            };
        }
    }
    copy_cuda_async(scratch.device_bindings, &host, sizeof(host), cudaMemcpyHostToDevice, stream,
                    "Upload batch decode bindings");
}

bool replay_decode(const DecoderModel& model, Allocations& execution, DecodeGraphs& graphs,
                   const ForwardLane* lanes, int count) {
    const auto shape = decode_graph_shape(model, lanes, count);
    if (!shape.has_value()) {
        return false;
    }
    auto* const begin = graphs.slots.begin();
    auto* const end = graphs.slots.end();
    auto* slot = std::find_if(begin, end, [&](const auto& entry) {
        return entry.width == shape->width && entry.split == shape->split &&
               entry.paged == shape->paged;
    });
    if (slot == end) {
        slot = std::min_element(begin, end,
                                [](const auto& a, const auto& b) { return a.used < b.used; });
        if (slot->used != 0U) {
            execution.reset_batch_graph(static_cast<int>(slot - begin));
        }
        *slot = *shape;
        slot->used = ++graphs.clock;
        return false;
    }
    slot->used = ++graphs.clock;
    return execution.replay_batch_graph(
        count,
        [&] {
            if (graphs.scratch == nullptr) {
                graphs.scratch = std::make_unique<BatchGraphScratch>(model.device);
            }
            graphs.scratch->prepare(count);
        },
        [&] { capture_decode_graph(model, *graphs.scratch, *shape, execution.stream()); },
        [&] {
            refresh_decode_bindings(*graphs.scratch, lanes, count, execution.stream());
            return true;
        },
        static_cast<int>(slot - begin));
}
}

DecodeRun run_decode_lanes(const DecoderModel& model, Allocations& execution,
                           DecodeGraphs& graphs, ForwardLane* lanes, int count,
                           DecodeRunOptions options) {
    DecodeRun run;
    if (count == 0) {
        return run;
    }
    const auto stream = execution.stream();
    auto& first = lanes[0];
    first.narrow_gqa = count == 1 && options.narrow_allowed;
    if (first.narrow_gqa && first.mode == TextGdnMode::decode) {
        first.singleton_scores = singleton_scores(execution, first);
    }
    const bool graph = options.cuda_graphs && options.graph_step &&
                       replay_decode(model, execution, graphs, lanes, count);
    if (graph) {
        for (int index = 0; index < count; ++index) {
            run.ready[index] = &graph_workspace(*graphs.scratch, index);
            run.graphed |= std::uint64_t{1} << static_cast<unsigned>(index);
        }
        return run;
    }
    model.forward(lanes, count, stream);
    if (options.graph_step && !model.backend.ordered_attention &&
        model.compute_eager_decode_logits(lanes, count, stream)) {
        for (int index = 0; index < count; ++index) {
            run.ready[index] = &lanes[index].request->work;
        }
    }
    return run;
}

}
