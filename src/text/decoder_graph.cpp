#include "sglang/decoder_internal.h"

#include <algorithm>
#include <utility>

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

bool replay_narrow(const DecoderModel& model, Allocations& execution, DecodeGraphs& graphs,
                   const ForwardLane* lanes, int count) {
    if (count == 1 && !lanes[0].singleton_scores.scores && lanes[0].position >=
        lanes[0].request->work.gqa_scores(text_attention_max_tokens).capacity) {
        return false;
    }
    auto& warmed = graphs.narrow_warmed[count - 1];
    if (!std::exchange(warmed, true)) {
        return false;
    }
    return execution.replay_batch_graph(count, [&] {
        if (!graphs.narrow) {
            graphs.narrow = std::make_unique<BatchGraphScratch>(model.device);
        }
    }, [&] {
        auto& scratch = *graphs.narrow;
        std::array<BatchGraphLane, decoder_batch_max_graph_lanes> capture;
        for (int index = 0; index < count; ++index) {
            capture[index] = {&*scratch.work[index], scratch.device_bindings + index};
        }
        model.capture_batch_decode(capture.data(), count, execution.stream());
    }, [&] {
        auto& scratch = *graphs.narrow;
        for (int index = 0; index < count; ++index) {
            const auto& lane = lanes[index];
            auto& request = *lane.request;
            auto scores = scratch.work[index]->gqa_scores(text_attention_max_tokens);
            if (lane.singleton_scores.scores) {
                scores = lane.singleton_scores;
            } else if (count > 1 && paged_decode_position(lane.position)) {
                scores = {};
            } else if (lane.position >= scores.capacity) {
                scores = request.batch_graph_scores();
                if (lane.position >= scores.capacity) {
                    return false;
                }
            }
            const int rotary = lane.rotary_position < 0 ? lane.position : lane.rotary_position;
            scratch.bindings[index] = {{0.0f, lane.token_id}, lane.position, rotary, scores,
                                       request.pages, request.caches, request.states};
        }
        check(cudaMemcpyAsync(scratch.device_bindings, scratch.bindings.data(),
                              count * sizeof(BatchDecodeBinding), cudaMemcpyHostToDevice,
                              execution.stream()), "Upload batch decode bindings");
        return true;
    });
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
        count <= decoder_batch_max_graph_lanes && replay_narrow(model, execution, graphs, lanes,
                                                                count);
    if (graph) {
        for (int index = 0; index < count; ++index) {
            run.ready[index] = &*graphs.narrow->work[index];
            run.graphed |= std::uint64_t{1} << index;
        }
        return run;
    }
    model.forward(lanes, count, stream);
    if (options.graph_step && model.compute_eager_decode_logits(lanes, count, stream)) {
        for (int index = 0; index < count; ++index) {
            run.ready[index] = &lanes[index].request->work;
        }
    }
    return run;
}

}
