#include "sglang/decoder_mtp.h"
#include "sglang/runtime/execution/decoder_prefill_plan.h"
#include "sglang/runtime/execution/decoder_prefill_staged_gqa.h"

#include <array>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using namespace sglang;
using namespace sglang::decoder;
using namespace sglang::decoder::prefill_detail;

void need(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

struct Input {
    int rows;
    int position;
    int remaining;
    int cached = 0;
};

struct Dispatch {
    char kind;
    int source;
    int offset;
    int rows;
    int lanes;
    std::vector<std::pair<int, int>> segments;
    bool operator==(const Dispatch&) const = default;
};

void check_tile(const PrefillGrant& tile, const PrefillGrant& owner, int offset, int rows) {
    need(tile.request == owner.request && tile.draft == owner.draft && tile.media == owner.media &&
             tile.scores == owner.scores && tile.checkpoint == owner.checkpoint &&
             tile.checkpoints.data == owner.checkpoints.data &&
             tile.checkpoints.size == owner.checkpoints.size,
         "Tile lost its request metadata");
    need(tile.tokens == owner.tokens + offset && tile.position == owner.position + offset &&
         tile.rows == rows && tile.tile_rows == owner.tile_rows &&
         tile.prompt_remaining == owner.prompt_remaining - offset, "Tile span mismatch");
    need(tile.tokens[0] == owner.tokens[offset] &&
         tile.tokens[rows - 1] == owner.tokens[offset + rows - 1], "Token endpoints mismatch");
    const int seed = std::min(rows, tile.prompt_remaining - 1);
    need(seed == std::min(rows, owner.prompt_remaining - offset - 1), "Draft boundary mismatch");
    if (seed) {
        need(tile.tokens[seed] == owner.tokens[offset + seed], "Shifted draft token mismatch");
    }
    const auto checkpoints = prefill_checkpoints(tile);
    for (int index = 0; index < checkpoints.size; ++index) {
        const int end = checkpoints.data[index].end;
        need(end > tile.position && end <= tile.position + rows, "Unowned prefill checkpoint");
    }
    publish_prefill_checkpoints(tile);
}

struct ScheduleState {
    const std::array<std::vector<PrefillGrant>, 2>* rank_grants;
    std::vector<Dispatch>* observed;
    std::vector<int>* consumed;
};

char dispatch_kind(PrefillAttentionMode mode) {
    switch (mode) {
    case PrefillAttentionMode::causal_segments:
        return 'C';
    case PrefillAttentionMode::full_group:
        return 'F';
    case PrefillAttentionMode::batched:
        return 'B';
    case PrefillAttentionMode::native:
    case PrefillAttentionMode::serial_segments:
        return 'S';
    }
    return 'S';
}

void check_group(const ScheduleState& state, const PrefillSegment* segments,
                 const PrefillGrant* tiles, PrefillGroup group, PrefillAttentionMode mode) {
    const auto& grants = *state.rank_grants;
    auto& consumed = *state.consumed;
    Dispatch dispatch{
        dispatch_kind(mode), segments[0].source, segments[0].offset, group.rows, group.grants, {},
    };
    int total = 0;
    for (int i = 0; i < group.grants; ++i) {
        const auto [owner, begin] = segments[i];
        need(begin == consumed.at(owner), "Discontinuous prefill segment");
        const auto& grant = grants[0].at(owner);
        const int rows = std::min(grant.tile_rows, grant.rows - begin);
        check_tile(tiles[i], grant, begin, rows);
        const auto peer = prefill_tile(grants[1].at(owner), {.offset = begin, .rows = rows});
        check_tile(peer, grants[1].at(owner), begin, rows);
        consumed[owner] += rows;
        total += rows;
        dispatch.segments.emplace_back(owner, begin);
    }
    need(group.rows == total, "Grouped row span mismatch");
    state.observed->push_back(std::move(dispatch));
}

void check_tail(const ScheduleState& state, int source, const PrefillGrant* grant, int offset) {
    need(grant == &state.rank_grants->at(0).at(source), "Tail owner mismatch");
    need(offset == state.consumed->at(source), "Discontinuous prefill tail");
    for (int begin = offset; begin < grant->rows;) {
        const int rows = std::min(grant->tile_rows, grant->rows - begin);
        check_tile(prefill_tile(*grant, {.offset = begin, .rows = rows}), *grant, begin, rows);
        begin += rows;
    }
    state.consumed->at(source) = grant->rows;
    state.observed->push_back({'T', source, offset, grant->rows - offset, 1, {}});
}

void check_coverage(const std::vector<Input>& inputs, const std::vector<int>& consumed,
                    const std::vector<std::vector<int>>& published) {
    for (std::size_t source = 0; source < inputs.size(); ++source) {
        need(consumed[source] == inputs[source].rows, "Incomplete prefill coverage");
        const auto& input = inputs[source];
        const auto& ends = published[source];
        need(ends.size() == static_cast<std::size_t>(input.rows > 1 ? 2 : 1) &&
                 ends.front() == input.position + 1 && ends.back() == input.position + input.rows,
             "Missing or duplicate prefill checkpoint publication");
    }
}

void run(const std::vector<Input>& inputs, const std::vector<Dispatch>& expected,
         int min_rows = prefill_chunk_rows, bool ranked = false) {
    std::vector<std::unique_ptr<Request>> requests;
    std::vector<std::vector<int>> tokens;
    std::array<std::vector<PrefillGrant>, 2> grants;
    std::vector<std::vector<int>> published(inputs.size());
    std::vector<std::function<void(int)>> publish(inputs.size());
    std::array<std::vector<std::array<PrefillCheckpoint, 2>>, 2> checkpoints{
        std::vector<std::array<PrefillCheckpoint, 2>>(inputs.size()),
        std::vector<std::array<PrefillCheckpoint, 2>>(inputs.size()),
    };
    alignas(MtpDraftRequest) std::byte draft_identity[2][sizeof(MtpDraftRequest)];
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceScope device(rank);
        for (std::size_t source = 0; source < inputs.size(); ++source) {
            const auto& input = inputs[source];
            auto& request = requests.emplace_back(std::make_unique<Request>(rank));
            request->pages.page_count = (input.position + input.rows + 127) / 128;
            request->cached_gqa_tokens = input.cached;
            if (rank == 0) {
                auto& ids = tokens.emplace_back(input.remaining);
                std::iota(ids.begin(), ids.end(), 100000 * static_cast<int>(requests.size()));
            }
            grants[rank].push_back({
                request.get(),
                tokens[source].data(),
                input.position,
                input.rows,
                std::min(1024, input.rows),
                input.remaining,
                reinterpret_cast<MtpDraftRequest*>(draft_identity[rank]),
                nullptr,
            });
            auto& grant = grants[rank].back();
            auto& points = checkpoints[rank][source];
            points[0].end = input.position + 1;
            points[1].end = input.position + input.rows;
            grant.checkpoints = {points.data(), input.rows > 1 ? 2 : 1};
            if (rank == 0) {
                publish[source] = [&, source](int end) { published[source].push_back(end); };
                grant.checkpoint = &publish[source];
            }
        }
    }
    const DeviceScope device(0);
    const auto count = static_cast<int>(inputs.size());
    validate_ranked_prefill_grants({grants[0].data(), grants[1].data()}, count);
    DecoderBatchStepStats stats{};
    std::vector<Dispatch> observed;
    std::vector<int> consumed(inputs.size());
    const ScheduleState state{&grants, &observed, &consumed};
    const PartialProjectionSchedule schedule{
        &state,
        [](const void* context, const PrefillSegment* segments, const PrefillGrant* tiles,
           PrefillGroup group, PrefillWorkspace&, PrefillAttentionMode mode, int) {
            check_group(*static_cast<const ScheduleState*>(context), segments, tiles, group, mode);
        },
        [](const void* context, int source, const PrefillGrant* grant, PrefillWorkspace&, int,
           int offset) {
            check_tail(*static_cast<const ScheduleState*>(context), source, grant, offset);
        },
    };
    const bool scheduled = execute_partial_projection_groups(
        grants[0].data(), count, cudaStreamPerThread, stats, schedule, min_rows, gqa_rank_geometry,
        0, false, true, ranked);
    need(scheduled == !expected.empty() && observed.size() == expected.size(),
         "Unexpected dispatch count");
    for (std::size_t step = 0; step < expected.size(); ++step) {
        if (expected[step].segments.empty())
            observed[step].segments.clear();
        need(observed[step] == expected[step], "Unexpected dispatch sequence");
    }
    if (scheduled)
        check_coverage(inputs, consumed, published);
    int rows = 0;
    int groups = 0;
    for (const auto& step : expected) {
        if (step.kind == 'T') continue;
        rows += step.rows;
        ++groups;
    }
    need(stats.layerwise_prefill_rows == rows && stats.layerwise_prefill_groups == groups,
         "Incorrect grouped row accounting");
}
}

void check_tp2_prefill_plan() {
    const std::vector<Input> short_pair{{128, 0, 128}, {128, 13, 129}};
    run(short_pair, {{'S', 0, 0, 256, 2, {}}});
    run(short_pair, {}, min_partial_prefill_rows);
    run({{128, 0, 128}}, {});
    run({{128, 0, 128}, {127, 3, 128}, {129, 0, 129}},
        {{'S', 0, 0, 257, 2, {{0, 0}, {2, 0}}}, {'T', 1, 0, 127, 1, {}}});
    run({{127, 3, 127}, {128, 7, 129}, {129, 9, 129}, {17, 2, 17}},
        {{'T', 0, 0, 127, 1, {}}, {'S', 1, 0, 257, 2, {}}, {'T', 3, 0, 17, 1, {}}});
    const std::vector<Input> long_pair{{257, 0, 257}, {511, 5, 512}};
    run(long_pair, {{'S', 0, 0, 768, 2, {}}});
    run(long_pair, {{'S', 0, 0, 768, 2, {}}}, min_partial_prefill_rows);
    run({{3071, 5, 3071}}, {});
    run({{3072, 262144 - 3072, 3072}}, {{'C', 0, 0, 3072, 3, {}}});
    run({{4096, 33, 4097}}, {{'C', 0, 0, 4096, 4, {}}});
    run({{8192, 127, 8192}}, {{'C', 0, 0, 4096, 4, {}}, {'C', 0, 4096, 4096, 4, {}}});
    run({{512, 13, 512}, {1024, 128, 1028}, {7185, 211, 7185}, {17, 0, 17}},
        {
            {'S', 0, 0, 1536, 2, {}},
            {'C', 2, 0, 4096, 4, {}},
            {'C', 2, 4096, 3072, 3, {}},
            {'T', 2, 7168, 17, 1, {}},
            {'T', 3, 0, 17, 1, {}},
        });
    run({{1537, 8192, 2048}, {127, 3, 128}, {1024, 8192, 1025}},
        {
            {'B', 0, 0, 2048, 2, {{0, 0}, {2, 0}}},
            {'T', 0, 1024, 513, 1, {}},
            {'T', 1, 0, 127, 1, {}},
        });
    run(
        {
            {1024, 8192, 1024},
            {256, 13, 256},
            {1024, 12288, 1024},
            {1024, 16384, 1024},
            {1024, 20480, 1024},
            {1024, 8192, 1024},
        },
        {
            {'F', 0, 0, 4096, 4, {{0, 0}, {2, 0}, {3, 0}, {5, 0}}},
            {'S', 1, 0, 1280, 2, {{1, 0}, {4, 0}}},
        });
    run({{2601, 0, 3000}, {97, 13, 97}, {2048, 0, 2049}},
        {
            {'S', 0, 0, 4096, 4, {{0, 0}, {0, 1024}, {2, 0}, {2, 1024}}},
            {'T', 0, 2048, 553, 1, {}},
            {'T', 1, 0, 97, 1, {}},
        },
        prefill_chunk_rows, true);
    run(
        {
            {1024, 0, 1024},
            {1024, 8192, 1024},
            {1024, 12288, 1024},
            {1024, 16384, 1024},
            {1024, 8192, 1024},
        },
        {
            {'F', 0, 0, 4096, 4, {{0, 0}, {1, 0}, {2, 0}, {4, 0}}},
            {'T', 3, 0, 1024, 1, {}},
        });
    {
        const DeviceScope device(0);
        Request first(0);
        Request second(0);
        Request third(0);
        first.cached_gqa_tokens = 65535;
        PrefillGrant grants[]{
            {&first, nullptr, 65000, 1024, 1024, 1024, nullptr, nullptr},
            {&second, nullptr, 65536, 1024, 1024, 1024, nullptr, nullptr},
            {&third, nullptr, 8192, 1024, 1024, 1024, nullptr, nullptr},
        };
        need(resident_gqa_staging_visible(grants, 3) == 66024, "Resident lane extent mismatch");
        need(wants_resident_gqa_staging(65535, 65000, 992) &&
                 !wants_resident_gqa_staging(65535, 65000, 991),
             "Resident row boundary mismatch");
        grants[0].position = grants[2].position;
        const auto partners = mixed_prefill_partners(grants, 3);
        need(partners && (*partners)[0] == 2 && (*partners)[1] == 1 && (*partners)[2] == 0,
             "Nonadjacent staged GQA pair mismatch");
        grants[1].position = grants[0].position;
        need(can_batch_staged_gqa(grants, 2), "Equal-position staged GQA pair declined");
    }
    std::cout << "TP2 causal planner, peer grant ownership and token boundaries passed\n";
}
