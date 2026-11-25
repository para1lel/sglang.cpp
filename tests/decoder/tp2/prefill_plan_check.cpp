#include "sglang/decoder_mtp.h"
#include "sglang/runtime/execution/decoder_prefill_plan.h"

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
};

struct Dispatch {
    char kind;
    int source;
    int offset;
    int rows;
    int lanes;
    bool operator==(const Dispatch&) const = default;
};

void check_tile(const PrefillGrant& tile, const PrefillGrant& owner, int offset, int rows) {
    need(tile.request == owner.request && tile.draft == owner.draft && tile.media == owner.media &&
         tile.scores == owner.scores, "Tile lost its request metadata");
    need(tile.tokens == owner.tokens + offset && tile.position == owner.position + offset &&
         tile.rows == rows && tile.tile_rows == owner.tile_rows &&
         tile.prompt_remaining == owner.prompt_remaining - offset, "Tile span mismatch");
    need(tile.tokens[0] == owner.tokens[offset] &&
         tile.tokens[rows - 1] == owner.tokens[offset + rows - 1], "Token endpoints mismatch");
    // Draft row j consumes token j+1. The final prompt token has no successor.
    const int seed = std::min(rows, tile.prompt_remaining - 1);
    need(seed == std::min(rows, owner.prompt_remaining - offset - 1), "Draft boundary mismatch");
    if (seed) {
        need(tile.tokens[seed] == owner.tokens[offset + seed], "Shifted draft token mismatch");
    }
}

void run(const std::vector<Input>& inputs, const std::vector<Dispatch>& expected) {
    std::vector<std::unique_ptr<Request>> requests;
    std::vector<std::vector<int>> tokens;
    std::array<std::vector<PrefillGrant>, 2> grants;
    // The planner preserves this opaque identity without calling the draft model.
    alignas(MtpDraftRequest) std::byte draft_identity[sizeof(MtpDraftRequest)];
    auto* draft = reinterpret_cast<MtpDraftRequest*>(draft_identity);
    for (int rank = 0; rank < 2; ++rank) {
        DeviceScope device(rank);
        for (const auto& input : inputs) {
            auto& request = requests.emplace_back(std::make_unique<Request>(rank));
            request->pages.page_count = (input.position + input.rows + 127) / 128;
            auto& ids = tokens.emplace_back(input.remaining);
            std::iota(ids.begin(), ids.end(), 100000 * int(requests.size()));
            grants[rank].push_back({request.get(), ids.data(), input.position, input.rows,
                                    std::min(1024, input.rows), input.remaining,
                                    rank == 0 ? draft : nullptr, nullptr});
        }
    }
    DeviceScope device(0);
    DecoderBatchStepStats stats{};
    std::vector<Dispatch> observed;
    const bool scheduled = execute_partial_projection_groups(
        grants[0].data(), int(inputs.size()), cudaStreamPerThread, stats,
        [&](int source, int offset, const PrefillGrant* tiles, PrefillGroup group,
            PrefillWorkspace&, PrefillAttentionMode mode, int) {
            const bool causal = mode == PrefillAttentionMode::causal_segments;
            need(causal || mode == PrefillAttentionMode::serial_segments, "Unexpected mode");
            std::array<PrefillGrant, causal_full_group.grants> peer;
            if (causal) peer_causal_segments(grants[1].data(), source, offset, peer.data(),
                                            group.grants);
            for (int i = 0; i < group.grants; ++i) {
                const int owner = causal ? source : source + i;
                const int begin = causal ? offset + 1024 * i : 0;
                const int rows = causal ? 1024 : inputs.at(owner).rows;
                check_tile(tiles[i], grants[0].at(owner), begin, rows);
                if (causal) check_tile(peer[i], grants[1].at(owner), begin, rows);
            }
            observed.push_back({causal ? 'C' : 'S', source, offset, group.rows, group.grants});
        },
        [&](int source, const PrefillGrant* grant, PrefillWorkspace&, int, int offset) {
            need(grant == &grants[0].at(source), "Tail owner mismatch");
            observed.push_back({'T', source, offset, grant->rows - offset, 1});
        });
    need(scheduled == !expected.empty() && observed == expected, "Unexpected dispatch sequence");
    int rows = 0, groups = 0;
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
    run({{3071, 5, 3071}}, {});
    run({{3072, 262144 - 3072, 3072}}, {{'C', 0, 0, 3072, 3}});
    run({{4096, 33, 4097}}, {{'C', 0, 0, 4096, 4}});
    run({{8192, 127, 8192}}, {{'C', 0, 0, 4096, 4}, {'C', 0, 4096, 4096, 4}});
    run({{512, 13, 512}, {1024, 128, 1028}, {7185, 211, 7185}, {17, 0, 17}},
        {{'S', 0, 0, 1536, 2}, {'C', 2, 0, 4096, 4}, {'C', 2, 4096, 3072, 3},
         {'T', 2, 7168, 17, 1}, {'T', 3, 0, 17, 1}});
    std::cout << "TP2 causal planner, peer grant ownership and token boundaries passed\n";
}
