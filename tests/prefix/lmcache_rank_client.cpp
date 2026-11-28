#include "sglang/runtime/cache/lmcache_group.h"
#include "sglang/runtime/cache/prefix_cache_layout.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace sglang::decoder;

namespace {

constexpr int prompt_rows = 225;

PrefixDeviceTransfer transfer(const PrefixCacheLayout& layout, int end, bool draft, bool gdn_only) {
    PrefixDeviceTransfer view{};
    view.transfer.device[0] = static_cast<unsigned char>(layout.rank + 1);
    const auto add = [&](std::size_t bytes) {
        const auto index = static_cast<std::uint32_t>(view.transfer.allocations.size());
        LmCacheGpuAllocation slab{{}, bytes};
        slab.handle[0] = static_cast<unsigned char>(index + 1);
        view.transfer.allocations.push_back(slab);
        return PrefixDeviceAddress{index, 0};
    };
    const auto planes = [&](PrefixDeviceMxfp8& plane, int rows, int layers, int heads) {
        plane = {0, rows, layers, 2, heads, {}};
        for (int i = 0; i < 4; ++i) {
            plane.planes[i] = add(layers * 256 * heads * (i < 2 ? 256 : 8));
        }
    };
    if (!gdn_only) {
        planes(view.layout.gqa, end, 16, layout.kv_heads);
        if (draft && layout.draft_lane && end > 1)
            planes(view.layout.draft, end - 1, 1, 4);
    }
    view.layout.gdn.end = end;
    const auto sizes = prefix_gdn_state_bytes(layout);
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        view.layout.gdn.planes[i] = {add(sizes[i]), sizes[i]};
    }
    return view;
}

struct Rank {
    PrefixCacheLayout layout;
    PrefixCacheKeys keys;

    Rank(int world, int rank) : layout(world == 1 ? tp1_prefix_cache_layout
                                                 : tp2_prefix_cache_layout(rank)),
        keys(std::vector<int>(prompt_rows, 42), "cpu-rank-test", nullptr, &layout) {}
};

}

int main(int argc, char** argv) {
    if (argc != 7) throw std::invalid_argument("expected op world rank end draft swap");
    const std::string op = argv[1];
    const int world = std::stoi(argv[2]), rank = std::stoi(argv[3]), end = std::stoi(argv[4]);
    const bool draft = std::stoi(argv[5]) != 0, swap = std::stoi(argv[6]) != 0;
    const LmCacheGroup group({default_lmcache_endpoint(), "sglang"}, world);
    Rank first(world, swap ? 1 : 0), second(world, swap ? 0 : 1);
    const auto& lane = rank ? second : first;
    std::vector<const PrefixCacheKeys*> keys{&first.keys};
    if (world == 2) keys.push_back(&second.keys);
    if (op == "lookup" || op == "draft-inventory") {
        const auto match = group.lookup_prefix(keys, end, draft);
        if (op == "lookup") {
            std::cout << match.matched_gqa << ' ' << match.matched_gdn << ' ' << match.usable_gqa
                      << ' ' << match.usable_gdn << ' ' << match.usable_draft;
            for (const int endpoint : match.matched_gdn_endpoints)
                std::cout << ' ' << endpoint;
        } else {
            for (const int endpoint : match.draft_ready_gdn_endpoints)
                std::cout << endpoint << ' ';
        }
        std::cout << '\n';
    } else if (op == "get" || op == "get-gdn") {
        const bool gdn_only = op == "get-gdn";
        const auto a = transfer(first.layout, end, draft, gdn_only);
        const auto b = transfer(second.layout, end, draft, gdn_only);
        std::vector<const PrefixDeviceTransfer*> transfers{&a};
        if (world == 2) transfers.push_back(&b);
        LmCacheReceipt receipt;
        const auto status = group.retrieve(keys, transfers, &receipt);
        if (status == LmCacheStatus::miss) std::cout << "miss\n";
        else if (status == LmCacheStatus::complete) {
            std::cout << receipt.bytes[0] << ' ' << receipt.bytes[1] << ' ' << receipt.bytes[2]
                      << ' ' << receipt.gqa_written_tokens << '\n';
        } else throw std::runtime_error("ranked GET declined");
    } else if (op == "store" || op == "store-gdn" || op == "store-retaining") {
        auto view = transfer(lane.layout, end, draft, op == "store-gdn");
        if (op == "store-retaining") {
            view.retention_endpoints = {65, 193};
            view.retention_endpoint_count = 2;
            view.retention_requires_draft = draft;
        }
        if (group.store(rank, lane.keys, view) != LmCacheStatus::complete) {
            throw std::runtime_error("ranked STORE incomplete");
        }
    } else if (op == "ready") {
        std::vector<std::array<std::uint8_t, 16>> devices;
        if (world == 2) {
            devices.resize(world);
            for (int r = 0; r < world; ++r)
                devices[r][0] = static_cast<std::uint8_t>(r + 1);
        }
        try {
            if (!group.ready(devices))
                return 1;
        } catch (const std::runtime_error& error) {
            if (std::string_view(error.what()) == "LMCache: unsupported daemon version")
                return 2;
            throw;
        }
    } else {
        throw std::invalid_argument("unknown operation");
    }
}
