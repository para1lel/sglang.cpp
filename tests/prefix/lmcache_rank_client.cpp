#include "sglang/runtime/cache/lmcache_group.h"
#include "sglang/runtime/cache/prefix_cache_layout.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

using namespace sglang::decoder;

namespace {

constexpr int prompt_rows = 257;

enum class Parts { all, gdn, draft };

struct PlaneShape {
    int rows;
    int layers;
    int heads;
};

PrefixDeviceTransfer transfer(const PrefixCacheLayout& layout, int end, bool draft, Parts parts) {
    PrefixDeviceTransfer view{};
    view.transfer.device[0] = static_cast<unsigned char>(layout.rank + 1);
    const auto add = [&](std::size_t bytes) {
        const auto index = static_cast<std::uint32_t>(view.transfer.allocations.size());
        LmCacheGpuAllocation slab{{}, bytes};
        slab.handle[0] = static_cast<unsigned char>(index + 1);
        view.transfer.allocations.push_back(slab);
        return PrefixDeviceAddress{index, 0};
    };
    const auto planes = [&](PrefixDeviceMxfp8& plane, PlaneShape shape) {
        const auto [rows, layers, heads] = shape;
        constexpr std::size_t offset = 128;
        const int pages = rows > 128 ? 3 : 2;
        plane.end = rows;
        plane.layers = layers;
        plane.page_count = pages;
        plane.kv_heads = heads;
        if (rows > 128)
            plane.physical_pages = {2, 0};
        const auto slots = static_cast<std::size_t>(layers) * pages * 128 * heads;
        const auto payload = slots * 256;
        const auto scales = slots * 8;
        const auto values = offset + payload;
        const auto key_scales = values + payload;
        const auto value_scales = key_scales + scales;
        const auto slab = add(value_scales + scales);
        plane.planes[0] = {slab.allocation, offset};
        plane.planes[1] = {slab.allocation, values};
        plane.planes[2] = {slab.allocation, key_scales};
        plane.planes[3] = {slab.allocation, value_scales};
    };
    if (parts == Parts::all)
        planes(view.layout.gqa, {end, 16, layout.kv_heads});
    if (parts != Parts::gdn && draft && layout.draft_lane && end > 1)
        planes(view.layout.draft, {end - 1, 1, layout.kv_heads});
    if (parts == Parts::draft)
        return view;
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

    Rank(int world, int rank)
        : layout(world == 1 ? tp1_prefix_cache_layout : tp2_prefix_cache_layout(rank)),
          keys(std::vector<int>(prompt_rows, 42), "cpu-rank-test", nullptr, &layout) {
    }
};

void print_lookup(const PrefixCacheRead& match, std::string_view op) {
    if (op == "draft-coverage") {
        std::cout << match.matched_gqa << ' ' << match.matched_gdn << ' ' << match.matched_draft;
    } else if (op == "lookup") {
        std::cout << match.matched_gqa << ' ' << match.matched_gdn << ' ' << match.usable_gqa << ' '
                  << match.usable_gdn << ' ' << match.usable_draft;
        for (const int endpoint : match.matched_gdn_endpoints)
            std::cout << ' ' << endpoint;
    } else {
        for (const int endpoint : match.draft_ready_gdn_endpoints)
            std::cout << endpoint << ' ';
    }
    std::cout << '\n';
}

void report_transfer(LmCacheStatus status, const LmCacheReceipt& receipt) {
    if (status == LmCacheStatus::miss) {
        if (receipt.gqa_written_tokens != 0)
            throw std::runtime_error("ranked miss reported written GQA");
        std::cout << "miss\n";
    } else if (status == LmCacheStatus::complete) {
        std::cout << receipt.bytes[0] << ' ' << receipt.bytes[1] << ' ' << receipt.bytes[2] << ' '
                  << receipt.gqa_written_tokens << '\n';
    } else {
        throw std::runtime_error("ranked transfer declined");
    }
}

void transfer_joint(const LmCacheGroup& group, const Rank& first, const Rank& second, int end,
                    bool draft, std::string_view op) {
    auto parts = op == "get" ? Parts::all : Parts::draft;
    if (op == "get-gdn")
        parts = Parts::gdn;
    const auto a = transfer(first.layout, end, draft, parts);
    const auto b = transfer(second.layout, end, draft && op != "get-no-peer", parts);
    std::vector<const PrefixCacheKeys*> keys{&first.keys};
    std::vector<const PrefixDeviceTransfer*> transfers{&a};
    if (first.layout.world_size == 2) {
        keys.push_back(&second.keys);
        transfers.push_back(&b);
    }
    LmCacheReceipt receipt;
    const auto status = op == "store-joint-draft"
                            ? group.store(keys, transfers, &receipt)
                            : group.retrieve(keys, transfers, &receipt, nullptr,
                                             op == "get-cancel" ? STDIN_FILENO : -1);
    report_transfer(status, receipt);
}

void store_tail(const LmCacheGroup& group, const Rank& lane, int rank) {
    const PrefixCacheKeys tail(std::vector<int>(321, 43), "cpu-rank-tail", nullptr, &lane.layout);
    auto view = transfer(lane.layout, 65, false, Parts::all);
    view.layout.gqa.begin = 256;
    view.layout.gqa.end = 321;
    view.layout.gqa.page_count = 2;
    view.layout.gqa.physical_pages = {1};
    view.layout.gdn = {};
    view.transfer.allocations.resize(1);
    if (group.store(rank, tail, view) != LmCacheStatus::complete) {
        throw std::runtime_error("ranked mapped tail STORE incomplete");
    }
}

int readiness(const LmCacheGroup& group, int world) {
    std::vector<std::array<std::uint8_t, 16>> devices;
    if (world == 2) {
        devices.resize(world);
        for (int r = 0; r < world; ++r)
            devices[r][0] = static_cast<std::uint8_t>(r + 1);
    }
    try {
        return group.ready(devices) ? 0 : 1;
    } catch (const std::runtime_error& error) {
        if (std::string_view(error.what()) == "LMCache: unsupported daemon version")
            return 2;
        throw;
    }
}

Parts store_parts(std::string_view op) {
    if (op == "store-gdn")
        return Parts::gdn;
    if (op == "store-draft")
        return Parts::draft;
    return Parts::all;
}

int run(int argc, const char* const* argv) {
    if (argc != 7)
        throw std::invalid_argument("expected op world rank end draft swap");
    const std::string_view op = argv[1];
    const int world = std::stoi(argv[2]);
    const int rank = std::stoi(argv[3]);
    const int end = std::stoi(argv[4]);
    const bool draft = std::stoi(argv[5]) != 0;
    const bool swap = std::stoi(argv[6]) != 0;
    const LmCacheGroup group({default_lmcache_endpoint(), "sglang"}, world);
    const Rank first(world, swap ? 1 : 0);
    const Rank second(world, swap ? 0 : 1);
    const auto& lane = rank == 0 ? first : second;
    std::vector<const PrefixCacheKeys*> keys{&first.keys};
    if (world == 2)
        keys.push_back(&second.keys);
    if (op == "lookup" || op == "draft-inventory" || op == "draft-coverage") {
        print_lookup(group.lookup_prefix(keys, end, draft), op);
    } else if (op == "get" || op == "get-gdn" || op == "get-draft" || op == "get-no-peer" ||
               op == "get-cancel" || op == "store-joint-draft") {
        transfer_joint(group, first, second, end, draft, op);
    } else if (op == "store-tail") {
        store_tail(group, lane, rank);
    } else if (op == "store" || op == "store-gdn" || op == "store-draft" ||
               op == "store-retaining") {
        auto view = transfer(lane.layout, end, draft, store_parts(op));
        if (op == "store-retaining") {
            view.retention_endpoints = {65, 193};
            view.retention_endpoint_count = 2;
            view.retention_requires_draft = draft;
        }
        if (group.store(rank, lane.keys, view) != LmCacheStatus::complete) {
            throw std::runtime_error("ranked STORE incomplete");
        }
    } else if (op == "ready") {
        return readiness(group, world);
    } else {
        throw std::invalid_argument("unknown operation");
    }
    return 0;
}

}

int main(int argc, char* const* argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
