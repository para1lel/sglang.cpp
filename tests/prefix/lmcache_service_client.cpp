#include "sglang/runtime/cache/prefix_cache_layout.h"
#include "sglang/runtime/server/batch_cache.h"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

using namespace sglang::decoder;
using namespace sglang::server;
using namespace std::chrono_literals;

namespace {

using Keys = std::vector<std::shared_ptr<PrefixCacheKeys>>;
using Device = std::array<std::uint8_t, 16>;

void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

PrefixDeviceTransfer view(int rank) {
    PrefixDeviceTransfer transfer{};
    transfer.transfer.device[0] = static_cast<uint8_t>(rank + 1);
    transfer.transfer.allocations.push_back({{}, 4});
    transfer.layout.gdn.end = 1;
    for (unsigned i = 0; i < 4; ++i) transfer.layout.gdn.planes[i] = {{0, i}, 1};
    return transfer;
}

void synchronize_delayed_daemon() {
    std::string command;
    require(bool(std::getline(std::cin, command)) && command == "held", "missing daemon gate");
}

void registration_cases(BatchCache& cache, const Keys& keys) {
    for (const std::string which : {"count", "null", "device", "duplicate", "cancel"}) {
        auto join = cache.begin_restore();
        auto first = view(0), peer = view(1);
        auto second_keys = keys;
        auto pending = cache.restore(join, keys, first, {});
        require(pending.wait_for(0s) == std::future_status::timeout, "early restore");
        if (which == "cancel") join->fail();
        else {
            if (which == "count") second_keys.pop_back();
            if (which == "null") second_keys[1].reset();
            if (which == "device") peer.transfer.device[0] = 10;
            if (which == "duplicate") peer.transfer.device = first.transfer.device;
            bool rejected = false;
            try { cache.restore(join, second_keys, peer, {}); }
            catch (const std::logic_error&) { rejected = true; }
            require(rejected, "invalid peer registered");
        }
        require(pending.wait_for(1s) == std::future_status::ready, "abandoned restore wedged");
        bool failed = false;
        try { failed = !pending.get(); } catch (const std::exception&) { failed = true; }
        require(failed, "invalid restore published");
    }
    std::shared_ptr<void> reservation;
    require(cache.reserve_store(reservation), "reservation rejected");
    auto pairing = std::make_shared<BatchStorePairing>();
    auto physical = std::make_shared<std::atomic_uint64_t>(0);
    auto owner = std::make_shared<int>(0);
    auto first = view(0);
    first.store_owner = owner;
    cache.store(pairing, keys, nullptr, 0, std::move(first), reservation, physical, {});
    auto bad_keys = keys;
    bad_keys[1].reset();
    bool rejected = false;
    try { cache.store(pairing, bad_keys, nullptr, 0, view(1), reservation, physical, {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && owner.use_count() == 2, "partial store lost ownership");
    reservation.reset();
    pairing.reset();
    require(owner.use_count() == 1 && !cache.stores_pending() && !*physical, "orphan store");
    cache.reset();
}

void store_case(BatchCache& cache, const Keys& keys, const std::string& mode) {
    auto pair = std::make_shared<BatchStorePairing>();
    auto owner = std::make_shared<int>(0);
    auto physical = std::make_shared<std::atomic_uint64_t>(0);
    std::array<std::promise<void>, 2> notifications;
    std::shared_ptr<void> ticket, blocker;
    require(cache.reserve_store(ticket) && cache.reserve_store(blocker), "capacity setup");
    require(cache.reserve_store(ticket) && !cache.store_capacity(), "ticket double charged");
    for (int rank : {1, 0}) {
        auto lane = view(rank);
        lane.store_owner = owner;
        cache.store(pair, keys, nullptr, 0, std::move(lane), ticket, physical,
                    [&notifications, rank] { notifications[rank].set_value(); });
    }
    ticket.reset();
    pair.reset();
    synchronize_delayed_daemon();
    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while (cache.metrics().rank_written_bytes[0][1] != 4 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    const auto held = cache.metrics();
    require(*physical == 1 && held.rank_stores == std::vector<std::uint64_t>({1, 0}) &&
                held.rank_written_bytes[0][1] == 4 && held.rank_written_bytes[1][1] == 0,
            "physical store receipt hidden behind delayed peer");
    require(cache.stores_pending() && !held.gdn_written_bytes && !held.gqa_written_tokens,
            "logical store receipt published before delayed peer finished");
    require(!cache.store_capacity() && owner.use_count() == 3 && !cache.metrics().stores,
            "released checkpoint before delayed peer finished");
    std::cout << "release" << std::endl;
    for (auto& n : notifications) {
        require(n.get_future().wait_for(3s) == std::future_status::ready, "lost store callback");
    }
    blocker.reset();
    const auto stats = cache.metrics();
    const bool success = mode == "success";
    require(owner.use_count() == 1 && !cache.stores_pending() && cache.store_capacity(),
            "store retained views or reservations");
    require(*physical == (success ? 2u : 1u) && stats.stores == success, "store count");
    require(stats.rank_stores == std::vector<std::uint64_t>({1, success}), "rank store count");
    require(stats.rank_written_bytes[0][1] == 4 && stats.rank_written_bytes[1][1] == 4 * success,
            "physical receipts");
    require(stats.gdn_written_bytes == (success ? 8u : 0u), "logical receipt bytes");
    require(stats.gqa_written_tokens == (success ? 3u : 0u), "logical token credit");
    require(stats.declined_stores == (mode == "decline") && stats.io_errors == (mode == "error"),
            "incorrect failure accounting");
}

void restore_case(BatchCache& cache, const Keys& keys, const std::string& mode) {
    auto a = view(0), b = view(1);
    auto join = cache.begin_restore();
    auto f = cache.restore(join, keys, a, {});
    auto peer = cache.restore(join, keys, b, {});
    synchronize_delayed_daemon();
    join->fail();
    require(f.wait_for(0s) == std::future_status::timeout, "owner abandoned armed restore");
    if (mode == "stop") cache.stop_restores();
    if (mode == "stop") {
        require(f.wait_for(50ms) == std::future_status::timeout, "stop released armed GET");
    }
    std::cout << "release" << std::endl;
    require(f.wait_for(3s) == std::future_status::ready, "restore worker stuck");
    const bool success = mode == "success";
    require(f.get() == success && peer.get() == success, "inconsistent joint restore");
    const auto stats = cache.metrics();
    require(stats.rank_restores == std::vector<std::uint64_t>({success, success}), "restores");
    require(stats.rank_read_bytes[0][1] == 4 * success && stats.rank_read_bytes[1][1] == success,
            "restore must use actual per-rank receipts");
    require(stats.gdn_read_bytes == 5 * success, "joint restore receipt");
    cache.reset();
}

}

int main(int argc, char** argv) {
    if (argc != 4) return 1;
    std::vector<Device> devices(2);
    devices[0][0] = 1;
    devices[1][0] = 2;
    BatchCache cache({argv[1], "service", 2000}, devices, "service", 1);
    require(cache.metrics().available, "shared daemon unavailable");
    Keys keys;
    for (int r = 0; r < 2; ++r) {
        const auto layout = tp2_prefix_cache_layout(r);
        keys.push_back(std::make_shared<PrefixCacheKeys>(std::vector<int>{52}, "test", nullptr,
                                                         &layout));
    }
    const std::string action = argv[2];
    if (action == "register") registration_cases(cache, keys);
    else if (action == "reset") {
        bool failed = false;
        try { cache.reset(); } catch (const CacheResetUnavailable&) { failed = true; }
        require(failed && cache.metrics().io_errors == 1 && !cache.metrics().available,
                "failed rank reset claimed success");
    }
    else if (action == "store") store_case(cache, keys, argv[3]);
    else if (action == "restore") restore_case(cache, keys, argv[3]);
    else throw std::logic_error("unknown test action");
}
