#include "sglang/runtime/cache/prefix_payload.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
    using namespace sglang::decoder;
    if (argc != 3) return 1;
    const LmCacheClient client({argv[1], "shutdown", 1000});
    const PrefixCacheKeys keys({1}, "shutdown");
    PrefixDeviceTransfer view{};
    view.transfer.allocations.push_back({{}, 4});
    view.layout.gdn.end = 1;
    for (std::size_t i = 0; i < view.layout.gdn.planes.size(); ++i) {
        view.layout.gdn.planes[i] = {{0, i}, 1};
    }
    try {
        client.retrieve(keys, view, nullptr, std::stoi(argv[2]));
    } catch (const std::runtime_error& error) {
        if (std::string_view(error.what()) == "LMCache: restore stopped") return 0;
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cerr << "GET completed without a transfer receipt\n";
    return 1;
}
