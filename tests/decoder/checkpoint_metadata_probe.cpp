#include "sglang/checkpoint.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

static_assert(sizeof(std::uintptr_t) >= sizeof(std::uint64_t));

struct Mapping {
    std::uintptr_t end;
    std::uint64_t offset;
    std::uint64_t inode;
    std::string permissions;
    std::filesystem::path path;
};

using Mappings = std::map<std::uintptr_t, Mapping>;

Mappings read_mappings() {
    std::ifstream maps("/proc/self/maps");
    if (!maps) {
        throw std::runtime_error("Cannot open /proc/self/maps");
    }
    Mappings mappings;
    for (std::string line; std::getline(maps, line);) {
        std::istringstream fields(line);
        Mapping mapping{};
        std::uintptr_t begin = 0;
        char separator = 0;
        std::string device;
        if (!(fields >> std::hex >> begin >> separator >> mapping.end >> mapping.permissions >>
              mapping.offset >> device >> std::dec >> mapping.inode) || separator != '-') {
            throw std::runtime_error("Cannot parse /proc/self/maps");
        }
        std::string path;
        std::getline(fields >> std::ws, path);
        mapping.path = path;
        mappings.emplace(begin, std::move(mapping));
    }
    return mappings;
}

void print_tensor(const sglang::HostTensorView& tensor, const Mappings& mappings) {
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.data);
    auto entry = mappings.upper_bound(address);
    if (entry == mappings.begin()) {
        throw std::runtime_error("Tensor does not point into a mapping");
    }
    const auto& [begin, mapping] = *--entry;
    if (!mapping.inode || mapping.permissions != "r--p" || address >= mapping.end ||
        tensor.size_bytes > mapping.end - address) {
        throw std::runtime_error("Tensor does not fit in a private read-only file mapping");
    }
    const auto dtype = tensor.dtype == sglang::TensorDtype::BF16 ? "BF16" : "F8_E4M3";
    std::cout << dtype << '\t' << tensor.size_bytes << '\t'
              << mapping.offset + (address - begin) << '\t' << mapping.inode << '\t'
              << mapping.path.filename().string();
    for (const auto dimension : tensor.shape) {
        std::cout << '\t' << dimension;
    }
    std::cout << '\n';
}

}

int main(int argc, const char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Usage: checkpoint-metadata-probe CHECKPOINT_DIRECTORY");
        }
        const auto start = std::chrono::steady_clock::now();
        const sglang::Checkpoint checkpoint(argv[1]);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        std::cerr << "load_ns=" << std::chrono::nanoseconds(elapsed).count() << '\n';
        const auto mappings = read_mappings();
        for (std::string name; std::getline(std::cin, name, '\0');) {
            const auto& view = checkpoint.tensor(name);
            if (&view != &checkpoint.tensor(name)) {
                throw std::runtime_error("Repeated lookup returned a different tensor view");
            }
            print_tensor(view, mappings);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
