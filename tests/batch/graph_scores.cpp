#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

template <typename Function>
Function next(const char* name) {
    const auto address = dlsym(RTLD_NEXT, name);
    if (!address) {
        std::abort();
    }
    return reinterpret_cast<Function>(address);
}

enum class Kernel { other, tiles, finish };

struct Logits {
    const void* device;
    int count;
};

struct Pending {
    const int* winner;
    int count;
};

thread_local std::unordered_map<const void*, Kernel> kernels;
thread_local std::unordered_map<const void*, Logits> reductions;
thread_local std::unordered_map<const void*, Logits> winners;
thread_local std::vector<Pending> pending;
thread_local unsigned short* rows[4]{};
thread_local cudaStream_t pending_stream = nullptr;

FILE* output() {
    static FILE* file = [] {
        const char* name = std::getenv("BATCH_SCORE_OUTPUT");
        if (!name) {
            return static_cast<FILE*>(nullptr);
        }
        FILE* result = std::fopen(name, "wb");
        if (!result) {
            std::abort();
        }
        return result;
    }();
    return file;
}

Kernel identify(const void* function) {
    const auto found = kernels.find(function);
    if (found != kernels.end()) {
        return found->second;
    }
    static const auto get_name = next<decltype(&cudaFuncGetName)>("cudaFuncGetName");
    const char* name = nullptr;
    if (get_name(&name, function) != cudaSuccess || !name) {
        std::abort();
    }
    const auto kind = std::strstr(name, "argmax_tiles_kernel") ? Kernel::tiles :
        std::strstr(name, "argmax_finish_kernel") ? Kernel::finish : Kernel::other;
    kernels.emplace(function, kind);
    return kind;
}

const void* pointer(void* argument) {
    const void* value = nullptr;
    std::memcpy(&value, argument, sizeof(value));
    return value;
}

void flush(cudaStream_t stream) {
    if (pending.empty() || stream != pending_stream) {
        return;
    }
    for (std::size_t lane = 0; lane < pending.size(); ++lane) {
        const auto item = pending[lane];
        const std::int32_t header[]{*item.winner, item.count};
        if (std::fwrite(header, sizeof(header), 1, output()) != 1 ||
            std::fwrite(rows[lane], sizeof(unsigned short), item.count, output()) !=
                static_cast<std::size_t>(item.count)) {
            std::abort();
        }
    }
    pending.clear();
    std::fflush(output());
}

}

extern "C" cudaError_t cudaLaunchKernel(const void* function, dim3 grid, dim3 block, void** args,
                                       std::size_t shared, cudaStream_t stream) {
    static const auto launch = next<decltype(&cudaLaunchKernel)>("cudaLaunchKernel");
    if (output()) {
        switch (identify(function)) {
        case Kernel::tiles:
            reductions[pointer(args[2])] = {pointer(args[0]), *static_cast<int*>(args[1])};
            break;
        case Kernel::finish:
            winners[static_cast<const char*>(pointer(args[2])) + sizeof(float)] =
                reductions.at(pointer(args[0]));
            break;
        case Kernel::other:
            break;
        }
    }
    return launch(function, grid, block, args, shared, stream);
}

extern "C" cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count,
                                      cudaMemcpyKind kind, cudaStream_t stream) {
    static const auto copy = next<decltype(&cudaMemcpyAsync)>("cudaMemcpyAsync");
    if (output() && kind == cudaMemcpyDeviceToHost && count == sizeof(int)) {
        const auto found = winners.find(src);
        if (found != winners.end()) {
            const auto [device, vocab] = found->second;
            if (pending.size() == 4 || (!pending.empty() && pending_stream != stream) ||
                vocab != 248320) {
                std::abort();
            }
            auto*& row = rows[pending.size()];
            if (!row) {
                static const auto allocate = next<decltype(&cudaMallocHost)>("cudaMallocHost");
                if (allocate(reinterpret_cast<void**>(&row), vocab * sizeof(*row)) != cudaSuccess) {
                    std::abort();
                }
            }
            if (copy(row, device, vocab * sizeof(*row), kind, stream) != cudaSuccess) {
                std::abort();
            }
            pending.push_back({static_cast<const int*>(dst), vocab});
            pending_stream = stream;
        }
    }
    return copy(dst, src, count, kind, stream);
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    static const auto synchronize = next<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize");
    const auto status = synchronize(stream);
    if (status == cudaSuccess) {
        flush(stream);
    }
    return status;
}
