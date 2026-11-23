#include "sglang/decoder_internal.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstddef>
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

constexpr int capacity = sglang::decoder_batch_max_lanes;
using Candidate = sglang::TextLogitCandidate;

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
thread_local unsigned short* rows[capacity]{};
thread_local cudaStream_t pending_stream = nullptr;

FILE* open_output(const char* variable, const char* mode) {
    const char* name = std::getenv(variable);
    if (!name) {
        return nullptr;
    }
    FILE* result = std::fopen(name, mode);
    if (!result) {
        std::abort();
    }
    return result;
}

FILE* output() {
    static FILE* file = open_output("BATCH_SCORE_OUTPUT", "wb");
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

using Copy = decltype(&cudaMemcpyAsync);

void* allocate_observer_buffer(std::size_t bytes) {
    static const auto allocate = next<decltype(&cudaMallocHost)>("cudaMallocHost");
    void* buffer = nullptr;
    if (allocate(&buffer, bytes) != cudaSuccess) {
        std::abort();
    }
    return buffer;
}

void observe_copy(Copy copy, void* dst, const void* src, std::size_t bytes, cudaStream_t stream) {
    if (copy(dst, src, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::abort();
    }
}

void observe_score(Copy copy, const int* winner, Logits logits, cudaStream_t stream) {
    const auto [device, vocab] = logits;
    if (pending.size() == capacity || (!pending.empty() && pending_stream != stream) ||
        vocab != sglang::decoder::vocab_size) {
        std::abort();
    }
    auto*& row = rows[pending.size()];
    if (!row) {
        row = static_cast<unsigned short*>(allocate_observer_buffer(vocab * sizeof(*row)));
    }
    observe_copy(copy, row, device, vocab * sizeof(*row), stream);
    pending.push_back({winner, vocab});
    pending_stream = stream;
}

void observe_readback(Copy copy, void* dst, const void* src, std::size_t count,
                      cudaStream_t stream) {
    if (count != sizeof(int)) {
        return;
    }
    const auto found = winners.find(src);
    if (found != winners.end()) {
        observe_score(copy, static_cast<const int*>(dst), found->second, stream);
    }
}

}

extern "C" cudaError_t cudaLaunchKernel(const void* function, dim3 grid, dim3 block, void** args,
                                       std::size_t shared, cudaStream_t stream) {
    static const auto launch = next<decltype(&cudaLaunchKernel)>("cudaLaunchKernel");
    if (output()) {
        const auto kind = identify(function);
        switch (kind) {
        case Kernel::tiles:
            reductions[pointer(args[2])] = {pointer(args[0]), *static_cast<int*>(args[1])};
            break;
        case Kernel::finish:
            winners[static_cast<const char*>(pointer(args[2])) + offsetof(Candidate, token_id)] =
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
    const auto status = copy(dst, src, count, kind, stream);
    if (status == cudaSuccess && kind == cudaMemcpyDeviceToHost && output()) {
        observe_readback(copy, dst, src, count, stream);
    }
    return status;
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    static const auto synchronize = next<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize");
    const auto status = synchronize(stream);
    if (status == cudaSuccess) {
        flush(stream);
    }
    return status;
}
