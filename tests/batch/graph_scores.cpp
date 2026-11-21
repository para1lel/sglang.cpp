#include "sglang/decoder_internal.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <array>
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

enum class Kernel { other, tiles, finish, packed_tiles, packed_finish };

constexpr int capacity = sglang::decoder_batch_max_lanes;
using Candidate = sglang::TextLogitCandidate;

struct PackedBindings {
    struct Lane {
        int position;
        sglang::Mxfp8PageTable pages;
        std::array<sglang::Mxfp8KvCache, sglang::decoder::full_attention_layers> caches;
    };

    int active;
    std::array<int, capacity> tokens;
    std::array<Lane, capacity> lanes;
};

struct Logits {
    const void* device;
    int count;
};

struct Pending {
    const int* winner;
    int count;
    int packed_row;
};

struct PackedHead {
    Logits logits;
    const void* bindings;
    const Candidate* winners;
    unsigned rows;
};

thread_local std::unordered_map<const void*, Kernel> kernels;
thread_local std::unordered_map<cudaKernel_t, Kernel> packed_kernels;
thread_local std::unordered_map<const void*, Logits> reductions;
thread_local std::unordered_map<const void*, Logits> winners;
thread_local std::unordered_map<const void*, Logits> packed_reductions;
thread_local std::unordered_map<cudaGraph_t, PackedHead> captured_graphs;
thread_local std::unordered_map<cudaGraphExec_t, PackedHead> executables;
thread_local PackedHead captured{}, submitted{};
thread_local cudaStream_t capture_stream = nullptr, packed_stream = nullptr;
thread_local bool packed_replay = false;
thread_local PackedBindings* bindings = nullptr;
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
        std::strstr(name, "argmax_finish_kernel") ? Kernel::finish :
        std::strstr(name, "packed_argmax_tiles") ? Kernel::packed_tiles :
        std::strstr(name, "packed_argmax_finish") ? Kernel::packed_finish : Kernel::other;
    kernels.emplace(function, kind);
    return kind;
}

const void* pointer(void* argument) {
    const void* value = nullptr;
    std::memcpy(&value, argument, sizeof(value));
    return value;
}

void flush(cudaStream_t stream) {
    const bool has_packed = submitted.rows && stream == packed_stream;
    if (has_packed) {
        static FILE* packed_output = open_output("BATCH_PACKED_OUTPUT", "w");
        if (!packed_output || !bindings || pending.empty() || stream != pending_stream) {
            std::abort();
        }
        std::fprintf(packed_output, "%d %u %d", packed_replay, submitted.rows, bindings->active);
        for (unsigned row = 0; row < submitted.rows; ++row) {
            std::fprintf(packed_output, " %d", bindings->lanes[row].position);
        }
        for (const auto& item : pending) {
            std::fprintf(packed_output, " %d", item.packed_row);
        }
        std::fputc('\n', packed_output);
        std::fflush(packed_output);
        submitted = {};
    }
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

using LaunchKernel = cudaError_t (*)(cudaKernel_t, dim3, dim3, void**, std::size_t, cudaStream_t);

void packed_launch(Kernel kind, dim3 grid, void** args, cudaStream_t stream) {
    if (kind == Kernel::packed_tiles) {
        packed_reductions[pointer(args[4])] = {pointer(args[0]), *static_cast<int*>(args[1])};
    } else if (kind == Kernel::packed_finish) {
        if (grid.y < 2 || grid.y > capacity) {
            std::abort();
        }
        const PackedHead head{packed_reductions.at(pointer(args[0])), pointer(args[2]),
                              static_cast<const Candidate*>(pointer(args[3])), grid.y};
        if (stream == capture_stream && capture_stream) {
            captured = head;
        } else {
            submitted = head;
            packed_stream = stream;
            packed_replay = false;
        }
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
        case Kernel::packed_tiles:
        case Kernel::packed_finish:
            packed_launch(kind, grid, args, stream);
            break;
        case Kernel::other:
            break;
        }
    }
    return launch(function, grid, block, args, shared, stream);
}

extern "C" cudaError_t __cudaGetKernel(cudaKernel_t* kernel, const void* function) {
    using GetKernel = cudaError_t (*)(cudaKernel_t*, const void*);
    static const auto get = next<GetKernel>("__cudaGetKernel");
    const auto status = get(kernel, function);
    if (output() && status == cudaSuccess) {
        const auto kind = identify(function);
        if (kind == Kernel::packed_tiles || kind == Kernel::packed_finish) {
            packed_kernels.emplace(*kernel, kind);
        }
    }
    return status;
}

extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block, void** args,
                                         std::size_t shared, cudaStream_t stream) {
    static const auto launch = next<LaunchKernel>("__cudaLaunchKernel");
    const auto found = packed_kernels.find(kernel);
    if (found != packed_kernels.end()) {
        packed_launch(found->second, grid, args, stream);
    }
    return launch(kernel, grid, block, args, shared, stream);
}

extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
    static const auto begin = next<decltype(&cudaStreamBeginCapture)>("cudaStreamBeginCapture");
    const auto status = begin(stream, mode);
    if (output() && status == cudaSuccess) {
        capture_stream = stream;
        captured = {};
    }
    return status;
}

extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
    static const auto end = next<decltype(&cudaStreamEndCapture)>("cudaStreamEndCapture");
    const auto status = end(stream, graph);
    if (capture_stream == stream) {
        if (status == cudaSuccess && captured.rows) {
            captured_graphs[*graph] = captured;
        }
        capture_stream = nullptr;
        captured = {};
    }
    return status;
}

extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t* executable, cudaGraph_t graph,
                                            unsigned long long flags) {
    static const auto instantiate = next<decltype(&cudaGraphInstantiate)>("cudaGraphInstantiate");
    const auto status = instantiate(executable, graph, flags);
    const auto found = captured_graphs.find(graph);
    if (status == cudaSuccess && found != captured_graphs.end()) {
        executables[*executable] = found->second;
    }
    return status;
}

extern "C" cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    static const auto destroy = next<decltype(&cudaGraphDestroy)>("cudaGraphDestroy");
    captured_graphs.erase(graph);
    return destroy(graph);
}

extern "C" cudaError_t cudaGraphExecDestroy(cudaGraphExec_t executable) {
    static const auto destroy = next<decltype(&cudaGraphExecDestroy)>("cudaGraphExecDestroy");
    executables.erase(executable);
    return destroy(executable);
}

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t stream) {
    static const auto launch = next<decltype(&cudaGraphLaunch)>("cudaGraphLaunch");
    const auto status = launch(executable, stream);
    const auto found = executables.find(executable);
    if (status == cudaSuccess && found != executables.end()) {
        submitted = found->second;
        packed_stream = stream;
        packed_replay = true;
    }
    return status;
}

extern "C" cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count,
                                      cudaMemcpyKind kind, cudaStream_t stream) {
    static const auto copy = next<decltype(&cudaMemcpyAsync)>("cudaMemcpyAsync");
    if (output() && kind == cudaMemcpyDeviceToHost && count == sizeof(int)) {
        static const auto allocate = next<decltype(&cudaMallocHost)>("cudaMallocHost");
        Logits logits{};
        int packed_row = -1;
        if (submitted.rows && packed_stream == stream) {
            const auto first = reinterpret_cast<std::uintptr_t>(submitted.winners) +
                offsetof(Candidate, token_id);
            const auto address = reinterpret_cast<std::uintptr_t>(src);
            if (address >= first && address - first < capacity * sizeof(Candidate) &&
                (address - first) % sizeof(Candidate) == 0) {
                packed_row = (address - first) / sizeof(Candidate);
                logits = submitted.logits;
                logits.device = static_cast<const unsigned short*>(logits.device) +
                    static_cast<std::size_t>(packed_row) * logits.count;
            }
            if (pending.empty()) {
                if (!bindings && allocate(reinterpret_cast<void**>(&bindings), sizeof(*bindings))
                    != cudaSuccess) {
                    std::abort();
                }
                if (copy(bindings, submitted.bindings, sizeof(*bindings), kind, stream)
                    != cudaSuccess) {
                    std::abort();
                }
            }
        }
        const auto found = winners.find(src);
        if (packed_row == -1 && found != winners.end()) {
            logits = found->second;
        }
        if (logits.device) {
            const auto [device, vocab] = logits;
            if (pending.size() == capacity || (!pending.empty() && pending_stream != stream) ||
                vocab != sglang::decoder::vocab_size) {
                std::abort();
            }
            auto*& row = rows[pending.size()];
            if (!row) {
                if (allocate(reinterpret_cast<void**>(&row), vocab * sizeof(*row)) != cudaSuccess) {
                    std::abort();
                }
            }
            if (copy(row, device, vocab * sizeof(*row), kind, stream) != cudaSuccess) {
                std::abort();
            }
            pending.push_back({static_cast<const int*>(dst), vocab, packed_row});
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
