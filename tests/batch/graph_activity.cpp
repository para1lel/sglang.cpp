#include <cupti.h>
#include <cupti_activity.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace {

FILE* trace;
bool record_kernels;

void check(CUptiResult result) {
    if (result != CUPTI_SUCCESS) {
        const char* message = nullptr;
        cuptiGetResultString(result, &message);
        std::fprintf(stderr, "CUPTI activity error: %s\n", message);
        std::abort();
    }
}

void request_buffer(std::uint8_t** buffer, std::size_t* size, std::size_t* limit) {
    *size = 2 * 1024 * 1024;
    *buffer = static_cast<std::uint8_t*>(std::malloc(*size));
    if (!*buffer) {
        std::abort();
    }
    *limit = 0;
}

void record_kernel(const CUpti_ActivityKernel13& kernel) {
    std::ostringstream line;
    line << "node\t" << kernel.graphId << '\t' << kernel.graphNodeId << '\t' << kernel.correlationId
         << '\t' << kernel.deviceId << '\t' << kernel.start << '\t' << kernel.end << '\t'
         << (kernel.name != nullptr ? kernel.name : "(null)") << '\t' << kernel.gridX << '\t'
         << kernel.gridY << '\t' << kernel.gridZ << '\n';
    if (std::fputs(line.str().c_str(), trace) == EOF)
        std::abort();
}

void record_api(const CUpti_ActivityAPI& api, bool runtime) {
    const char* name = nullptr;
    check(cuptiGetCallbackName(runtime ? CUPTI_CB_DOMAIN_RUNTIME_API : CUPTI_CB_DOMAIN_DRIVER_API,
                               api.cbid, &name));
    if (name == nullptr ||
        (std::strstr(name, "Graph") == nullptr && std::strstr(name, "Capture") == nullptr &&
         (!record_kernels || std::strstr(name, "LaunchKernel") == nullptr)))
        return;
    std::ostringstream line;
    line << (runtime ? "runtime" : "driver") << '\t' << name << '\t' << api.correlationId << '\t'
         << api.returnValue << '\t' << api.start << '\t' << api.end << '\n';
    if (std::fputs(line.str().c_str(), trace) == EOF)
        std::abort();
}

void record_graph(const CUpti_ActivityGraphTrace2& graph) {
    std::ostringstream line;
    line << "gpu\t" << graph.correlationId << '\t' << graph.graphId << '\t' << graph.deviceId
         << '\t' << graph.contextId << '\t' << graph.streamId << '\t' << graph.start << '\t'
         << graph.end << '\n';
    if (std::fputs(line.str().c_str(), trace) == EOF)
        std::abort();
}

void complete_buffer(CUcontext context, std::uint32_t stream, std::uint8_t* buffer,
                     std::size_t buffer_size, std::size_t valid_size) {
    if (valid_size > buffer_size)
        std::abort();
    flockfile(trace);
    if (valid_size != 0) {
        CUpti_Activity* record = nullptr;
        while (true) {
            const auto status = cuptiActivityGetNextRecord(buffer, valid_size, &record);
            if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            }
            check(status);
            if (record->kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE) {
                const auto& graph = *reinterpret_cast<CUpti_ActivityGraphTrace2*>(record);
                record_graph(graph);
                continue;
            }
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                const auto& kernel = *reinterpret_cast<CUpti_ActivityKernel13*>(record);
                record_kernel(kernel);
                continue;
            }
            if (record->kind != CUPTI_ACTIVITY_KIND_RUNTIME &&
                record->kind != CUPTI_ACTIVITY_KIND_DRIVER) {
                continue;
            }
            record_api(*reinterpret_cast<CUpti_ActivityAPI*>(record),
                       record->kind == CUPTI_ACTIVITY_KIND_RUNTIME);
        }
    }
    std::size_t dropped = 0;
    check(cuptiActivityGetNumDroppedRecords(context, stream, &dropped));
    if (dropped) {
        std::fprintf(trace, "dropped\t%zu\n", dropped);
    }
    funlockfile(trace);
    std::free(buffer);
}

void flush() {
    check(cuptiActivityFlushAll(0));
    std::fflush(trace);
}

__attribute__((constructor)) void enable() {
    const char* path = std::getenv("BATCH_GRAPH_ACTIVITY");
    if (!path) {
        return;
    }
    record_kernels = std::getenv("BATCH_KERNEL_ACTIVITY") != nullptr;
    const bool record_nodes = std::getenv("BATCH_GRAPH_NODE_ACTIVITY") != nullptr;
    trace = std::fopen(path, "w");
    if (!trace) {
        std::perror(path);
        std::abort();
    }
    check(cuptiActivityRegisterCallbacks(request_buffer, complete_buffer));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    check(cuptiActivityEnable(record_nodes ? CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
                                           : CUPTI_ACTIVITY_KIND_GRAPH_TRACE));
    std::atexit(flush);
}

}

extern "C" std::uint64_t batch_graph_activity_timestamp() {
    std::uint64_t timestamp = 0;
    check(cuptiGetTimestamp(&timestamp));
    return timestamp;
}

extern "C" std::uint32_t batch_graph_activity_device() {
    std::uint32_t device = 0;
    check(cuptiGetDeviceId(nullptr, &device));
    return device;
}
