#include <cupti.h>
#include <cupti_activity.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

void complete_buffer(CUcontext context, std::uint32_t stream, std::uint8_t* buffer,
                     std::size_t, std::size_t valid_size) {
    flockfile(trace);
    if (valid_size) {
        CUpti_Activity* record = nullptr;
        while (true) {
            const auto status = cuptiActivityGetNextRecord(buffer, valid_size, &record);
            if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            }
            check(status);
            if (record->kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE) {
                const auto& graph = *reinterpret_cast<CUpti_ActivityGraphTrace2*>(record);
                std::fprintf(trace, "gpu\t%u\t%u\t%u\t%u\t%u\t%llu\t%llu\n",
                             graph.correlationId, graph.graphId, graph.deviceId, graph.contextId,
                             graph.streamId, static_cast<unsigned long long>(graph.start),
                             static_cast<unsigned long long>(graph.end));
                continue;
            }
            if (record->kind != CUPTI_ACTIVITY_KIND_RUNTIME &&
                record->kind != CUPTI_ACTIVITY_KIND_DRIVER) {
                continue;
            }
            const auto& api = *reinterpret_cast<CUpti_ActivityAPI*>(record);
            const bool runtime = record->kind == CUPTI_ACTIVITY_KIND_RUNTIME;
            const char* name = nullptr;
            check(cuptiGetCallbackName(runtime ? CUPTI_CB_DOMAIN_RUNTIME_API
                                              : CUPTI_CB_DOMAIN_DRIVER_API, api.cbid, &name));
            if (name && (std::strstr(name, "Graph") || std::strstr(name, "Capture") ||
                         (record_kernels && std::strstr(name, "LaunchKernel")))) {
                std::fprintf(trace, "%s\t%s\t%u\t%u\t%llu\t%llu\n",
                             runtime ? "runtime" : "driver", name, api.correlationId,
                             api.returnValue, static_cast<unsigned long long>(api.start),
                             static_cast<unsigned long long>(api.end));
            }
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
    trace = std::fopen(path, "w");
    if (!trace) {
        std::perror(path);
        std::abort();
    }
    check(cuptiActivityRegisterCallbacks(request_buffer, complete_buffer));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE));
    std::atexit(flush);
}

}
