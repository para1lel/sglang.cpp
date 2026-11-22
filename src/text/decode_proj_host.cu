#include "sglang/text/decode_proj.h"
#include "sglang/text/decode_proj_sm100.cuh"

#include <algorithm>
#include <functional>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

namespace sglang {
namespace decode_proj {
namespace {

constexpr int max_devices = 64;

struct DeviceState {
    std::once_flag once;
    cudaError_t status = cudaSuccess;
    bool supported = false;
    int sm_count = 0;
};

DeviceState device_states[max_devices];
std::once_flag encode_once;
void* encode_entry = nullptr;
cudaError_t encode_status = cudaSuccess;

void query_device(int device, DeviceState& state) {
    int major = 0, minor = 0;
    state.status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (state.status == cudaSuccess) {
        state.status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    }
    if (state.status == cudaSuccess) {
        state.status = cudaDeviceGetAttribute(&state.sm_count, cudaDevAttrMultiProcessorCount,
                                              device);
    }
    state.supported = state.status == cudaSuccess && major == 10 && minor == 0;
}

}

std::optional<cudaError_t> prepare_launch(cudaStream_t stream, LaunchTarget& target) {
    auto status = cudaGetDevice(&target.device);
    if (status != cudaSuccess) {
        return status;
    }
    int stream_device = 0;
    status = cudaStreamGetDevice(stream, &stream_device);
    if (status != cudaSuccess) {
        return status;
    }
    if (stream_device != target.device) {
        return cudaErrorInvalidResourceHandle;
    }
    if (target.device < 0 || target.device >= max_devices) {
        return std::nullopt;
    }
    auto& state = device_states[target.device];
    std::call_once(state.once, query_device, target.device, std::ref(state));
    if (state.status != cudaSuccess) {
        return state.status;
    }
    std::call_once(encode_once, [] {
        encode_status = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &encode_entry,
                                                         12000, cudaEnableDefault, nullptr);
    });
    if (!state.supported || encode_status == cudaErrorNotSupported ||
        (encode_status == cudaSuccess && !encode_entry)) {
        return std::nullopt;
    }
    if (encode_status != cudaSuccess) {
        return encode_status;
    }
    target.sm_count = state.sm_count;
    target.encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(encode_entry);
    return cudaSuccess;
}

bool aligned16(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
}

namespace {

struct PlanEntry {
    const void* kernel;
    int device;
    int row_blocks;
    int k_blocks;
    SplitPlan plan;
};

std::mutex plan_mutex;
std::vector<PlanEntry> plan_cache;

int resident_ctas(const void* kernel, int splits, int shared_bytes, int sm_count) {
    if (splits == 1) {
        int blocks = 0;
        const auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel,
                                                                          threads, shared_bytes);
        return status == cudaSuccess ? blocks * sm_count : 0;
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(splits);
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = shared_bytes;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = splits;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    int clusters = 0;
    const auto status = cudaOccupancyMaxActiveClusters(&clusters, kernel, &config);
    return status == cudaSuccess ? clusters * splits : 0;
}

SplitPlan compute_plan(const void* kernel, const LaunchTarget& target, int row_blocks,
                       int k_blocks, int fixed_bytes, int bytes_per_block, int limit) {
    SplitPlan best{0, 0};
    SplitPlan fallback{0, 0};
    int best_ctas = 0;
    for (int splits = 1; splits <= std::min(max_splits, k_blocks); ++splits) {
        const int shared_bytes = fixed_bytes + (k_blocks + splits - 1) / splits * bytes_per_block;
        if (shared_bytes > limit) {
            continue;
        }
        if (fallback.splits == 0) {
            fallback = {splits, shared_bytes};
        }
        const int ctas = row_blocks * splits;
        if (ctas > best_ctas &&
            ctas <= resident_ctas(kernel, splits, shared_bytes, target.sm_count)) {
            best = {splits, shared_bytes};
            best_ctas = ctas;
        }
    }
    return best.splits != 0 ? best : fallback;
}

}

cudaError_t plan_splits(const void* kernel, int slot, const LaunchTarget& target,
                        int row_blocks, int k_blocks, int fixed_bytes, int bytes_per_block,
                        SplitPlan& plan) {
    std::lock_guard lock(plan_mutex);
    for (const auto& entry : plan_cache) {
        if (entry.kernel == kernel && entry.device == target.device &&
            entry.row_blocks == row_blocks && entry.k_blocks == k_blocks) {
            plan = entry.plan;
            return cudaSuccess;
        }
    }
    int limit = 0;
    const auto status = configure(kernel, slot, target.device, limit);
    if (status != cudaSuccess) {
        return status;
    }
    plan = compute_plan(kernel, target, row_blocks, k_blocks, fixed_bytes, bytes_per_block, limit);
    plan_cache.push_back({kernel, target.device, row_blocks, k_blocks, plan});
    return cudaSuccess;
}

CUresult encode_map(const LaunchTarget& target, CUtensorMap& map, const void* data,
                    CUtensorMapDataType type, int element_bytes, int rows, int width,
                    int box_rows) {
    const cuuint64_t dimensions[] = {static_cast<cuuint64_t>(width), static_cast<cuuint64_t>(rows)};
    const cuuint64_t stride[] = {static_cast<cuuint64_t>(width) * element_bytes};
    const cuuint32_t box[] = {static_cast<cuuint32_t>(128 / element_bytes),
                              static_cast<cuuint32_t>(box_rows)};
    const cuuint32_t element_stride[] = {1, 1};
    return target.encode(&map, type, 2, const_cast<void*>(data), dimensions, stride, box,
                         element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

cudaError_t launch_split(const void* kernel, dim3 grid, int shared_bytes, void** args,
                         cudaStream_t stream, unsigned cluster) {
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = shared_bytes;
    config.stream = stream;
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = 1;
    attributes[1].id = cudaLaunchAttributeClusterDimension;
    attributes[1].val.clusterDim.x = cluster;
    attributes[1].val.clusterDim.y = 1;
    attributes[1].val.clusterDim.z = 1;
    config.attrs = attributes;
    config.numAttrs = cluster > 1 ? 2 : 1;
    return cudaLaunchKernelExC(&config, kernel, args);
}

cudaError_t configure(const void* kernel, int slot, int device, int& limit) {
    static std::mutex mutex;
    static int limits[max_devices][15] = {};
    std::lock_guard lock(mutex);
    if (limits[device][slot] > 0) {
        limit = limits[device][slot];
        return cudaSuccess;
    }
    int optin = 0;
    auto status = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaFuncAttributes attributes{};
    if (status == cudaSuccess) {
        status = cudaFuncGetAttributes(&attributes, kernel);
    }
    limit = optin - static_cast<int>(attributes.sharedSizeBytes);
    if (status == cudaSuccess) {
        status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, limit);
    }
    if (status == cudaSuccess) {
        limits[device][slot] = limit;
    }
    return status;
}

}

}
