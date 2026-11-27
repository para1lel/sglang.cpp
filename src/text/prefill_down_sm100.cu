#include "sglang/decoder_internal.h"
#include "sglang/prefill_fp8.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/cuda_host_adapter.hpp>

namespace {

struct TensorMapError {
    CUresult status;
};

template <class Function, class... Args>
CUresult checked_tensor_map(Function encode, Args... args) {
    const auto status = encode(args...);
    if (status != CUDA_SUCCESS) throw TensorMapError{status};
    return status;
}

}

#undef CUTLASS_CUDA_DRIVER_WRAPPER_CALL
#define CUTLASS_CUDA_DRIVER_WRAPPER_CALL(func) \
    [](auto... args) { \
        return checked_tensor_map(cutlass::call_##func<decltype(args)...>, args...); \
    }
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>
#undef CUTLASS_CUDA_DRIVER_WRAPPER_CALL

namespace sglang {
namespace {

using namespace cute;
constexpr int out_width = 5120;
constexpr int scale_group = 128;
constexpr int scale_tile = 16;
using Fp8 = cutlass::float_e4m3_t;
using Bf16 = cutlass::bfloat16_t;
using Arch = cutlass::arch::Sm100;
using TensorOp = cutlass::arch::OpClassTensorOp;
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;
using Tile = Shape<_256, _128, _128>;
using Cluster = Shape<_2, _1, _1>;
using Scale = cutlass::detail::Sm100BlockwiseScaleConfig<1, scale_group, scale_group,
                                                        UMMA::Major::MN, UMMA::Major::K>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, TensorOp, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, void, RowMajor, 8, Bf16, RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized2Sm>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, TensorOp, Fp8, tuple<RowMajor, decltype(Scale::deduce_layoutSFA())>, 16,
    Fp8, tuple<ColumnMajor, decltype(Scale::deduce_layoutSFB())>, 16, float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(Epilogue::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwise2SmSm100>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

template <int rows, int groups>
__global__ void prepare_scales(const float* activation, const __nv_bfloat16* weight,
                               float* transposed, float* converted) {
    __shared__ float tile[scale_tile][scale_tile + 1];
    const int group_in = blockIdx.x * scale_tile + threadIdx.x;
    const int row_in = blockIdx.y * scale_tile + threadIdx.y;
    constexpr int weight_scale_count = (out_width / scale_group) * groups;
    if (group_in < groups) tile[threadIdx.y][threadIdx.x] = activation[row_in * groups + group_in];
    const int index = (blockIdx.y * gridDim.x + blockIdx.x) * scale_tile * scale_tile +
                      threadIdx.y * scale_tile + threadIdx.x;
    if (index < weight_scale_count) converted[index] = __bfloat162float(weight[index]);
    __syncthreads();
    const int row_out = blockIdx.y * scale_tile + threadIdx.x;
    const int group_out = blockIdx.x * scale_tile + threadIdx.y;
    if (group_out < groups) transposed[group_out * rows + row_out] = tile[threadIdx.x][threadIdx.y];
}

Gemm::Arguments arguments(TextFp8ActivationView input, const std::uint8_t* weight,
                          __nv_bfloat16* output, float* transposed, float* converted,
                          int rows, int in_width, int device) {
    const auto shape = make_shape(rows, out_width, in_width, 1);
    const auto a = cutlass::make_cute_packed_stride(Kernel::StrideA{},
                                                    make_shape(rows, in_width, 1));
    const auto b = cutlass::make_cute_packed_stride(Kernel::StrideB{},
                                                    make_shape(out_width, in_width, 1));
    const auto d = cutlass::make_cute_packed_stride(Kernel::StrideD{},
                                                    make_shape(rows, out_width, 1));
    const Kernel::MainloopArguments matrix{
        reinterpret_cast<const Fp8*>(input.values), a, reinterpret_cast<const Fp8*>(weight), b,
        transposed, Scale::tile_atom_to_shape_SFA(shape),
        converted, Scale::tile_atom_to_shape_SFB(shape)};
    const Kernel::EpilogueArguments epilogue{{}, nullptr, d, reinterpret_cast<Bf16*>(output), d};
    return {cutlass::gemm::GemmUniversalMode::kGemm, shape, matrix, epilogue, {device, 0}, {}};
}

bool aligned(const void* ptr, std::uintptr_t alignment) {
    return ptr && reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

cudaError_t submit_status(cudaError_t direct) {
    const auto recorded = cudaGetLastError();
    if (direct == cudaSuccess || (!decoder::CudaError::terminal(direct) &&
                                  decoder::CudaError::terminal(recorded))) {
        return recorded;
    }
    return direct;
}

std::optional<cudaError_t> prepare_kernel(cudaStream_t stream, int& device) {
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    int owner = device;
    if (stream) {
        status = cudaStreamGetDevice(stream, &owner);
        if (status != cudaSuccess) return status;
        if (owner != device) return cudaErrorInvalidResourceHandle;
    }
    int major = 0, minor = 0, shared_limit = 0;
    status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (status != cudaSuccess) return status;
    status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (status != cudaSuccess) return status;
    if (major != 10 || minor != 0) return std::nullopt;
    status = cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) return status;
    if (shared_limit < Kernel::SharedStorageSize) return std::nullopt;
    return cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                Kernel::SharedStorageSize);
}

}

std::optional<cudaError_t> try_prefill_scaled_sm100(
    TextFp8ActivationView input, const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
    int row_count, int output_width, int input_width, __nv_bfloat16* output,
    __nv_bfloat16* gate_scratch, __nv_bfloat16* up_scratch, cudaStream_t stream) {
    const auto pending = cudaGetLastError();
    if (pending != cudaSuccess) return pending;
    const bool attention_output = row_count == grouped_prefill_max_rows && input_width == 6144;
    const bool down = (row_count == grouped_prefill_max_rows || row_count == 1024) &&
                      (input_width == 17408 || input_width == 8704);
    if (output_width != out_width || !(attention_output || down)) return std::nullopt;
    if (!aligned(input.values, 16) || !aligned(input.scales, alignof(float)) ||
        !aligned(weight, 16) || !aligned(weight_scales, alignof(__nv_bfloat16)) ||
        !aligned(output, 16) || !aligned(gate_scratch, 16) || !aligned(up_scratch, 16) ||
        gate_scratch == up_scratch) return cudaErrorInvalidValue;
    int device = 0;
    const auto setup = prepare_kernel(stream, device);
    if (const auto status = submit_status(setup.value_or(cudaSuccess)); status != cudaSuccess) {
        return status;
    }
    if (!setup) return std::nullopt;
    auto* transposed = reinterpret_cast<float*>(gate_scratch);
    auto* converted = reinterpret_cast<float*>(up_scratch);
    const auto args = arguments(input, weight, output, transposed, converted, row_count,
                                input_width, device);
    if (Gemm::can_implement(args) != cutlass::Status::kSuccess || Gemm::get_workspace_size(args)) {
        return std::nullopt;
    }
    Kernel::Params params;
    try {
        params = Kernel::to_underlying_arguments(args, nullptr);
    } catch (const TensorMapError& error) {
        return submit_status(decoder::CudaError::from_driver(error.status));
    }
    const int groups = input_width / scale_group;
    dim3 scale_grid((groups + scale_tile - 1) / scale_tile, row_count / scale_tile);
    auto scale_kernel = attention_output ? prepare_scales<4096, 6144 / scale_group> :
        row_count == 1024 ? prepare_scales<1024, 17408 / scale_group> :
                           prepare_scales<4096, 17408 / scale_group>;
    if (input_width == 8704) {
        scale_kernel = row_count == 1024 ? prepare_scales<1024, 8704 / scale_group> :
                                           prepare_scales<4096, 8704 / scale_group>;
    }
    void* scale_args[] = {&input.scales, &weight_scales, &transposed, &converted};
    const auto status = submit_status(cudaLaunchKernel(
        scale_kernel, scale_grid, dim3(scale_tile, scale_tile), scale_args, 0, stream));
    if (status != cudaSuccess) return status;
    cudaLaunchAttribute cluster{};
    cluster.id = cudaLaunchAttributeClusterDimension;
    cluster.val.clusterDim = {2, 1, 1};
    cudaLaunchConfig_t config{Kernel::get_grid_shape(params), Kernel::get_block_shape(),
                              Kernel::SharedStorageSize, stream, &cluster, 1};
    return submit_status(cudaLaunchKernelEx(&config, cutlass::device_kernel<Kernel>, params));
}

}
