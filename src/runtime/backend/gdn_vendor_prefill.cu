#include "sglang/runtime/backend/gdn_vendor_prefill.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/text/gdn_local.h"
#include "sglang/text/gdn/gdn_prefill_kernels.cuh"
#include "sglang/text/gdn/gdn_finish_history.cuh"
#include "sglang/text/text_cuda_primitives.cuh"

#include <cuda_fp8.h>

#include <tvm/ffi/c_api.h>

#include <atomic>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <string>

extern "C" int vendor_aot_entry(void* handle, const TVMFFIAny* args, int32_t num_args,
                                TVMFFIAny* result)
    asm("__tvm_ffi_gdn_blackwell_prefill_sm_100a_148_bfloat16_float32_8_24_0_1_1_0_0_"
        "int32_none_none_none_none_1");

namespace sglang {
namespace decoder {
namespace {

using Bf16 = __nv_bfloat16;

constexpr int vendor_sms = 148;
constexpr int vendor_workspace_bytes_fixed = 75776;

std::size_t vendor_workspace_bytes(int rows) {
    const std::size_t fixed = static_cast<std::size_t>(rows) * 16576 + 16;
    return fixed + 512 + vendor_workspace_bytes_fixed;
}

bool vendor_device_supported(int device) {
    constexpr int max_devices = 64;
    static std::atomic<unsigned char> cached[max_devices];
    if (device < 0 || device >= max_devices) {
        return false;
    }
    const unsigned char state = cached[device].load(std::memory_order_relaxed);
    if (state != 0) {
        return state == 2;
    }
    int major = 0;
    int minor = 0;
    int sms = 0;
    const bool supported =
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) ==
            cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) ==
            cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
        major == 10 && minor == 0 && sms == vendor_sms;
    cached[device].store(supported ? 2 : 1, std::memory_order_relaxed);
    return supported;
}

template <typename S>
__global__ void gdn_vendor_prepare_kernel(const Bf16* qkv, const Bf16* history,
                                          TextGdnWeights weights, const Bf16* a,
                                          const Bf16* b, int rows, Bf16* q, Bf16* k,
                                          Bf16* v, float* gate, float* beta,
                                          float2* gates, int* cu_seqlens) {
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    constexpr int blocks_per_group = conv_blocks_per_token + 1;
    constexpr int warps = conv_threads / warp_size;
    const int first = blockIdx.x / blocks_per_group * prefill_rows_per_block;
    const int block = blockIdx.x % blocks_per_group;
    const int feature = threadIdx.x;
    if (block == conv_blocks_per_token) {
        if (feature < S::value_heads) {
            const float negative_a = -expf(__bfloat162float(weights.a_log[feature]));
            const float dt_bias = __bfloat162float(weights.dt_bias[feature]);
            #pragma unroll
            for (int row = 0; row < prefill_rows_per_block; ++row) {
                const int index = (first + row) * S::value_heads + feature;
                const float decay =
                    expf(negative_a * softplus(__bfloat162float(a[index]) + dt_bias));
                const float activation = sigmoid(__bfloat162float(b[index]));
                gate[index] = decay;
                beta[index] = activation;
                gates[index] = make_float2(decay, activation);
            }
        }
        if (first == 0 && feature == 0) {
            cu_seqlens[0] = 0;
            cu_seqlens[1] = rows;
        }
        return;
    }

    const int channel = block * conv_threads + feature;
    const auto* taps = weights.convolution + channel * conv_taps;
    const float weight0 = __bfloat162float(taps[0]);
    const float weight1 = __bfloat162float(taps[1]);
    const float weight2 = __bfloat162float(taps[2]);
    const float weight3 = __bfloat162float(taps[3]);
    float oldest = __bfloat162float(raw_sample<S>(qkv, history, first - 3, channel));
    float middle = __bfloat162float(raw_sample<S>(qkv, history, first - 2, channel));
    float latest = __bfloat162float(raw_sample<S>(qkv, history, first - 1, channel));
    Bf16 mixed[prefill_rows_per_block];
    #pragma unroll
    for (int row = 0; row < prefill_rows_per_block; ++row) {
        const float current = __bfloat162float(raw_sample<S>(qkv, history, first + row, channel));
        float value = fmaf(oldest, weight0, 0.0f);
        value = fmaf(middle, weight1, value);
        value = fmaf(latest, weight2, value);
        value = fmaf(current, weight3, value);
        mixed[row] = __float2bfloat16(value * sigmoid(value));
        oldest = middle;
        middle = latest;
        latest = current;
    }
    if (channel < 2 * S::qk_width) {
        __shared__ float warp_squares[prefill_rows_per_block][warps];
        #pragma unroll
        for (int row = 0; row < prefill_rows_per_block; ++row) {
            const float component = __bfloat162float(mixed[row]);
            const float partial = warp_sum(component * component);
            if (feature % warp_size == 0) {
                warp_squares[row][feature / warp_size] = partial;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int row = 0; row < prefill_rows_per_block; ++row) {
            float squares = 0.0f;
            #pragma unroll
            for (int warp = 0; warp < warps; ++warp) {
                squares += warp_squares[row][warp];
            }
            const float component = __bfloat162float(mixed[row]);
            mixed[row] = __float2bfloat16(component * rsqrtf(squares + norm_epsilon));
        }
    }
    #pragma unroll
    for (int row = 0; row < prefill_rows_per_block; ++row) {
        const int token = first + row;
        if (channel < S::qk_width) {
            q[static_cast<std::size_t>(token) * S::qk_width + channel] = mixed[row];
        } else if (channel < 2 * S::qk_width) {
            k[static_cast<std::size_t>(token) * S::qk_width + (channel - S::qk_width)] =
                mixed[row];
        } else {
            v[static_cast<std::size_t>(token) * S::value_width +
              (channel - 2 * S::qk_width)] = mixed[row];
        }
    }
}

__device__ __forceinline__ Bf16 vendor_finish_value(float value, float inverse_rms, Bf16 norm,
                                                    Bf16 z) {
    const float gate = __bfloat162float(z);
    return __float2bfloat16(value * inverse_rms * __bfloat162float(norm)
                            * (gate * sigmoid(gate)));
}

template <typename S>
__global__ void gdn_vendor_finish_kernel(int token_count, const Bf16* qkv, const Bf16* z,
                                         const Bf16* norm, const Bf16* observable,
                                         Bf16* history, Bf16* output) {
    const int norm_blocks = output ? token_count * S::value_heads : 0;
    const int channel = threadIdx.x;
    if (blockIdx.x >= norm_blocks) {
        const int feature = (blockIdx.x - norm_blocks) * text_gdn_head_dim + channel;
        finish_history<S>(token_count, qkv, history, history, feature);
        return;
    }
    __shared__ float warp_squares[text_gdn_head_dim / warp_size];
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * text_gdn_head_dim + channel;
    const float value = __bfloat162float(observable[index]);
    const float partial = warp_sum(value * value);
    if (channel % warp_size == 0) {
        warp_squares[channel / warp_size] = partial;
    }
    __syncthreads();
    float squares = 0.0f;
    #pragma unroll
    for (int warp = 0; warp < text_gdn_head_dim / warp_size; ++warp) {
        squares += warp_squares[warp];
    }
    const float inverse_rms = rsqrtf(squares / text_gdn_head_dim + norm_epsilon);
    output[index] = vendor_finish_value(value, inverse_rms, norm[channel], z[index]);
}

constexpr int finish_heads_per_block = 4;
constexpr int finish_threads = finish_heads_per_block * warp_size;

template <typename S>
__global__ void gdn_vendor_finish_fp8_kernel(int rows, const Bf16* qkv, const Bf16* z,
                                             const Bf16* norm, const Bf16* observable,
                                             Bf16* history, TextFp8Activation output) {
    constexpr int parts = text_gdn_head_dim / warp_size;
    const int norm_blocks = rows * (S::value_heads / finish_heads_per_block);
    if (blockIdx.x >= norm_blocks) {
        const int feature = (blockIdx.x - norm_blocks) * finish_threads + threadIdx.x;
        finish_history<S>(rows, qkv, history, history, feature);
        return;
    }

    const int lane = threadIdx.x % warp_size;
    const int group = blockIdx.x * finish_heads_per_block + threadIdx.x / warp_size;
    const auto offset = static_cast<std::size_t>(group) * text_gdn_head_dim + lane;
    float values[parts];
    float squares = 0.0f;
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        const float value = __bfloat162float(observable[offset + part * warp_size]);
        values[part] = value;
        squares += warp_sum(value * value);
    }
    const float inverse_rms = rsqrtf(squares / text_gdn_head_dim + norm_epsilon);
    float maximum = 0.0f;
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        const int channel = lane + part * warp_size;
        const auto value =
            vendor_finish_value(values[part], inverse_rms, norm[channel],
                                z[offset + part * warp_size]);
        values[part] = __bfloat162float(value);
        maximum = fmaxf(maximum, fabsf(values[part]));
    }
    const float amax = text_kernel::warp_max(maximum);
    float reciprocal = 0.0f;
    if (lane == 0) {
        const float scale =
            __fdiv_rn(fmaxf(text_kernel::quant_epsilon, amax), text_kernel::fp8_max);
        output.scales[group] = scale;
        reciprocal = __fdiv_rn(1.0f, scale);
    }
    reciprocal = __shfl_sync(text_kernel::warp_mask, reciprocal, 0);
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        output.values[offset + part * warp_size] = __nv_cvt_float_to_fp8(
            values[part] * reciprocal, __NV_SATFINITE, __NV_E4M3);
    }
}

void fill_tensor(DLTensor* tensor, void* data, int device, int ndim, uint8_t code,
                 uint8_t bits, int64_t* shape, int64_t* strides) {
    tensor->data = data;
    tensor->device = DLDevice{kDLCUDA, static_cast<int>(device)};
    tensor->ndim = static_cast<int32_t>(ndim);
    tensor->dtype = DLDataType{code, bits, 1};
    tensor->shape = shape;
    tensor->strides = strides;
    tensor->byte_offset = 0;
}

void fill_arg(TVMFFIAny* arg, const DLTensor* tensor) {
    arg->type_index = kTVMFFIDLTensorPtr;
    arg->v_ptr = const_cast<DLTensor*>(tensor);
}

void fill_none(TVMFFIAny* arg) {
    arg->type_index = kTVMFFINone;
    arg->v_ptr = nullptr;
}

struct VendorCarve {
    Bf16* q_plane;
    Bf16* k_plane;
    Bf16* v_plane;
    Bf16* o_plane;
    float* gate;
    float* beta;
    int* cu_seqlens;
    void* descriptor_workspace;
};

template <typename S>
VendorCarve carve_vendor_workspace(Bf16* workspace, int rows) {
    VendorCarve carve;
    carve.q_plane = workspace;
    carve.k_plane = workspace + static_cast<std::size_t>(rows) * S::qk_width;
    carve.v_plane = workspace + static_cast<std::size_t>(rows) * 2 * S::qk_width;
    carve.o_plane =
        workspace + static_cast<std::size_t>(rows) * (2 * S::qk_width + S::value_width);
    carve.gate = reinterpret_cast<float*>(workspace + static_cast<std::size_t>(rows) * 8192);
    carve.beta = carve.gate + static_cast<std::size_t>(rows) * S::value_heads;
    carve.cu_seqlens =
        reinterpret_cast<int*>(carve.beta + static_cast<std::size_t>(rows) * S::value_heads);
    const std::uintptr_t fixed_end = reinterpret_cast<std::uintptr_t>(carve.cu_seqlens) + 16;
    const std::uintptr_t vendor_end = (fixed_end + 255) & ~static_cast<std::uintptr_t>(255);
    carve.descriptor_workspace = reinterpret_cast<void*>(vendor_end);
    return carve;
}

struct FfiGeometry {
    int64_t shape_qk[3];
    int64_t strides_qk[3];
    int64_t shape_vo[3];
    int64_t strides_vo[3];
    int64_t shape_gate[2];
    int64_t strides_gate[2];
    int64_t shape_cu[1];
    int64_t strides_cu[1];
    int64_t shape_state[4];
    int64_t strides_state[4];
    int64_t shape_ws[1];
    int64_t strides_ws[1];
};

template <typename S>
void fill_ffi_tensors(FfiGeometry* geometry, DLTensor* tensors, const VendorCarve& carve,
                      float* recurrent, int device, int rows) {
    geometry->shape_qk[0] = rows;
    geometry->shape_qk[1] = S::qk_heads;
    geometry->shape_qk[2] = text_gdn_head_dim;
    geometry->strides_qk[0] = S::qk_width;
    geometry->strides_qk[1] = text_gdn_head_dim;
    geometry->strides_qk[2] = 1;
    geometry->shape_vo[0] = rows;
    geometry->shape_vo[1] = S::value_heads;
    geometry->shape_vo[2] = text_gdn_head_dim;
    geometry->strides_vo[0] = S::value_width;
    geometry->strides_vo[1] = text_gdn_head_dim;
    geometry->strides_vo[2] = 1;
    geometry->shape_gate[0] = rows;
    geometry->shape_gate[1] = S::value_heads;
    geometry->strides_gate[0] = S::value_heads;
    geometry->strides_gate[1] = 1;
    geometry->shape_cu[0] = 2;
    geometry->strides_cu[0] = 1;
    geometry->shape_state[0] = 1;
    geometry->shape_state[1] = S::value_heads;
    geometry->shape_state[2] = text_gdn_head_dim;
    geometry->shape_state[3] = text_gdn_head_dim;
    geometry->strides_state[0] = S::value_heads * text_gdn_head_dim * text_gdn_head_dim;
    geometry->strides_state[1] = text_gdn_head_dim * text_gdn_head_dim;
    geometry->strides_state[2] = text_gdn_head_dim;
    geometry->strides_state[3] = 1;
    geometry->shape_ws[0] = vendor_workspace_bytes_fixed;
    geometry->strides_ws[0] = 1;

    fill_tensor(&tensors[0], carve.q_plane, device, 3, kDLBfloat, 16, geometry->shape_qk,
                geometry->strides_qk);
    fill_tensor(&tensors[1], carve.k_plane, device, 3, kDLBfloat, 16, geometry->shape_qk,
                geometry->strides_qk);
    fill_tensor(&tensors[2], carve.v_plane, device, 3, kDLBfloat, 16, geometry->shape_vo,
                geometry->strides_vo);
    fill_tensor(&tensors[3], carve.gate, device, 2, kDLFloat, 32, geometry->shape_gate,
                geometry->strides_gate);
    fill_tensor(&tensors[4], carve.beta, device, 2, kDLFloat, 32, geometry->shape_gate,
                geometry->strides_gate);
    fill_tensor(&tensors[5], carve.o_plane, device, 3, kDLBfloat, 16, geometry->shape_vo,
                geometry->strides_vo);
    fill_tensor(&tensors[6], carve.cu_seqlens, device, 1, kDLInt, 32, geometry->shape_cu,
                geometry->strides_cu);
    fill_tensor(&tensors[7], recurrent, device, 4, kDLFloat, 32, geometry->shape_state,
                geometry->strides_state);
    fill_tensor(&tensors[8], recurrent, device, 4, kDLFloat, 32, geometry->shape_state,
                geometry->strides_state);
    fill_tensor(&tensors[9], carve.descriptor_workspace, device, 1, kDLInt, 8, geometry->shape_ws,
                geometry->strides_ws);
}

cudaError_t launch_vendor_recurrence(const DLTensor* tensors, cudaStream_t stream) {
    TVMFFIAny args[19];
    fill_arg(&args[0], &tensors[0]);
    fill_arg(&args[1], &tensors[1]);
    fill_arg(&args[2], &tensors[2]);
    fill_arg(&args[3], &tensors[3]);
    fill_arg(&args[4], &tensors[4]);
    fill_arg(&args[5], &tensors[5]);
    fill_arg(&args[6], &tensors[6]);
    fill_arg(&args[7], &tensors[7]);
    fill_arg(&args[8], &tensors[8]);
    for (int index = 9; index < 15; ++index) {
        fill_none(&args[index]);
    }
    args[15].type_index = kTVMFFIInt;
    args[15].v_int64 = 0;
    args[16].type_index = kTVMFFIFloat;
    args[16].v_float64 = 0.08838834764831845;
    fill_arg(&args[17], &tensors[9]);
    args[18].type_index = kTVMFFIOpaquePtr;
    args[18].v_ptr = stream;

    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    result.v_ptr = nullptr;
    const int rc = vendor_aot_entry(nullptr, args, 19, &result);
    return rc == 0 ? cudaSuccess : cudaErrorUnknown;
}

template <bool Fp8>
cudaError_t gdn_vendor_prefill_run(int rows, const Bf16* qkv, const Bf16* z, const Bf16* b,
                                   const Bf16* a, TextGdnWeights weights, TextGdnState state,
                                   Bf16* workspace, std::size_t workspace_bytes, float2* gates,
                                   void* output, cudaStream_t stream) {
    using S = GdnLocal;
    if (rows == 0) {
        return cudaSuccess;
    }
    if (rows < 0 || rows > text_attention_max_tokens || rows % prefill_rows_per_block != 0 ||
        !qkv || !z || !b || !a || !gates || !output || !workspace || !state.history ||
        !state.recurrent) {
        return cudaErrorInvalidValue;
    }
    if constexpr (Fp8) {
        const auto& activation = *static_cast<const TextFp8Activation*>(output);
        if (!activation.values || !activation.scales) {
            return cudaErrorInvalidValue;
        }
    }
    if (workspace_bytes < vendor_workspace_bytes(rows)) {
        return cudaErrorInvalidValue;
    }
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    if (!vendor_device_supported(device)) {
        return cudaErrorNotSupported;
    }

    const VendorCarve carve = carve_vendor_workspace<S>(workspace, rows);
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    constexpr int blocks_per_group = conv_blocks_per_token + 1;
    const dim3 prepare_grid(static_cast<unsigned>((rows / prefill_rows_per_block) *
                                                  blocks_per_group));
    gdn_vendor_prepare_kernel<S><<<prepare_grid, conv_threads, 0, stream>>>(
        qkv, state.history, weights, a, b, rows, carve.q_plane, carve.k_plane, carve.v_plane,
        carve.gate, carve.beta, gates, carve.cu_seqlens);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    FfiGeometry geometry;
    DLTensor tensors[10];
    fill_ffi_tensors<S>(&geometry, tensors, carve, state.recurrent, device, rows);
    status = launch_vendor_recurrence(tensors, stream);
    if (status != cudaSuccess) {
        return status;
    }

    if (Fp8) {
        const dim3 finish_grid(static_cast<unsigned>(
            rows * (S::value_heads / finish_heads_per_block) +
            S::qkv_width / finish_threads));
        gdn_vendor_finish_fp8_kernel<S><<<finish_grid, finish_threads, 0, stream>>>(
            rows, qkv, z, weights.norm, carve.o_plane, state.history,
            *static_cast<TextFp8Activation*>(output));
    } else {
        const dim3 finish_grid(static_cast<unsigned>(rows * S::value_heads +
                                                     S::qkv_width / text_gdn_head_dim));
        gdn_vendor_finish_kernel<S><<<finish_grid, text_gdn_head_dim, 0, stream>>>(
            rows, qkv, z, weights.norm, carve.o_plane, state.history,
            static_cast<Bf16*>(output));
    }
    return cudaGetLastError();
}

}

bool gdn_vendor_prefill_requested() {
    static const bool requested = [] {
        const char* mode = std::getenv("SGLANG_GDN_VENDOR_PREFILL");
        return mode != nullptr && mode[0] == '1' && mode[1] == '\0';
    }();
    return requested;
}

bool gdn_vendor_prefill_enabled() {
    if (!gdn_vendor_prefill_requested()) {
        return false;
    }
    int device = 0;
    return cudaGetDevice(&device) == cudaSuccess && vendor_device_supported(device);
}

std::string gdn_vendor_prefill_identity_tag(const int* devices, int count) {
    if (!gdn_vendor_prefill_requested() || devices == nullptr || count <= 0) {
        return {};
    }
    std::string mask(static_cast<std::size_t>(count), '0');
    bool supported = false;
    for (int index = 0; index < count; ++index) {
        if (devices[index] >= 0 && vendor_device_supported(devices[index])) {
            mask[static_cast<std::size_t>(index)] = '1';
            supported = true;
        }
    }
    if (!supported) {
        return {};
    }
    std::string tag = "\ngdn-recurrence-vendor-v1 mask=";
    tag += mask;
    tag += '\n';
    return tag;
}

bool gdn_vendor_prefill_eligible_rows(int rows) {
    return rows > 0 && rows <= text_attention_max_tokens &&
           rows % prefill_rows_per_block == 0;
}

std::size_t gdn_vendor_prefill_workspace_bytes(int rows) {
    const std::size_t native = text_gdn_local_workspace_bytes(rows);
    if (rows <= 0 || rows > text_attention_max_tokens) {
        return 0;
    }
    const std::size_t vendor = vendor_workspace_bytes(rows);
    return native > vendor ? native : vendor;
}

cudaError_t gdn_vendor_prefill_local(int rows, const Bf16* qkv, const Bf16* z, const Bf16* b,
                                     const Bf16* a, TextGdnWeights weights, TextGdnState state,
                                     Bf16* workspace, std::size_t workspace_bytes, float2* gates,
                                     Bf16* output, cudaStream_t stream) {
    return gdn_vendor_prefill_run<false>(rows, qkv, z, b, a, weights, state, workspace,
                                         workspace_bytes, gates, output, stream);
}

cudaError_t gdn_vendor_prefill_local_fp8(int rows, const Bf16* qkv, const Bf16* z,
                                         const Bf16* b, const Bf16* a, TextGdnWeights weights,
                                         TextGdnState state, Bf16* workspace,
                                         std::size_t workspace_bytes, float2* gates,
                                         TextFp8Activation output, cudaStream_t stream) {
    return gdn_vendor_prefill_run<true>(rows, qkv, z, b, a, weights, state, workspace,
                                        workspace_bytes, gates, &output, stream);
}

}
}
