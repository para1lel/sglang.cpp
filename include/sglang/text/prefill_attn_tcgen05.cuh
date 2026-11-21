#pragma once

#include <cstdint>

namespace sglang {
namespace prefill_attn {

using Bits = std::uint32_t;

__device__ __forceinline__ Bits shared_address(const void* pointer) {
    return static_cast<Bits>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void barrier_init(Bits barrier, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(barrier), "r"(count) : "memory");
}

__device__ __forceinline__ void barrier_arrive(Bits barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(barrier) : "memory");
}

__device__ __forceinline__ void barrier_wait(Bits barrier, Bits parity) {
    asm volatile("{ .reg .pred ready;\n"
                 "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1;\n"
                 "@!ready bra.uni WAIT; }"
                 :: "r"(barrier), "r"(parity) : "memory");
}

__device__ __forceinline__ void tensor_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tensor_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tensor_wait(Bits barrier, Bits parity) {
    barrier_wait(barrier, parity);
    tensor_after_sync();
}

__device__ __forceinline__ void proxy_fence() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void named_sync(int id, int threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ void tensor_alloc(Bits destination, int columns) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 :: "r"(destination), "r"(columns) : "memory");
}

__device__ __forceinline__ void tensor_dealloc(Bits address, int columns) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(address), "r"(columns) : "memory");
}

// Shared-memory matrix descriptor for 128-byte swizzled operands (SM100 version 1).
__device__ __forceinline__ std::uint64_t smem_descriptor(const void* pointer, Bits leading) {
    constexpr auto swizzle = std::uint64_t{2} << 61;
    constexpr auto version = std::uint64_t{1} << 46;
    constexpr auto stride = std::uint64_t{1024 >> 4} << 32;
    return swizzle | version | stride | (std::uint64_t{leading} << 16) |
           (shared_address(pointer) >> 4);
}

// kind::f16 instruction descriptor: FP32 accumulate, BF16 A and B, M = 128.
template <int Columns, bool ValueMajor>
__host__ __device__ constexpr Bits mma_instruction() {
    constexpr Bits format = (1u << 4) | (1u << 7) | (1u << 10);
    return (Bits{128 / 16} << 24) | (Bits{Columns / 8} << 17) |
           (Bits{ValueMajor} << 16) | format;
}

__device__ __forceinline__ void mma_shared(Bits destination, std::uint64_t lhs,
                                           std::uint64_t rhs, Bits instruction,
                                           bool accumulate) {
    asm volatile("{ .reg .pred accum;\n"
                 "setp.ne.u32 accum, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accum; }"
                 :: "r"(destination), "l"(lhs), "l"(rhs), "r"(instruction),
                    "r"(static_cast<Bits>(accumulate)) : "memory");
}

__device__ __forceinline__ void mma_tensor(Bits destination, Bits lhs, std::uint64_t rhs,
                                           Bits instruction, bool accumulate) {
    asm volatile("{ .reg .pred accum;\n"
                 "setp.ne.u32 accum, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, accum; }"
                 :: "r"(destination), "r"(lhs), "l"(rhs), "r"(instruction),
                    "r"(static_cast<Bits>(accumulate)) : "memory");
}

__device__ __forceinline__ void mma_commit(Bits barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(barrier) : "memory");
}

__device__ __forceinline__ void tensor_load(Bits address, float* values) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,"
                 "%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
                 "%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                 : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]),
                   "=f"(values[4]), "=f"(values[5]), "=f"(values[6]), "=f"(values[7]),
                   "=f"(values[8]), "=f"(values[9]), "=f"(values[10]), "=f"(values[11]),
                   "=f"(values[12]), "=f"(values[13]), "=f"(values[14]), "=f"(values[15]),
                   "=f"(values[16]), "=f"(values[17]), "=f"(values[18]), "=f"(values[19]),
                   "=f"(values[20]), "=f"(values[21]), "=f"(values[22]), "=f"(values[23]),
                   "=f"(values[24]), "=f"(values[25]), "=f"(values[26]), "=f"(values[27]),
                   "=f"(values[28]), "=f"(values[29]), "=f"(values[30]), "=f"(values[31])
                 : "r"(address) : "memory");
}

__device__ __forceinline__ void tensor_load_wait() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tensor_store(Bits address, const Bits* bits) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,"
                 "%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,"
                 "%25,%26,%27,%28,%29,%30,%31,%32};"
                 :: "r"(address), "r"(bits[0]), "r"(bits[1]), "r"(bits[2]), "r"(bits[3]),
                    "r"(bits[4]), "r"(bits[5]), "r"(bits[6]), "r"(bits[7]), "r"(bits[8]),
                    "r"(bits[9]), "r"(bits[10]), "r"(bits[11]), "r"(bits[12]),
                    "r"(bits[13]), "r"(bits[14]), "r"(bits[15]), "r"(bits[16]),
                    "r"(bits[17]), "r"(bits[18]), "r"(bits[19]), "r"(bits[20]),
                    "r"(bits[21]), "r"(bits[22]), "r"(bits[23]), "r"(bits[24]),
                    "r"(bits[25]), "r"(bits[26]), "r"(bits[27]), "r"(bits[28]),
                    "r"(bits[29]), "r"(bits[30]), "r"(bits[31]) : "memory");
}

__device__ __forceinline__ void tensor_store_release() {
    asm volatile("tcgen05.wait::st.sync.aligned;\n"
                 "tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ float fast_exp2(float value) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ Bits pack_bf16(float low, float high) {
    Bits packed;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed) : "f"(low), "f"(high));
    return packed;
}

}
}
