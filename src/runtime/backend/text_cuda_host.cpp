#include "sglang/text/text_cuda_primitives.cuh"

namespace sglang::text_kernel {

bool aligned(const void* pointer, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

int divide_up(int count, int tile) {
    return count / tile + (count % tile != 0);
}

}
