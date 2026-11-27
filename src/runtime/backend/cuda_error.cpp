#include "sglang/runtime/backend/cuda_resources.h"

#include <string>

namespace sglang::decoder {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw CudaError(status, operation);
    }
}

CudaError::CudaError(cudaError_t status, const char* operation)
    : std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status)),
      status(status) {}

cudaError_t CudaError::from_driver(CUresult status) noexcept {
    switch (status) {
    case CUDA_SUCCESS:
        return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE:
        return cudaErrorInvalidValue;
    case CUDA_ERROR_OUT_OF_MEMORY:
        return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED:
        return cudaErrorInitializationError;
    case CUDA_ERROR_DEINITIALIZED:
        return cudaErrorCudartUnloading;
    case CUDA_ERROR_INVALID_CONTEXT:
        return cudaErrorIncompatibleDriverContext;
    case CUDA_ERROR_ECC_UNCORRECTABLE:
        return cudaErrorECCUncorrectable;
    case CUDA_ERROR_INVALID_HANDLE:
        return cudaErrorInvalidResourceHandle;
    case CUDA_ERROR_ILLEGAL_ADDRESS:
        return cudaErrorIllegalAddress;
    case CUDA_ERROR_LAUNCH_TIMEOUT:
        return cudaErrorLaunchTimeout;
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:
        return cudaErrorContextIsDestroyed;
    case CUDA_ERROR_ASSERT:
        return cudaErrorAssert;
    case CUDA_ERROR_HARDWARE_STACK_ERROR:
        return cudaErrorHardwareStackError;
    case CUDA_ERROR_ILLEGAL_INSTRUCTION:
        return cudaErrorIllegalInstruction;
    case CUDA_ERROR_MISALIGNED_ADDRESS:
        return cudaErrorMisalignedAddress;
    case CUDA_ERROR_INVALID_ADDRESS_SPACE:
        return cudaErrorInvalidAddressSpace;
    case CUDA_ERROR_INVALID_PC:
        return cudaErrorInvalidPc;
    case CUDA_ERROR_LAUNCH_FAILED:
        return cudaErrorLaunchFailure;
    case CUDA_ERROR_NOT_SUPPORTED:
        return cudaErrorNotSupported;
    default:
        return cudaErrorUnknown;
    }
}

bool CudaError::terminal(cudaError_t error) {
    switch (error) {
        case cudaErrorCudartUnloading:
        case cudaErrorECCUncorrectable:
        case cudaErrorIllegalAddress:
        case cudaErrorLaunchTimeout:
        case cudaErrorContextIsDestroyed:
        case cudaErrorAssert:
        case cudaErrorHardwareStackError:
        case cudaErrorIllegalInstruction:
        case cudaErrorMisalignedAddress:
        case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPc:
        case cudaErrorLaunchFailure:
            return true;
        default:
            return false;
    }
}

}
