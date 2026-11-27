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
