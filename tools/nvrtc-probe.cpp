#include <array>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

namespace {
constexpr unsigned int element_count = 257;
constexpr unsigned int threads = 128;
constexpr unsigned int blocks = (element_count + threads - 1) / threads;
constexpr std::size_t allocation_bytes = element_count * sizeof(int);
constexpr char kernel_name[] = "fill_sequence";
constexpr char architecture[] = "sm_100";
constexpr char kernel_source[] = R"cuda(
extern "C" __global__ void fill_sequence(int* output, unsigned int count, int seed) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = seed + static_cast<int>(index) + 1;
    }
}
)cuda";

const char* cuda_error(CUresult status) {
    const char* name = nullptr;
    cuGetErrorName(status, &name);
    return name ? name : "unknown CUDA error";
}

void check_cuda(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cuda_error(status));
    }
}

void check_nvrtc(nvrtcResult status, const char* operation) {
    if (status != NVRTC_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + nvrtcGetErrorString(status));
    }
}

class Context {
public:
    explicit Context(CUdevice device) {
        check_cuda(cuCtxCreate(&context_, nullptr, 0, device), "cuCtxCreate");
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    ~Context() {
        if (context_) {
            const auto status = cuCtxDestroy(context_);
            if (status != CUDA_SUCCESS) {
                std::fprintf(stderr, "cuCtxDestroy: %s\n", cuda_error(status));
            }
        }
    }

    void close() {
        check_cuda(cuCtxDestroy(context_), "cuCtxDestroy");
        context_ = nullptr;
    }

private:
    CUcontext context_ = nullptr;
};

class Program {
public:
    Program() {
        check_nvrtc(nvrtcCreateProgram(&program_, kernel_source, "fill_sequence.cu",
                                       0, nullptr, nullptr), "nvrtcCreateProgram");
    }

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    ~Program() {
        if (program_) {
            const auto status = nvrtcDestroyProgram(&program_);
            if (status != NVRTC_SUCCESS) {
                std::fprintf(stderr, "nvrtcDestroyProgram: %s\n", nvrtcGetErrorString(status));
            }
        }
    }

    nvrtcProgram get() const { return program_; }

    void close() {
        check_nvrtc(nvrtcDestroyProgram(&program_), "nvrtcDestroyProgram");
        program_ = nullptr;
    }

private:
    nvrtcProgram program_ = nullptr;
};

std::string compiler_log(nvrtcProgram program) {
    std::size_t size = 0;
    if (nvrtcGetProgramLogSize(program, &size) != NVRTC_SUCCESS || size == 0) {
        return {};
    }
    std::string log(size, '\0');
    if (nvrtcGetProgramLog(program, log.data()) != NVRTC_SUCCESS) {
        return {};
    }
    log.pop_back();
    return log;
}

struct Compilation {
    std::vector<char> cubin;
    double milliseconds;
};

Compilation compile_kernel() {
    Program program;
    const std::string target = std::string("--gpu-architecture=") + architecture;
    const char* options[] = {target.c_str(), "--no-cache"};
    const auto start = std::chrono::steady_clock::now();
    const auto status = nvrtcCompileProgram(program.get(), 2, options);
    const auto end = std::chrono::steady_clock::now();
    if (status != NVRTC_SUCCESS) {
        const auto log = compiler_log(program.get());
        if (!log.empty()) {
            std::fprintf(stderr, "%s\n", log.c_str());
        }
        check_nvrtc(status, "nvrtcCompileProgram");
    }

    std::size_t size = 0;
    check_nvrtc(nvrtcGetCUBINSize(program.get(), &size), "nvrtcGetCUBINSize");
    if (size == 0) {
        throw std::runtime_error("NVRTC produced an empty cubin");
    }
    std::vector<char> cubin(size);
    check_nvrtc(nvrtcGetCUBIN(program.get(), cubin.data()), "nvrtcGetCUBIN");
    program.close();
    return {std::move(cubin), std::chrono::duration<double, std::milli>(end - start).count()};
}

struct KernelLookup {
    CUfunction function;
    bool hit;
    double compile_ms;
};

class ProbeKernel {
public:
    KernelLookup get() {
        if (function_) {
            return {function_, true, 0.0};
        }
        const auto compiled = compile_kernel();
        CUmodule module = nullptr;
        CUfunction function = nullptr;
        check_cuda(cuModuleLoadData(&module, compiled.cubin.data()), "cuModuleLoadData");
        check_cuda(cuModuleGetFunction(&function, module, kernel_name), "cuModuleGetFunction");
        check_cuda(cuFuncLoad(function), "cuFuncLoad");
        function_ = function;
        return {function_, false, compiled.milliseconds};
    }

private:
    CUfunction function_ = nullptr;
};

std::string uuid_string(const CUuuid& uuid) {
    constexpr char hex[] = "0123456789abcdef";
    std::string value = "GPU-";
    for (std::size_t index = 0; index < sizeof(uuid.bytes); ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) {
            value += '-';
        }
        const auto byte = static_cast<unsigned char>(uuid.bytes[index]);
        value += hex[byte >> 4];
        value += hex[byte & 15];
    }
    return value;
}

int device_attribute(CUdevice device, CUdevice_attribute attribute) {
    int value = 0;
    check_cuda(cuDeviceGetAttribute(&value, attribute, device), "cuDeviceGetAttribute");
    return value;
}

CUdevice select_device() {
    check_cuda(cuInit(0), "cuInit");
    int count = 0;
    check_cuda(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (count != 1) {
        throw std::runtime_error("Expected one visible GPU. "
                                 "Use tools/build.sh --nvrtc --gpu INDEX");
    }
    CUdevice device = 0;
    check_cuda(cuDeviceGet(&device, 0), "cuDeviceGet");
    return device;
}

void report_device(CUdevice device) {
    char name[128]{};
    char pci_bus_id[32]{};
    CUuuid uuid{};
    std::size_t total_bytes = 0;
    int driver_version = 0;
    int nvrtc_major = 0;
    int nvrtc_minor = 0;
    check_cuda(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName");
    check_cuda(cuDeviceGetUuid(&uuid, device), "cuDeviceGetUuid");
    check_cuda(cuDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device), "cuDeviceGetPCIBusId");
    check_cuda(cuDeviceTotalMem(&total_bytes, device), "cuDeviceTotalMem");
    check_cuda(cuDriverGetVersion(&driver_version), "cuDriverGetVersion");
    check_nvrtc(nvrtcVersion(&nvrtc_major, &nvrtc_minor), "nvrtcVersion");
    const int major = device_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    const int minor = device_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    const int sms = device_attribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    std::printf("device_name=\"%s\" uuid=%s pci_bus_id=%s compute_capability=%d.%d "
                "sm_count=%d total_memory_bytes=%zu driver_cuda_version=%d nvrtc_version=%d.%d\n",
                name, uuid_string(uuid).c_str(), pci_bus_id, major, minor, sms, total_bytes,
                driver_version, nvrtc_major, nvrtc_minor);
    if (major != 10 || minor != 0) {
        throw std::runtime_error("The NVRTC probe requires compute capability 10.0");
    }
}

void launch_and_verify(KernelLookup kernel, CUdeviceptr output, CUstream stream,
                       CUevent start, CUevent done, bool expect_hit) {
    if (kernel.hit != expect_hit) {
        throw std::runtime_error("Unexpected probe kernel cache status");
    }
    unsigned int count = element_count;
    int seed = 41;
    void* arguments[] = {&output, &count, &seed};
    check_cuda(cuMemsetD32Async(output, 0, element_count, stream), "cuMemsetD32Async");
    check_cuda(cuEventRecord(start, stream), "cuEventRecord(start)");
    check_cuda(cuLaunchKernel(kernel.function, blocks, 1, 1, threads, 1, 1, 0, stream,
                              arguments, nullptr), "cuLaunchKernel");
    check_cuda(cuEventRecord(done, stream), "cuEventRecord(done)");
    check_cuda(cuEventSynchronize(done), "cuEventSynchronize");
    float kernel_ms = 0;
    check_cuda(cuEventElapsedTime(&kernel_ms, start, done), "cuEventElapsedTime");

    std::array<int, element_count> actual{};
    check_cuda(cuMemcpyDtoH(actual.data(), output, allocation_bytes), "cuMemcpyDtoH");
    for (unsigned int index = 0; index < element_count; ++index) {
        const int expected = 42 + static_cast<int>(index);
        if (actual[index] != expected) {
            throw std::runtime_error("Mismatch at index " + std::to_string(index) + ": got " +
                                     std::to_string(actual[index]) + ", expected " +
                                     std::to_string(expected));
        }
    }
    std::printf("phase=%s cache=%s compiled=%d compile_ms=%.6f kernel_ms=%.6f "
                "verified_elements=%u first=%d last=%d\n",
                expect_hit ? "warm" : "cold", kernel.hit ? "hit" : "miss", !kernel.hit,
                kernel.compile_ms, kernel_ms, element_count, actual.front(), actual.back());
}

void run_probe() {
    const auto device = select_device();
    report_device(device);
    Context context(device);
    {
        CUdeviceptr output = 0;
        CUstream stream = nullptr;
        CUevent start = nullptr;
        CUevent done = nullptr;
        check_cuda(cuMemAlloc(&output, allocation_bytes), "cuMemAlloc");
        check_cuda(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        check_cuda(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
        check_cuda(cuEventCreate(&done, CU_EVENT_DEFAULT), "cuEventCreate(done)");
        std::printf("kernel=%s architecture=%s allocation_bytes=%zu "
                    "cache_scope=process_context nvrtc_persistent_cache=disabled\n",
                    kernel_name, architecture, allocation_bytes);
        ProbeKernel cache;
        launch_and_verify(cache.get(), output, stream, start, done, false);
        launch_and_verify(cache.get(), output, stream, start, done, true);
    }
    context.close();
}
}

int main() {
    try {
        run_probe();
        std::puts("NVRTC probe passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "NVRTC probe failed: %s\n", error.what());
        return 1;
    }
}
