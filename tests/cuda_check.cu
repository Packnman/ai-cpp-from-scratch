#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__global__ void add_one(int* value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ++(*value);
    }
}

}  // namespace

int main() {
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::cerr << "CUDA device not found\n";
        return EXIT_FAILURE;
    }

    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    std::cout << "GPU: " << properties.name << '\n'
              << "compute capability: " << properties.major << '.' << properties.minor << '\n'
              << "VRAM: " << properties.totalGlobalMem / (1024 * 1024) << " MiB\n";

    int* value = nullptr;
    check(cudaMallocManaged(&value, sizeof(*value)), "cudaMallocManaged");
    *value = 41;
    add_one<<<1, 1>>>(value);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    const bool passed = *value == 42;
    check(cudaFree(value), "cudaFree");
    std::cout << "CUDA kernel: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
