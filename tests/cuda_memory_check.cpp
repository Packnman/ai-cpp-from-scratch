#include "cuda_memory.h"
#include "cuda_matrix.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

void require(bool value, const char* message) {
    if(!value) throw std::runtime_error(message);
}
int main() {
    try {
        cu_memory::Buffer empty;
        require(empty.data() == nullptr, "zero allocation");
        { cu_memory::Buffer first(4); }
        const auto initial = cu_memory::statistics();
        const bool pooled = initial.pooled;
        setenv("AI_CPP_CUDA_MEMORY_POOL", pooled ? "0" : "1", 1);
        require(cu_memory::statistics().pooled == pooled, "allocator setting changed");
        {
            cufMat view;
            {
                cufMat source(2, 2);
                const float data[] = {1, 2, 3, 4};
                source.copyFromHost(data, 4);
                view = source.reshape({4, 1});
                source = cufMat(7, 3);
            }
            require(view.toHost() == std::vector<float>({1, 2, 3, 4}), "view lifetime");
            cunMat integers(2, 1);
            const std::int32_t data[] = {-7, 123456};
            integers.copyFromHost(data, 2);
            require(integers.toHost() == std::vector<std::int32_t>({-7, 123456}), "integer buffer");
            cunMat zero(0, 3);
            require(zero.numel() == 0, "zero tensor");
        }
        try {
            cu_memory::Buffer buffer(4096);
            throw std::runtime_error("intentional unwind");
        } catch(const std::runtime_error&) {}
        require(cu_memory::statistics().usedBytes == initial.usedBytes, "exception leaked memory");
        std::uint64_t warmed = 0;
        for(int i = 0; i < 40; ++i) {
            {
                cu_memory::Buffer buffer(1024*1024);
                require(cudaMemset(buffer.data(), 0, 1024*1024) == cudaSuccess, "buffer write");
            }
            const auto stats = cu_memory::statistics();
            require(stats.usedBytes == initial.usedBytes, "allocation not returned");
            if(pooled) {
                require(stats.reservedBytes >= 1024*1024, "pool did not retain reservation");
                if(i == 3) warmed = stats.reservedBytes;
                if(i > 3) require(stats.reservedBytes == warmed, "reservation keeps growing");
            }
        }
        std::cout << "Warmed reserved bytes=" << warmed << '\n';
        cu_memory::trimUnused();
        require(cu_memory::statistics().reservedBytes == 0, "trim did not release unused memory");
        std::cout << "CUDA memory checks passed allocator=" << (pooled ? "pool" : "legacy") << '\n';
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
