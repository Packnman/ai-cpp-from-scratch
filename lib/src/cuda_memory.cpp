#include "cuda_memory.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace cu_memory {
namespace {
void check(cudaError_t error, const char* operation) {
    if(error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
void report(cudaError_t error, const char* operation) noexcept {
    if(error != cudaSuccess)
        std::fprintf(stderr, "CUDA memory: %s: %s\n", operation, cudaGetErrorString(error));
}
class DeviceGuard {
    int previous_;
    bool changed_;
public:
    explicit DeviceGuard(int device) {
        check(cudaGetDevice(&previous_), "cudaGetDevice");
        changed_ = previous_ != device;
        if(changed_) check(cudaSetDevice(device), "cudaSetDevice");
    }
    ~DeviceGuard() noexcept { if(changed_) report(cudaSetDevice(previous_), "restore device"); }
};
bool requested() {
    static const bool enabled = [] {
        const char* value = std::getenv("AI_CPP_CUDA_MEMORY_POOL");
        if(!value || std::string(value) == "1") return true;
        if(std::string(value) == "0") return false;
        throw std::invalid_argument("AI_CPP_CUDA_MEMORY_POOL must be 0 or 1");
    }();
    return enabled;
}
}
struct DevicePool {
    int device;
    cudaMemPool_t handle = nullptr;
    std::atomic<std::size_t> live{0};
    explicit DevicePool(int id) : device(id) {
        if(requested()) {
            int supported = 0;
            auto error = cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device);
            if(error == cudaErrorNotSupported || error == cudaErrorInvalidValue) {
                std::fprintf(stderr, "CUDA memory device=%d: pool unsupported by driver (%s); using legacy\n", device, cudaGetErrorString(error));
            } else {
                check(error, "query memory pool support");
                if(!supported) std::fprintf(stderr, "CUDA memory device=%d: memory pools unsupported; using legacy\n", device);
                else {
                    cudaMemPoolProps props{};
                    props.allocType = cudaMemAllocationTypePinned;
                    props.location.type = cudaMemLocationTypeDevice;
                    props.location.id = device;
                    error = cudaMemPoolCreate(&handle, &props);
                    if(error == cudaErrorNotSupported) {
                        handle = nullptr;
                        std::fprintf(stderr, "CUDA memory device=%d: pool creation unsupported; using legacy\n", device);
                    } else {
                        check(error, "cudaMemPoolCreate");
                        std::uint64_t threshold = UINT64_MAX;
                        error = cudaMemPoolSetAttribute(handle, cudaMemPoolAttrReleaseThreshold, &threshold);
                        if(error != cudaSuccess) {
                            report(cudaMemPoolDestroy(handle), "destroy incomplete pool");
                            handle = nullptr;
                            check(error, "set pool release threshold");
                        }
                    }
                }
            }
        }
        std::fprintf(stderr, "CUDA memory device=%d allocator=%s\n", device, handle ? "pool" : "legacy");
    }
    ~DevicePool() noexcept {
        if(!handle) return;
        try {
            DeviceGuard guard(device);
            check(cudaDeviceSynchronize(), "pool shutdown synchronize");
            if(live != 0) {
                std::fprintf(stderr, "CUDA memory: refusing to destroy pool with live allocations\n");
                return;
            }
            report(cudaMemPoolDestroy(handle), "destroy pool");
        } catch(const std::exception& error) {
            std::fprintf(stderr, "CUDA memory shutdown: %s\n", error.what());
        }
    }
};
namespace {
std::shared_ptr<DevicePool> currentPool(bool create = true) {
    // Initialize CUDA before registering registry destruction with atexit.
    // Otherwise the runtime can shut down before our pools are destroyed.
    static const bool initialized = [] {
        check(cudaFree(nullptr), "initialize CUDA memory runtime");
        return true;
    }();
    (void)initialized;
    static std::mutex mutex;
    static std::map<int, std::shared_ptr<DevicePool>> pools;
    int device;
    check(cudaGetDevice(&device), "cudaGetDevice");
    std::lock_guard<std::mutex> lock(mutex);
    if(!create) {
        auto found = pools.find(device);
        return found == pools.end() ? nullptr : found->second;
    }
    auto& pool = pools[device];
    if(!pool) pool = std::make_shared<DevicePool>(device);
    return pool;
}
}
Buffer::Buffer(std::size_t bytes) {
    if(!bytes) return;
    pool_ = currentPool();
    check(pool_->handle ? cudaMallocFromPoolAsync(&pointer_, bytes, pool_->handle, nullptr)
                        : cudaMalloc(&pointer_, bytes), "CUDA memory allocate");
    ++pool_->live;
}
void Buffer::release() {
    if(!pointer_) return;
    DeviceGuard guard(pool_->device);
    check(pool_->handle ? cudaFreeAsync(pointer_, nullptr) : cudaFree(pointer_), "CUDA memory release");
    pointer_ = nullptr;
    --pool_->live;
}
Buffer::~Buffer() noexcept {
    try { release(); }
    catch(const std::exception& error) { std::fprintf(stderr, "CUDA memory release: %s\n", error.what()); }
}
Statistics statistics() {
    auto pool = currentPool(false);
    check(cudaDeviceSynchronize(), "pool statistics synchronize");
    Statistics result;
    if(!pool) return result;
    result.pooled = pool->handle != nullptr;
    if(result.pooled) {
        check(cudaMemPoolGetAttribute(pool->handle, cudaMemPoolAttrUsedMemCurrent, &result.usedBytes), "pool used bytes");
        check(cudaMemPoolGetAttribute(pool->handle, cudaMemPoolAttrReservedMemCurrent, &result.reservedBytes), "pool reserved bytes");
    }
    return result;
}
void trimUnused() {
    auto pool = currentPool(false);
    check(cudaDeviceSynchronize(), "pool trim synchronize");
    if(pool && pool->handle) check(cudaMemPoolTrimTo(pool->handle, 0), "pool trim");
}
}
