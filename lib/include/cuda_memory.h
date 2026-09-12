#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

// Internal allocator. All accesses must be ordered on the default stream.
namespace cu_memory {
struct DevicePool;
struct Statistics {
    bool pooled = false;
    std::uint64_t usedBytes = 0;
    std::uint64_t reservedBytes = 0;
};
class Buffer {
public:
    explicit Buffer(std::size_t bytes = 0);
    ~Buffer() noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void* data() const noexcept { return pointer_; }
    void release();
private:
    std::shared_ptr<DevicePool> pool_;
    void* pointer_ = nullptr;
};
// Synchronize the current device before sampling or explicitly trimming.
Statistics statistics();
void trimUnused();
}
