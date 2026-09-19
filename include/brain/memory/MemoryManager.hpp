#pragma once
#include "brain/memory/MemoryBackend.hpp"
#include <deque>
#include <mutex>
namespace ai::brain {
class IMemoryManager {
    public:
        virtual ~IMemoryManager() = default;
        virtual MemoryId remember(MemoryItem) = 0;
        virtual std::vector<MemoryItem> recall(const MemoryQuery &) = 0;
        virtual bool persistent() const noexcept = 0;
};
class MemoryManager final : public IMemoryManager {
    public:
        explicit MemoryManager(std::unique_ptr<IMemoryBackend> backend = {},
                               std::size_t stmCapacity = 128);
        MemoryId remember(MemoryItem) override;
        std::vector<MemoryItem> recall(const MemoryQuery &) override;
        bool persistent() const noexcept override;
        std::size_t shortTermSize() const;

    private:
        bool matches(const MemoryItem &, const MemoryQuery &) const;
        std::unique_ptr<IMemoryBackend> _backend;
        std::size_t _capacity;
        mutable std::mutex _mutex;
        std::deque<MemoryItem> _shortTerm;
        MemoryId _nextFallbackId{1};
};
} // namespace ai::brain
