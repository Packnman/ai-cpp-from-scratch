#pragma once
#include "brain/memory/MemoryTypes.hpp"
#include <memory>
namespace ai::brain {
class IMemoryBackend {
    public:
        virtual ~IMemoryBackend() = default;
        virtual bool available() const noexcept = 0;
        virtual MemoryId store(MemoryItem) = 0;
        virtual bool update(const MemoryItem &) = 0;
        virtual bool remove(MemoryId) = 0;
        virtual std::optional<MemoryItem> get(MemoryId) = 0;
        virtual std::vector<MemoryItem> search(const MemoryQuery &) = 0;
};
class SQLiteMemoryBackend final : public IMemoryBackend {
    public:
        explicit SQLiteMemoryBackend(const std::string &path);
        ~SQLiteMemoryBackend() override;
        SQLiteMemoryBackend(const SQLiteMemoryBackend &) = delete;
        SQLiteMemoryBackend &operator=(const SQLiteMemoryBackend &) = delete;
        bool available() const noexcept override;
        MemoryId store(MemoryItem) override;
        bool update(const MemoryItem &) override;
        bool remove(MemoryId) override;
        std::optional<MemoryItem> get(MemoryId) override;
        std::vector<MemoryItem> search(const MemoryQuery &) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
};
} // namespace ai::brain
