#pragma once
#include "ai/agent/components.h"

struct sqlite3;

namespace ai::agent {

class SqliteMemory final : public IMemoryManager {
    public:
        explicit SqliteMemory(const std::filesystem::path &path);
        ~SqliteMemory() override;
        SqliteMemory(const SqliteMemory &) = delete;
        SqliteMemory &operator=(const SqliteMemory &) = delete;
        void store(const MemoryCandidate &) override;
        void store_batch(const std::vector<MemoryCandidate> &) override;
        std::vector<MemoryRecord>
            retrieve(std::string_view, std::optional<MemoryType> = std::nullopt,
                     std::size_t = 8) override;

    private:
        void migrate();
        void store_unchecked(const MemoryCandidate &);
        sqlite3 *_db{};
};

class MemoryRetrieveTool final : public ITool {
    public:
        explicit MemoryRetrieveTool(std::shared_ptr<IMemoryManager> memory)
            : _memory(std::move(memory)) {}
        ToolResult execute(const nlohmann::json &) override;

    private:
        std::shared_ptr<IMemoryManager> _memory;
};

} // namespace ai::agent
