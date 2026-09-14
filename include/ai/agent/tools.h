#pragma once
#include "ai/agent/components.h"

namespace ai::agent {
class CalculatorTool final : public ITool {
    public:
        ToolResult execute(const nlohmann::json &) override;
};
class FileReadTool final : public ITool {
    public:
        explicit FileReadTool(std::filesystem::path root);
        ToolResult execute(const nlohmann::json &) override;

    private:
        std::filesystem::path root_;
};
} // namespace ai::agent
