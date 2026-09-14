#pragma once
#include "ai/agent/types.h"
#include <functional>

namespace ai::agent {
struct ContextInput {
        std::string current_input, goal, current_task;
        std::vector<Constraint> constraints;
        std::vector<ConversationTurn> recent;
        std::vector<MemoryRecord> memories;
        std::string summary;
        std::vector<ToolResult> previous_results;
};
class ContextBuilder {
    public:
        using Counter = std::function<std::size_t(std::string_view)>;
        explicit ContextBuilder(std::size_t budget = 1024,
                                Counter counter = {});
        std::string build(const ContextInput &) const;
        static std::size_t utf8_codepoints(std::string_view);

    private:
        std::size_t _budget;
        Counter _counter;
};
} // namespace ai::agent
