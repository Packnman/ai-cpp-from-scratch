#pragma once

#include "ai/agent/types.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string_view>

namespace ai::agent {

class IInputParser {
    public:
        virtual ~IInputParser() = default;
        virtual ParsedInput parse(std::string_view input) = 0;
};

class IRouter {
    public:
        virtual ~IRouter() = default;
        virtual RequestType route(const ParsedInput &input) = 0;
};

class IPlanner {
    public:
        virtual ~IPlanner() = default;
        virtual Plan create(const ParsedInput &,
                            const std::vector<MemoryRecord> &) = 0;
        virtual Plan replan(const ParsedInput &, const Plan &,
                            const ToolResult &, const EvaluationResult &) = 0;
};

class ITool {
    public:
        virtual ~ITool() = default;
        virtual ToolResult execute(const nlohmann::json &arguments) = 0;
};

class ToolRegistry {
    public:
        void add(std::string name, std::shared_ptr<ITool> tool);
        ToolResult execute(std::string_view name,
                           const nlohmann::json &arguments) const;

    private:
        std::map<std::string, std::shared_ptr<ITool>, std::less<>> _tools;
};

class IExecutor {
    public:
        virtual ~IExecutor() = default;
        virtual ToolResult execute(const Task &,
                                   const std::vector<ToolResult> &) = 0;
};

class IEvaluator {
    public:
        virtual ~IEvaluator() = default;
        virtual EvaluationResult evaluate(const Task &, const ToolResult &) = 0;
};

class IAggregator {
    public:
        virtual ~IAggregator() = default;
        virtual nlohmann::json aggregate(const std::vector<ToolResult> &) = 0;
};

class IMemoryManager {
    public:
        virtual ~IMemoryManager() = default;
        virtual void store(const MemoryCandidate &) = 0;
        virtual void store_batch(const std::vector<MemoryCandidate> &) = 0;
        virtual std::vector<MemoryRecord>
        retrieve(std::string_view, std::optional<MemoryType> = std::nullopt,
                 std::size_t limit = 8) = 0;
};

enum class ModelMode {
    Chat,
    Parse,
    Plan,
    Evaluate,
    Summarize,
    MemoryWrite,
    MemoryQuery,
    Final,
    Tool
};

class ILanguageModel {
    public:
        virtual ~ILanguageModel() = default;
        virtual std::string complete(ModelMode mode,
                                     std::string_view prompt) = 0;
};

class IReasoner {
    public:
        virtual ~IReasoner() = default;
        virtual ParsedInput parse(std::string_view) = 0;
        virtual Plan plan(const ParsedInput &,
                          const std::vector<MemoryRecord> &) = 0;
        virtual Plan replan(const ParsedInput &, const Plan &,
                            const ToolResult &, const EvaluationResult &) = 0;
        virtual EvaluationResult evaluate(const Task &, const ToolResult &) = 0;
        virtual std::string summarize(const std::vector<ConversationTurn> &,
                                      std::string_view previous) = 0;
        virtual std::vector<MemoryCandidate>
        memory_candidates(const ParsedInput &, std::string_view response) = 0;
        virtual std::string chat(const ParsedInput &,
                                 const std::vector<MemoryRecord> &,
                                 const std::vector<ConversationTurn> &,
                                 std::string_view summary) = 0;
        virtual std::string final_response(const ParsedInput &,
                                           const nlohmann::json &) = 0;
};

class DefaultInputParser final : public IInputParser {
    public:
        explicit DefaultInputParser(std::shared_ptr<IReasoner> reasoner)
            : _reasoner(std::move(reasoner)) {}
        ParsedInput parse(std::string_view input) override;

    private:
        std::shared_ptr<IReasoner> _reasoner;
};

class DefaultRouter final : public IRouter {
    public:
        RequestType route(const ParsedInput &) override;
};
class DefaultPlanner final : public IPlanner {
    public:
        explicit DefaultPlanner(std::shared_ptr<IReasoner> r)
            : _reasoner(std::move(r)) {}
        Plan create(const ParsedInput &,
                    const std::vector<MemoryRecord> &) override;
        Plan replan(const ParsedInput &, const Plan &, const ToolResult &,
                    const EvaluationResult &) override;

    private:
        std::shared_ptr<IReasoner> _reasoner;
};
class DefaultExecutor final : public IExecutor {
    public:
        explicit DefaultExecutor(std::shared_ptr<ToolRegistry> tools)
            : _tools(std::move(tools)) {}
        ToolResult execute(const Task &,
                           const std::vector<ToolResult> &) override;

    private:
        std::shared_ptr<ToolRegistry> _tools;
};
class DefaultEvaluator final : public IEvaluator {
    public:
        explicit DefaultEvaluator(std::shared_ptr<IReasoner> r)
            : _reasoner(std::move(r)) {}
        EvaluationResult evaluate(const Task &, const ToolResult &) override;

    private:
        std::shared_ptr<IReasoner> _reasoner;
};
class DefaultAggregator final : public IAggregator {
    public:
        nlohmann::json aggregate(const std::vector<ToolResult> &) override;
};

bool validate_plan(const Plan &plan, std::string &error);
std::vector<std::size_t> topological_order(const Plan &plan);

} // namespace ai::agent
