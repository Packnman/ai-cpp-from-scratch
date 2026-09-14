#pragma once
#include "ai/agent/components.h"

namespace ai::agent {

class RuleReasoner final : public IReasoner {
    public:
        ParsedInput parse(std::string_view) override;
        Plan plan(const ParsedInput &,
                  const std::vector<MemoryRecord> &) override;
        Plan replan(const ParsedInput &, const Plan &, const ToolResult &,
                    const EvaluationResult &) override;
        EvaluationResult evaluate(const Task &, const ToolResult &) override;
        std::string summarize(const std::vector<ConversationTurn> &,
                              std::string_view) override;
        std::vector<MemoryCandidate>
        memory_candidates(const ParsedInput &, std::string_view) override;
        std::string chat(const ParsedInput &, const std::vector<MemoryRecord> &,
                         const std::vector<ConversationTurn> &,
                         std::string_view) override;
        std::string final_response(const ParsedInput &,
                                   const nlohmann::json &) override;
};

class ModelReasoner final : public IReasoner {
    public:
        explicit ModelReasoner(std::shared_ptr<ILanguageModel> model)
            : _model(std::move(model)) {}
        ParsedInput parse(std::string_view) override;
        Plan plan(const ParsedInput &,
                  const std::vector<MemoryRecord> &) override;
        Plan replan(const ParsedInput &, const Plan &, const ToolResult &,
                    const EvaluationResult &) override;
        EvaluationResult evaluate(const Task &, const ToolResult &) override;
        std::string summarize(const std::vector<ConversationTurn> &,
                              std::string_view) override;
        std::vector<MemoryCandidate>
        memory_candidates(const ParsedInput &, std::string_view) override;
        std::string chat(const ParsedInput &, const std::vector<MemoryRecord> &,
                         const std::vector<ConversationTurn> &,
                         std::string_view) override;
        std::string final_response(const ParsedInput &,
                                   const nlohmann::json &) override;

    private:
        nlohmann::json structured(ModelMode, std::string_view,
                                  std::string_view schema);
        std::shared_ptr<ILanguageModel> _model;
};

} // namespace ai::agent
