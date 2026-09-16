#pragma once
#include "ai/agent/components.h"
#include "ai/agent/context_builder.h"

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
        std::string build_prompt(const ContextInput &) const;
        nlohmann::json structured(ModelMode, std::string_view,
                                  std::string_view schema);
        std::shared_ptr<ILanguageModel> _model; // 構造化推論に用いる言語モデル
};

class HybridReasoner final : public IReasoner {
    public:
        explicit HybridReasoner(std::shared_ptr<ILanguageModel> model)
            : _model(std::move(model)) {}
        ParsedInput parse(std::string_view v) override {
            return _rule.parse(v);
        }
        Plan plan(const ParsedInput &p,
                  const std::vector<MemoryRecord> &m) override {
            return _rule.plan(p, m);
        }
        Plan replan(const ParsedInput &p, const Plan &v, const ToolResult &r,
                    const EvaluationResult &e) override {
            return _rule.replan(p, v, r, e);
        }
        EvaluationResult evaluate(const Task &t, const ToolResult &r) override {
            return _rule.evaluate(t, r);
        }
        std::string summarize(const std::vector<ConversationTurn> &t,
                              std::string_view old) override {
            return _rule.summarize(t, old);
        }
        std::vector<MemoryCandidate>
        memory_candidates(const ParsedInput &p, std::string_view r) override {
            return _rule.memory_candidates(p, r);
        }
        std::string chat(const ParsedInput &, const std::vector<MemoryRecord> &,
                         const std::vector<ConversationTurn> &,
                         std::string_view) override;
        std::string final_response(const ParsedInput &p,
                                   const nlohmann::json &a) override {
            return _rule.final_response(p, a);
        }

    private:
        RuleReasoner _rule; // 決定的な処理を担うルール推論器
        std::shared_ptr<ILanguageModel> _model; // 会話生成に用いる言語モデル
};
} // namespace ai::agent
