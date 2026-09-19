#pragma once
#include "ai/agent/components.h"
#include "ai/ner/extractor.h"

namespace ai::agent {
class Agent {
    public:
        Agent(std::shared_ptr<IInputParser>, std::shared_ptr<IRouter>,
              std::shared_ptr<IPlanner>, std::shared_ptr<IExecutor>,
              std::shared_ptr<IEvaluator>, std::shared_ptr<IAggregator>,
              std::shared_ptr<IMemoryManager>, std::shared_ptr<IReasoner>,
              std::shared_ptr<ai::ner::IEntityExtractor> = nullptr);
        AgentResponse process(std::string_view input);

    private:
        void finish_turn(const ParsedInput &, std::string_view response);
        void compress_for_chat(const ParsedInput &,
                               const std::vector<MemoryRecord> &);
        std::shared_ptr<IInputParser> _parser; // ユーザー入力の解析器
        std::shared_ptr<IRouter> _router;      // 要求種別の振り分け器
        std::shared_ptr<IPlanner> _planner;    // タスク計画の生成器
        std::shared_ptr<IExecutor> _executor;  // タスクの実行器
        std::shared_ptr<IEvaluator> _evaluator;   // 実行結果の評価器
        std::shared_ptr<IAggregator> _aggregator; // 複数結果の集約器
        std::shared_ptr<IMemoryManager> _memory;  // 長期記憶の管理器
        std::shared_ptr<IReasoner> _reasoner;     // 推論処理の実装
        std::shared_ptr<ai::ner::IEntityExtractor> _entity_extractor;
        std::uint64_t _turn_sequence{};
        std::vector<ConversationTurn> _recent;     // 要約前の直近会話
        std::string _summary;                      // 過去の会話要約
        nlohmann::json _discussion_state{nullptr}; // canonical debate state
};
} // namespace ai::agent
