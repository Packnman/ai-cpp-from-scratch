#pragma once
#include "ai/agent/components.h"
#include "ai/agent/types.h"
#include <functional>

namespace ai::agent {
struct ContextInput {
    std::string current_input, goal, current_task; // 現在の入力、目標、処理中のタスク
    std::vector<Constraint> constraints; // プロンプトへ含める制約
    std::vector<ConversationTurn> recent; // 直近の会話履歴
    std::vector<MemoryRecord> memories; // 関連する長期記憶
    std::string summary; // 過去の会話要約
    std::vector<ToolResult> previous_results; // 先行タスクの実行結果
};
class ContextBuilder {
    public:
        using Counter = std::function<std::size_t(std::string_view)>;
        explicit ContextBuilder(std::size_t budget = 1024,
                                Counter counter = {});
        std::string build(const ContextInput &) const;
        static std::size_t utf8_codepoints(std::string_view);

    private:
        std::size_t _budget; // コンテキストに許容する最大トークン数
        Counter _counter; // 文字列のトークン数を数える関数
};
ContextInput structured_prompt_input(
    ModelMode mode,
    std::string_view payload,
    std::string_view schema,
    bool repair = false
);
std::string build_model_prompt(
    const ContextInput &,
    std::size_t budget = 1022,
    ContextBuilder::Counter counter = {}
);
} // namespace ai::agent
