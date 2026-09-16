#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace ai::agent {

enum class RequestType {
    SimpleConversation,
    QuestionAnswer,
    MemoryRecall,
    ComplexReasoning,
    ToolTask,
    RobotTask
};
enum class MemoryType { Semantic, Episodic, Project };
enum class TaskType { Tool, Reasoning };
enum class ToolStatus { Success, RetryableError, PermanentError, NeedsReplan };
enum class EvaluationStatus { Success, Retry, Replan, Failed };

struct Constraint {
        std::string text; // 制約の内容
        bool critical{false}; // 違反を許容しない必須制約か
};

struct ParsedInput {
        std::string raw; // 入力された原文
        std::string intent; // 推定した要求の意図
        std::string goal; // 達成すべき目標
        std::vector<Constraint> constraints; // 入力から抽出した制約
        nlohmann::json arguments{nlohmann::json::object()}; // 抽出した構造化引数
};

struct Task {
        std::string id; // タスクを識別するID
        TaskType type{TaskType::Reasoning}; // タスクの実行種別
        std::string operation; // 実行する操作名
        nlohmann::json arguments{nlohmann::json::object()}; // 操作へ渡す引数
        std::vector<std::string> depends_on; // 先に完了すべきタスクID
};

struct Plan {
        std::string goal; // 計画全体の目標
        std::vector<Task> tasks; // 実行するタスク列
};

struct ToolResult {
        std::string task_id; // 対応するタスクID
        ToolStatus status{ToolStatus::Success}; // ツール実行の状態
        nlohmann::json value{nlohmann::json::object()}; // ツールが返した値
        std::string error; // 失敗時のエラー内容
};

struct EvaluationResult {
        EvaluationStatus status{EvaluationStatus::Success}; // 評価結果の状態
        std::string reason; // 判定理由
};

struct MemoryCandidate {
        MemoryType type{MemoryType::Semantic}; // 記憶の分類
        std::string content; // 保存する記憶の本文
        double importance{0.0}; // 記憶の重要度
        double confidence{0.0}; // 内容の確信度
};

struct MemoryRecord : MemoryCandidate {
        std::int64_t id{}; // 永続化された記憶のID
        std::int64_t created_at{}; // 作成時刻のUnix時刻
        std::int64_t updated_at{}; // 最終更新時刻のUnix時刻
        std::int64_t last_accessed_at{}; // 最終参照時刻のUnix時刻
};

struct ConversationTurn {
        std::string user; // ユーザーの発話
        std::string assistant; // アシスタントの応答
};

struct AgentState {
        RequestType request_type{RequestType::SimpleConversation}; // 現在の要求種別
        ParsedInput input; // 解析済みの入力
        std::vector<Constraint> constraints; // 処理中に適用する制約
        std::vector<MemoryRecord> retrieved_memories; // 検索で取得した記憶
        std::optional<Plan> current_plan; // 実行中の計画（未作成なら空）
        std::vector<ToolResult> task_results; // 各タスクの実行結果
        std::vector<ConversationTurn> recent_conversation; // 直近の会話履歴
        std::string conversation_summary; // 過去の会話要約
};

struct AgentResponse {
        bool success{true}; // 処理が成功したか
        std::string text; // ユーザーへ返す応答本文
        AgentState state; // 処理後のエージェント状態
        std::string error; // 失敗時のエラー内容
};

std::string to_string(RequestType value);
std::string to_string(MemoryType value);
std::optional<MemoryType> memory_type_from_string(std::string_view value);

} // namespace ai::agent
