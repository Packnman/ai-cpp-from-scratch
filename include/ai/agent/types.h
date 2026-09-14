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
        std::string text;
        bool critical{false};
};

struct ParsedInput {
        std::string raw;
        std::string intent;
        std::string goal;
        std::vector<Constraint> constraints;
        nlohmann::json arguments{nlohmann::json::object()};
};

struct Task {
        std::string id;
        TaskType type{TaskType::Reasoning};
        std::string operation;
        nlohmann::json arguments{nlohmann::json::object()};
        std::vector<std::string> depends_on;
};

struct Plan {
        std::string goal;
        std::vector<Task> tasks;
};

struct ToolResult {
        std::string task_id;
        ToolStatus status{ToolStatus::Success};
        nlohmann::json value{nlohmann::json::object()};
        std::string error;
};

struct EvaluationResult {
        EvaluationStatus status{EvaluationStatus::Success};
        std::string reason;
};

struct MemoryCandidate {
        MemoryType type{MemoryType::Semantic};
        std::string content;
        double importance{0.0};
        double confidence{0.0};
};

struct MemoryRecord : MemoryCandidate {
        std::int64_t id{};
        std::int64_t created_at{};
        std::int64_t updated_at{};
        std::int64_t last_accessed_at{};
};

struct ConversationTurn {
        std::string user;
        std::string assistant;
};

struct AgentState {
        RequestType request_type{RequestType::SimpleConversation};
        ParsedInput input;
        std::vector<Constraint> constraints;
        std::vector<MemoryRecord> retrieved_memories;
        std::optional<Plan> current_plan;
        std::vector<ToolResult> task_results;
        std::vector<ConversationTurn> recent_conversation;
        std::string conversation_summary;
};

struct AgentResponse {
        bool success{true};
        std::string text;
        AgentState state;
        std::string error;
};

std::string to_string(RequestType value);
std::string to_string(MemoryType value);
std::optional<MemoryType> memory_type_from_string(std::string_view value);

} // namespace ai::agent
