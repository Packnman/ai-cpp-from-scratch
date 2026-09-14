#include "ai/agent/reasoner.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace ai::agent {
namespace {
bool starts(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.substr(0, p.size()) == p;
}
std::string after(std::string_view s, std::string_view p) {
    auto v = s.substr(p.size());
    while (!v.empty() && v.front() == ' ')
        v.remove_prefix(1);
    return std::string(v);
}
MemoryType parse_type_or_throw(const nlohmann::json &j) {
    auto t = memory_type_from_string(j.get<std::string>());
    if (!t)
        throw std::runtime_error("invalid memory type");
    return *t;
}
Task task_from_json(const nlohmann::json &j) {
    Task t;
    t.id = j.at("id").get<std::string>();
    t.type = j.value("type", "reasoning") == "tool" ? TaskType::Tool
                                                    : TaskType::Reasoning;
    t.operation = j.at("operation").get<std::string>();
    t.arguments = j.value("arguments", nlohmann::json::object());
    t.depends_on = j.value("depends_on", std::vector<std::string>{});
    return t;
}
Plan plan_from_json(const nlohmann::json &j) {
    Plan p;
    p.goal = j.at("goal").get<std::string>();
    for (const auto &x : j.at("tasks"))
        p.tasks.push_back(task_from_json(x));
    std::string e;
    if (!validate_plan(p, e))
        throw std::runtime_error(e);
    return p;
}
} // namespace

ParsedInput RuleReasoner::parse(std::string_view input) {
    if (input.empty())
        throw std::invalid_argument("input must not be empty");
    ParsedInput p;
    p.raw = std::string(input);
    p.goal = p.raw;
    p.intent = "chat";
    if (starts(input, "/calc ")) {
        p.intent = "calculate";
        p.goal = "式を安全に計算する";
        p.arguments = {{"expression", after(input, "/calc")}};
    } else if (starts(input, "/read ")) {
        p.intent = "read";
        p.goal = "root内のファイルを読む";
        p.arguments = {{"path", after(input, "/read")}};
    } else if (starts(input, "/recall ")) {
        p.intent = "recall";
        p.goal = "長期メモリを検索する";
        p.arguments = {{"query", after(input, "/recall")}, {"limit", 8}};
    } else if (starts(input, "/remember ")) {
        std::istringstream ss(after(input, "/remember"));
        std::string type;
        ss >> type;
        std::string content;
        std::getline(ss, content);
        if (!content.empty() && content.front() == ' ')
            content.erase(0, 1);
        if (!memory_type_from_string(type) || content.empty())
            throw std::invalid_argument(
                "usage: /remember TYPE TEXT (TYPE: semantic|episodic|project)");
        p.intent = "remember";
        p.goal = "指定内容を記憶する";
        p.arguments = {{"type", type}, {"content", content}};
    } else if (input.find("ロボット") != std::string_view::npos ||
               input.find("モータ") != std::string_view::npos)
        p.intent = "robot";
    else if (input.find("分解") != std::string_view::npos ||
             input.find("計画") != std::string_view::npos)
        p.intent = "complex";
    else if (input.find('?') != std::string_view::npos ||
             input.find("？") != std::string_view::npos ||
             input.find("説明") != std::string_view::npos ||
             input.find("教えて") != std::string_view::npos)
        p.intent = "question";
    else if (input.find("思い出") != std::string_view::npos ||
             input.find("覚えて") != std::string_view::npos)
        p.intent = "recall";
    const bool negative = input.find("ない") != std::string_view::npos ||
                          input.find("禁止") != std::string_view::npos ||
                          input.find("するな") != std::string_view::npos;
    const bool safety = input.find("安全") != std::string_view::npos ||
                        input.find("人が") != std::string_view::npos ||
                        input.find("危険") != std::string_view::npos;
    const bool quantity =
        std::any_of(input.begin(), input.end(),
                    [](unsigned char c) { return std::isdigit(c); });
    if (negative || safety || quantity)
        p.constraints.push_back({p.raw, true});
    return p;
}
Plan RuleReasoner::plan(const ParsedInput &p,
                        const std::vector<MemoryRecord> &) {
    Plan plan{p.goal, {}};
    Task t;
    t.id = "task-1";
    if (p.intent == "calculate") {
        t.type = TaskType::Tool;
        t.operation = "calculator.calculate";
        t.arguments = p.arguments;
    } else if (p.intent == "read") {
        t.type = TaskType::Tool;
        t.operation = "file.read";
        t.arguments = p.arguments;
    } else if (p.intent == "recall") {
        t.type = TaskType::Tool;
        t.operation = "memory.retrieve";
        t.arguments = p.arguments.contains("query")
                          ? p.arguments
                          : nlohmann::json{{"query", p.raw}, {"limit", 8}};
    } else if (p.intent == "robot") {
        t.type = TaskType::Tool;
        t.operation = "robot.control";
        t.arguments = {{"request", p.raw}};
    } else {
        t.type = TaskType::Reasoning;
        t.operation = p.intent == "remember" ? "memory acknowledgement"
                                             : "rule-based reasoning";
        t.arguments = p.arguments;
    }
    plan.tasks.push_back(std::move(t));
    return plan;
}
Plan RuleReasoner::replan(const ParsedInput &, const Plan &plan,
                          const ToolResult &, const EvaluationResult &) {
    return plan;
}
EvaluationResult RuleReasoner::evaluate(const Task &, const ToolResult &r) {
    switch (r.status) {
    case ToolStatus::Success:
        return {EvaluationStatus::Success, "ok"};
    case ToolStatus::RetryableError:
        return {EvaluationStatus::Retry, r.error};
    case ToolStatus::NeedsReplan:
        return {EvaluationStatus::Replan, r.error};
    case ToolStatus::PermanentError:
        return {EvaluationStatus::Failed, r.error};
    }
    return {EvaluationStatus::Failed, "invalid tool status"};
}
std::string RuleReasoner::summarize(const std::vector<ConversationTurn> &turns,
                                    std::string_view previous) {
    std::string s(previous);
    for (const auto &t : turns) {
        if (!s.empty())
            s += '\n';
        s += "User: " + t.user + " / Agent: " + t.assistant;
    }
    if (s.size() > 8192)
        s.erase(0, s.size() - 8192);
    return s;
}
std::vector<MemoryCandidate>
RuleReasoner::memory_candidates(const ParsedInput &p, std::string_view) {
    if (p.intent != "remember")
        return {};
    return {{parse_type_or_throw(p.arguments.at("type")),
             p.arguments.at("content").get<std::string>(), 1.0, 1.0}};
}
std::string RuleReasoner::chat(const ParsedInput &p,
                               const std::vector<MemoryRecord> &,
                               const std::vector<ConversationTurn> &,
                               std::string_view) {
    return "受け取りました: " + p.raw +
           "（rule backendは動作確認用で、学習済み日本語モデルではありません）";
}
std::string RuleReasoner::final_response(const ParsedInput &p,
                                         const nlohmann::json &aggregate) {
    if (p.intent == "remember")
        return "記憶しました。";
    const auto &rs = aggregate.at("results");
    if (rs.empty())
        return "結果はありません。";
    const auto &v = rs.front().at("value");
    if (p.intent == "calculate")
        return v.at("value").dump();
    if (p.intent == "read")
        return v.at("content").get<std::string>();
    if (p.intent == "recall") {
        std::string out;
        for (const auto &m : v.at("memories")) {
            if (!out.empty())
                out += '\n';
            out += "[" + m.at("type").get<std::string>() + "] " +
                   m.at("content").get<std::string>();
        }
        return out.empty() ? "一致する記憶はありません。" : out;
    }
    return "rule backendによる処理結果: " + v.dump();
}

nlohmann::json ModelReasoner::structured(ModelMode mode,
                                         std::string_view prompt,
                                         std::string_view schema_text) {
    const auto schema = nlohmann::json::parse(schema_text);
    ContextInput context;
    context.current_input = std::string(prompt);
    context.goal = "Return an answer matching the requested mode";
    context.current_task = "Schema: " + std::string(schema_text);
    std::string answer = _model->complete(mode, build_prompt(context));
    for (int attempt = 0; attempt < 2; ++attempt) {
        try {
            auto j = nlohmann::json::parse(answer);
            if (!j.is_object())
                throw std::runtime_error("root must be object");
            for (const auto &key : schema.at("required"))
                if (!j.contains(key.get<std::string>()))
                    throw std::runtime_error("missing required field");
            const auto require_string = [&](const char *key) {
                if (j.contains(key) && !j[key].is_string())
                    throw std::runtime_error(std::string(key) +
                                             " must be a string");
            };
            const auto require_array = [&](const char *key) {
                if (j.contains(key) && !j[key].is_array())
                    throw std::runtime_error(std::string(key) +
                                             " must be an array");
            };
            for (const char *key :
                 {"raw", "intent", "goal", "status", "reason"})
                require_string(key);
            for (const char *key : {"constraints", "tasks", "memories"})
                require_array(key);
            if (mode == ModelMode::Parse)
                for (const auto &item : j.at("constraints"))
                    if (!item.is_object() || !item.contains("text") ||
                        !item.at("text").is_string() ||
                        (item.contains("critical") &&
                         !item.at("critical").is_boolean()))
                        throw std::runtime_error("invalid nested constraint");
            if (mode == ModelMode::Plan)
                for (const auto &item : j.at("tasks"))
                    if (!item.is_object() || !item.contains("id") ||
                        !item.at("id").is_string() ||
                        !item.contains("operation") ||
                        !item.at("operation").is_string() ||
                        (item.contains("arguments") &&
                         !item.at("arguments").is_object()) ||
                        (item.contains("depends_on") &&
                         !item.at("depends_on").is_array()))
                        throw std::runtime_error("invalid nested task");
            if (mode == ModelMode::MemoryWrite)
                for (const auto &item : j.at("memories"))
                    if (!item.is_object() || !item.contains("type") ||
                        !item.at("type").is_string() ||
                        !item.contains("content") ||
                        !item.at("content").is_string() ||
                        !item.contains("importance") ||
                        !item.at("importance").is_number() ||
                        !item.contains("confidence") ||
                        !item.at("confidence").is_number())
                        throw std::runtime_error("invalid nested memory");
            return j;
        } catch (const std::exception &e) {
            if (attempt == 1)
                throw std::runtime_error(
                    std::string("model returned invalid structured JSON after "
                                "one repair: ") +
                    e.what());
            ContextInput repair;
            repair.current_input = answer;
            repair.goal = "Repair invalid JSON once and return JSON only";
            repair.current_task = "Schema: " + std::string(schema_text);
            answer = _model->complete(mode, build_prompt(repair));
        }
    }
    throw std::runtime_error("unreachable");
}
std::string ModelReasoner::build_prompt(const ContextInput &input) const {
    ContextBuilder builder(1022, [this](std::string_view text) {
        return _model->token_count(text);
    });
    return builder.build(input);
}
ParsedInput ModelReasoner::parse(std::string_view s) {
    auto j =
        structured(ModelMode::Parse, s,
                   R"({"required":["raw","intent","goal","constraints"]})");
    ParsedInput p{j.at("raw"), j.at("intent"), j.at("goal")};
    p.arguments = j.value("arguments", nlohmann::json::object());
    for (const auto &c : j.at("constraints"))
        p.constraints.push_back({c.at("text"), c.value("critical", false)});
    return p;
}
Plan ModelReasoner::plan(const ParsedInput &p,
                         const std::vector<MemoryRecord> &m) {
    return plan_from_json(structured(
        ModelMode::Plan,
        nlohmann::json{{"input", p.raw}, {"memories", m.size()}}.dump(),
        R"({"required":["goal","tasks"]})"));
}
Plan ModelReasoner::replan(const ParsedInput &p, const Plan &,
                           const ToolResult &r, const EvaluationResult &e) {
    return plan_from_json(structured(ModelMode::Plan,
                                     nlohmann::json{{"input", p.raw},
                                                    {"failed_result", r.error},
                                                    {"reason", e.reason}}
                                         .dump(),
                                     R"({"required":["goal","tasks"]})"));
}
EvaluationResult ModelReasoner::evaluate(const Task &t, const ToolResult &r) {
    auto j = structured(ModelMode::Evaluate,
                        nlohmann::json{{"task", t.id},
                                       {"status", static_cast<int>(r.status)},
                                       {"value", r.value},
                                       {"error", r.error}}
                            .dump(),
                        R"({"required":["status","reason"]})");
    std::string s = j.at("status");
    if (s == "success")
        return {EvaluationStatus::Success, j.at("reason")};
    if (s == "retry")
        return {EvaluationStatus::Retry, j.at("reason")};
    if (s == "replan")
        return {EvaluationStatus::Replan, j.at("reason")};
    return {EvaluationStatus::Failed, j.at("reason")};
}
std::string ModelReasoner::summarize(const std::vector<ConversationTurn> &t,
                                     std::string_view old) {
    ContextInput context;
    context.current_input = "Summarize the conversation";
    context.goal = "Preserve decisions, open questions, and current intent";
    context.current_task = "Conversation summary";
    context.recent = t;
    context.summary = std::string(old);
    return _model->complete(ModelMode::Summarize, build_prompt(context));
}
std::vector<MemoryCandidate>
ModelReasoner::memory_candidates(const ParsedInput &p,
                                 std::string_view response) {
    auto j = structured(
        ModelMode::MemoryWrite,
        nlohmann::json{{"input", p.raw}, {"response", response}}.dump(),
        R"({"required":["memories"]})");
    std::vector<MemoryCandidate> out;
    for (const auto &m : j.at("memories"))
        out.push_back({parse_type_or_throw(m.at("type")), m.at("content"),
                       m.at("importance"), m.at("confidence")});
    return out;
}
std::string ModelReasoner::chat(const ParsedInput &p,
                                const std::vector<MemoryRecord> &memories,
                                const std::vector<ConversationTurn> &recent,
                                std::string_view summary) {
    ContextInput context;
    context.current_input = p.raw;
    context.goal = p.goal;
    context.current_task = "Natural Japanese conversation";
    context.constraints = p.constraints;
    context.memories = memories;
    context.recent = recent;
    context.summary = std::string(summary);
    return _model->complete(ModelMode::Chat, build_prompt(context));
}
std::string ModelReasoner::final_response(const ParsedInput &p,
                                          const nlohmann::json &a) {
    ContextInput context;
    context.current_input = p.raw;
    context.goal = p.goal;
    context.current_task = "Produce the final response from: " + a.dump();
    context.constraints = p.constraints;
    return _model->complete(ModelMode::Final, build_prompt(context));
}
} // namespace ai::agent
