#include "ai/agent/components.h"

#include <algorithm>
#include <queue>
#include <set>
#include <stdexcept>

namespace ai::agent {

std::string to_string(RequestType v) {
    static constexpr const char *names[] = {
        "simple_conversation", "question_answer", "memory_recall",
        "complex_reasoning",   "tool_task",       "robot_task"};
    return names[static_cast<int>(v)];
}
std::string to_string(MemoryType v) {
    static constexpr const char *names[] = {"semantic", "episodic", "project"};
    return names[static_cast<int>(v)];
}
std::optional<MemoryType> memory_type_from_string(std::string_view v) {
    if (v == "semantic")
        return MemoryType::Semantic;
    if (v == "episodic")
        return MemoryType::Episodic;
    if (v == "project")
        return MemoryType::Project;
    return std::nullopt;
}

void ToolRegistry::add(std::string name, std::shared_ptr<ITool> tool) {
    if (name.empty() || !tool)
        throw std::invalid_argument(
            "tool name and implementation are required");
    tools_[std::move(name)] = std::move(tool);
}
ToolResult ToolRegistry::execute(std::string_view name,
                                 const nlohmann::json &args) const {
    const auto it = tools_.find(name);
    if (it == tools_.end())
        return {.status = ToolStatus::PermanentError,
                .error = "unknown tool: " + std::string(name)};
    try {
        return it->second->execute(args);
    } catch (const std::exception &e) {
        return {.status = ToolStatus::PermanentError, .error = e.what()};
    }
}

ParsedInput DefaultInputParser::parse(std::string_view input) {
    return reasoner_->parse(input);
}

RequestType DefaultRouter::route(const ParsedInput &in) {
    if (in.intent == "calculate" || in.intent == "read" ||
        in.intent == "remember")
        return RequestType::ToolTask;
    if (in.intent == "recall")
        return RequestType::MemoryRecall;
    if (in.intent == "robot")
        return RequestType::RobotTask;
    if (in.intent == "complex")
        return RequestType::ComplexReasoning;
    if (in.intent == "question")
        return RequestType::QuestionAnswer;
    return RequestType::SimpleConversation;
}
Plan DefaultPlanner::create(const ParsedInput &p,
                            const std::vector<MemoryRecord> &m) {
    return reasoner_->plan(p, m);
}
Plan DefaultPlanner::replan(const ParsedInput &p, const Plan &plan,
                            const ToolResult &r, const EvaluationResult &e) {
    return reasoner_->replan(p, plan, r, e);
}
ToolResult DefaultExecutor::execute(const Task &task,
                                    const std::vector<ToolResult> &) {
    ToolResult result;
    if (task.type == TaskType::Tool)
        result = tools_->execute(task.operation, task.arguments);
    else {
        result.status = ToolStatus::Success;
        result.value = {{"result", task.operation}};
    }
    result.task_id = task.id;
    return result;
}
EvaluationResult DefaultEvaluator::evaluate(const Task &t,
                                            const ToolResult &r) {
    return reasoner_->evaluate(t, r);
}

nlohmann::json
DefaultAggregator::aggregate(const std::vector<ToolResult> &results) {
    nlohmann::json values = nlohmann::json::array();
    std::set<std::string> seen;
    std::map<std::string, nlohmann::json> scalar;
    std::vector<std::string> contradictions;
    for (const auto &result : results) {
        const std::string dump = result.value.dump();
        if (!seen.insert(dump).second)
            continue;
        if (result.value.is_object()) {
            for (auto it = result.value.begin(); it != result.value.end();
                 ++it) {
                if (!it.value().is_primitive())
                    continue;
                auto old = scalar.find(it.key());
                if (old != scalar.end() && old->second != it.value())
                    contradictions.push_back(it.key());
                scalar[it.key()] = it.value();
            }
        }
        values.push_back(
            {{"task_id", result.task_id}, {"value", result.value}});
    }
    return {{"results", values}, {"contradictions", contradictions}};
}

bool validate_plan(const Plan &plan, std::string &error) {
    if (plan.tasks.size() > 32) {
        error = "plan exceeds 32 tasks";
        return false;
    }
    std::map<std::string, std::size_t> ids;
    for (std::size_t i = 0; i < plan.tasks.size(); ++i) {
        if (plan.tasks[i].id.empty() ||
            !ids.emplace(plan.tasks[i].id, i).second) {
            error = "empty or duplicate task id";
            return false;
        }
    }
    std::vector<int> degree(plan.tasks.size());
    std::vector<std::vector<std::size_t>> next(plan.tasks.size());
    for (std::size_t i = 0; i < plan.tasks.size(); ++i)
        for (const auto &dep : plan.tasks[i].depends_on) {
            auto it = ids.find(dep);
            if (it == ids.end()) {
                error = "missing dependency: " + dep;
                return false;
            }
            if (it->second == i) {
                error = "plan contains a cycle";
                return false;
            }
            ++degree[i];
            next[it->second].push_back(i);
        }
    std::queue<std::size_t> q;
    for (std::size_t i = 0; i < degree.size(); ++i)
        if (!degree[i])
            q.push(i);
    std::size_t visited = 0;
    while (!q.empty()) {
        auto i = q.front();
        q.pop();
        ++visited;
        for (auto n : next[i])
            if (!--degree[n])
                q.push(n);
    }
    if (visited != plan.tasks.size()) {
        error = "plan contains a cycle";
        return false;
    }
    return true;
}

std::vector<std::size_t> topological_order(const Plan &plan) {
    std::string error;
    if (!validate_plan(plan, error))
        throw std::invalid_argument(error);
    std::map<std::string, std::size_t> ids;
    for (std::size_t i = 0; i < plan.tasks.size(); ++i)
        ids[plan.tasks[i].id] = i;
    std::vector<int> degree(plan.tasks.size());
    std::vector<std::vector<std::size_t>> next(plan.tasks.size());
    for (std::size_t i = 0; i < plan.tasks.size(); ++i)
        for (const auto &d : plan.tasks[i].depends_on) {
            ++degree[i];
            next[ids.at(d)].push_back(i);
        }
    std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<>>
        q;
    for (std::size_t i = 0; i < degree.size(); ++i)
        if (!degree[i])
            q.push(i);
    std::vector<std::size_t> out;
    while (!q.empty()) {
        auto i = q.top();
        q.pop();
        out.push_back(i);
        for (auto n : next[i])
            if (!--degree[n])
                q.push(n);
    }
    return out;
}

} // namespace ai::agent
