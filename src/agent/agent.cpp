#include "ai/agent/agent.h"

#include <stdexcept>

namespace ai::agent {
Agent::Agent(std::shared_ptr<IInputParser> a, std::shared_ptr<IRouter> b,
             std::shared_ptr<IPlanner> c, std::shared_ptr<IExecutor> d,
             std::shared_ptr<IEvaluator> e, std::shared_ptr<IAggregator> f,
             std::shared_ptr<IMemoryManager> g, std::shared_ptr<IReasoner> h)
    : _parser(std::move(a)), _router(std::move(b)), _planner(std::move(c)),
      _executor(std::move(d)), _evaluator(std::move(e)),
      _aggregator(std::move(f)), _memory(std::move(g)),
      _reasoner(std::move(h)) {
    if (!_parser || !_router || !_planner || !_executor || !_evaluator ||
        !_aggregator || !_memory || !_reasoner)
        throw std::invalid_argument("all Agent components are required");
}

void Agent::finish_turn(const ParsedInput &p, std::string_view response) {
    auto candidates = _reasoner->memory_candidates(p, response);
    std::vector<MemoryCandidate> accepted;
    for (auto &c : candidates)
        if (c.importance >= 0.6 && c.confidence >= 0.5 && !c.content.empty() &&
            c.content.size() <= 16 * 1024)
            accepted.push_back(std::move(c));
    if (!accepted.empty())
        _memory->store_batch(accepted);
    _recent.push_back({p.raw, std::string(response)});
    if (_recent.size() > 4) {
        std::vector<ConversationTurn> old(_recent.begin(), _recent.end() - 4);
        _summary = _reasoner->summarize(old, _summary);
        _recent.erase(_recent.begin(), _recent.end() - 4);
    }
}
AgentResponse Agent::process(std::string_view raw) {
    AgentResponse response;
    try {
        auto parsed = _parser->parse(raw);
        auto type = _router->route(parsed);
        auto memories = _memory->retrieve(parsed.raw, std::nullopt, 8);
        AgentState state;
        state.request_type = type;
        state.input = parsed;
        state.constraints = parsed.constraints;
        state.retrieved_memories = memories;
        state.recent_conversation = _recent;
        state.conversation_summary = _summary;
        if (type == RequestType::SimpleConversation) {
            response.text =
                _reasoner->chat(parsed, memories, _recent, _summary);
            finish_turn(parsed, response.text);
            response.state = std::move(state);
            return response;
        }
        Plan plan = _planner->create(parsed, memories);
        std::string error;
        if (!validate_plan(plan, error))
            throw std::runtime_error("invalid plan: " + error);
        int replans = 0;
        std::vector<ToolResult> results;
        for (;;) {
            bool replaced = false;
            for (auto index : topological_order(plan)) {
                const Task &task = plan.tasks[index];
                ToolResult result;
                EvaluationResult evaluation;
                int retries = 0;
                for (;;) {
                    result = _executor->execute(task, results);
                    evaluation = _evaluator->evaluate(task, result);
                    if (evaluation.status == EvaluationStatus::Retry &&
                        retries < 2) {
                        ++retries;
                        continue;
                    }
                    break;
                }
                results.push_back(result);
                if (evaluation.status == EvaluationStatus::Retry)
                    throw std::runtime_error("tool retry limit reached: " +
                                             evaluation.reason);
                if (evaluation.status == EvaluationStatus::Failed)
                    throw std::runtime_error("task failed: " +
                                             evaluation.reason);
                if (evaluation.status == EvaluationStatus::Replan) {
                    if (replans >= 3)
                        throw std::runtime_error("replan limit reached");
                    plan = _planner->replan(parsed, plan, result, evaluation);
                    if (!validate_plan(plan, error))
                        throw std::runtime_error("invalid replanned plan: " +
                                                 error);
                    ++replans;
                    replaced = true;
                    break;
                }
            }
            if (!replaced)
                break;
        }
        auto aggregate = _aggregator->aggregate(results);
        if (!aggregate.at("contradictions").empty())
            throw std::runtime_error(
                "aggregated results contain contradictions");
        response.text = _reasoner->final_response(parsed, aggregate);
        state.current_plan = plan;
        state.task_results = results;
        response.state = std::move(state);
        finish_turn(parsed, response.text);
        return response;
    } catch (const std::exception &e) {
        response.success = false;
        response.error = e.what();
        response.text = "Error: " + response.error;
        return response;
    }
}
} // namespace ai::agent
