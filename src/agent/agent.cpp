#include "ai/agent/agent.h"

#include <stdexcept>

namespace ai::agent {
Agent::Agent(std::shared_ptr<IInputParser> a, std::shared_ptr<IRouter> b,
             std::shared_ptr<IPlanner> c, std::shared_ptr<IExecutor> d,
             std::shared_ptr<IEvaluator> e, std::shared_ptr<IAggregator> f,
             std::shared_ptr<IMemoryManager> g, std::shared_ptr<IReasoner> h)
    : parser_(std::move(a)), router_(std::move(b)), planner_(std::move(c)),
      executor_(std::move(d)), evaluator_(std::move(e)),
      aggregator_(std::move(f)), memory_(std::move(g)),
      reasoner_(std::move(h)) {
    if (!parser_ || !router_ || !planner_ || !executor_ || !evaluator_ ||
        !aggregator_ || !memory_ || !reasoner_)
        throw std::invalid_argument("all Agent components are required");
}

void Agent::finish_turn(const ParsedInput &p, std::string_view response) {
    auto candidates = reasoner_->memory_candidates(p, response);
    std::vector<MemoryCandidate> accepted;
    for (auto &c : candidates)
        if (c.importance >= 0.6 && c.confidence >= 0.5 && !c.content.empty() &&
            c.content.size() <= 16 * 1024)
            accepted.push_back(std::move(c));
    if (!accepted.empty())
        memory_->store_batch(accepted);
    recent_.push_back({p.raw, std::string(response)});
    if (recent_.size() > 4) {
        std::vector<ConversationTurn> old(recent_.begin(), recent_.end() - 4);
        summary_ = reasoner_->summarize(old, summary_);
        recent_.erase(recent_.begin(), recent_.end() - 4);
    }
}
AgentResponse Agent::process(std::string_view raw) {
    AgentResponse response;
    try {
        auto parsed = parser_->parse(raw);
        auto type = router_->route(parsed);
        auto memories = memory_->retrieve(parsed.raw, std::nullopt, 8);
        AgentState state;
        state.request_type = type;
        state.input = parsed;
        state.constraints = parsed.constraints;
        state.retrieved_memories = memories;
        state.recent_conversation = recent_;
        state.conversation_summary = summary_;
        if (type == RequestType::SimpleConversation) {
            response.text =
                reasoner_->chat(parsed, memories, recent_, summary_);
            finish_turn(parsed, response.text);
            response.state = std::move(state);
            return response;
        }
        Plan plan = planner_->create(parsed, memories);
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
                    result = executor_->execute(task, results);
                    evaluation = evaluator_->evaluate(task, result);
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
                    plan = planner_->replan(parsed, plan, result, evaluation);
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
        auto aggregate = aggregator_->aggregate(results);
        if (!aggregate.at("contradictions").empty())
            throw std::runtime_error(
                "aggregated results contain contradictions");
        response.text = reasoner_->final_response(parsed, aggregate);
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
