#include "ai/agent/agent.h"
#include "ai/agent/context_builder.h"
#include "ai/agent/reasoner.h"

#include <limits>
#include <stdexcept>

namespace ai::agent {
Agent::Agent(std::shared_ptr<IInputParser> a, std::shared_ptr<IRouter> b,
             std::shared_ptr<IPlanner> c, std::shared_ptr<IExecutor> d,
             std::shared_ptr<IEvaluator> e, std::shared_ptr<IAggregator> f,
             std::shared_ptr<IMemoryManager> g, std::shared_ptr<IReasoner> h,
             std::shared_ptr<ai::ner::IEntityExtractor> entity_extractor)
    : _parser(std::move(a)), _router(std::move(b)), _planner(std::move(c)),
      _executor(std::move(d)), _evaluator(std::move(e)),
      _aggregator(std::move(f)), _memory(std::move(g)), _reasoner(std::move(h)),
      _entity_extractor(std::move(entity_extractor)) {
    if (!_parser || !_router || !_planner || !_executor || !_evaluator ||
        !_aggregator || !_memory || !_reasoner)
        throw std::invalid_argument("all Agent components are required");
}

void Agent::finish_turn(const ParsedInput &p, std::string_view response) {
    auto candidates = p.intent == "comparison"
                          ? std::vector<MemoryCandidate>{}
                          : _reasoner->memory_candidates(p, response);
    std::vector<MemoryCandidate> accepted;
    for (auto &c : candidates)
        if (c.importance >= 0.6 && c.confidence >= 0.5 && !c.content.empty() &&
            c.content.size() <= 16 * 1024)
            accepted.push_back(std::move(c));
    if (!accepted.empty())
        _memory->store_batch(accepted);
    const auto recent_input =
        p.intent == "comparison"
            ? "[structured comparison " +
                  p.arguments.value("scenario_id", std::string("unknown")) + "]"
            : p.raw;
    _recent.push_back(
        {recent_input, std::string(response), p.entity_candidates});
}

void Agent::compress_for_chat(const ParsedInput &parsed,
                              const std::vector<MemoryRecord> &memories) {
    if (_recent.size() <= 2)
        return;
    ContextInput full;
    full.current_input = parsed.raw;
    full.goal = parsed.goal;
    full.current_task = "Natural Japanese conversation";
    full.constraints = parsed.constraints;
    full.memories = memories;
    full.recent = _recent;
    full.summary = _summary;
    full.require_summary = true;
    const auto prompt = ContextBuilder(std::numeric_limits<std::size_t>::max(),
                                       [this](std::string_view text) {
                                           return _reasoner->token_count(text);
                                       })
                            .build(full);
    if (_reasoner->token_count(prompt) <= mode_prompt_tokens(ModelMode::Chat))
        return;
    for (std::size_t count = _recent.size() - 2; count > 0; --count) {
        std::vector<ConversationTurn> prefix(_recent.begin(),
                                             _recent.begin() + count);
        try {
            auto updated = _reasoner->summarize(prefix, _summary);
            if (!validate_structured_summary(updated,
                                             _reasoner->token_count(updated)))
                return;
            _summary = std::move(updated);
            _recent.erase(_recent.begin(), _recent.begin() + count);
            return;
        } catch (const std::length_error &) {
            continue;
        } catch (const std::exception &) {
            return;
        }
    }
}
AgentResponse Agent::process(std::string_view raw) {
    AgentResponse response;
    try {
        auto parsed = _parser->parse(raw);
        if (_entity_extractor)
            parsed.entity_candidates = _entity_extractor->extract(
                raw, "user:" + std::to_string(++_turn_sequence));
        auto type = _router->route(parsed);
        auto memories = _memory->retrieve(parsed.raw, std::nullopt, 8);
        if (type == RequestType::SimpleConversation)
            compress_for_chat(parsed, memories);
        AgentState state;
        state.request_type = type;
        state.input = parsed;
        state.constraints = parsed.constraints;
        state.retrieved_memories = memories;
        state.recent_conversation = _recent;
        state.conversation_summary = _summary;
        state.entity_candidates = parsed.entity_candidates;
        state.discussion_state = _discussion_state;
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
        for (const auto &item : aggregate.at("results"))
            if (item.at("value").is_object() &&
                item.at("value").contains("discussion_state"))
                state.discussion_state =
                    item.at("value").at("discussion_state");
        if (!state.discussion_state.is_null())
            _discussion_state = state.discussion_state;
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
