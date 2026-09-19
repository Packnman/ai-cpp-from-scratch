#include "brain/planning/Planner.hpp"
#include <algorithm>
#include <functional>
#include <set>
namespace ai::brain {
namespace {
BrainError err(std::string s) {
    return {1,  ErrorLevel::Error,     8, steady_now(), std::move(s),
            {}, RecoveryAction::Replan};
}
bool precondition_satisfied(const ConditionExpression &condition,
                            const WorldState &world) {
    if (condition.expression.empty())
        return true;
    if (condition.expression == "safety_normal")
        return world.safetyState.valid &&
               world.safetyState.level == SafetyLevel::Normal;
    if (condition.expression == "condition_active") {
        const auto found = condition.arguments.find("id");
        if (found == condition.arguments.end())
            return false;
        ConditionId id{};
        if (std::holds_alternative<std::uint64_t>(found->second))
            id = std::get<std::uint64_t>(found->second);
        else if (std::holds_alternative<std::int64_t>(found->second) &&
                 std::get<std::int64_t>(found->second) > 0)
            id =
                static_cast<ConditionId>(std::get<std::int64_t>(found->second));
        const auto state = world.conditions.find(id);
        return state != world.conditions.end() && state->second.active;
    }
    return false;
}
} // namespace
Result<void> PlanValidator::validate(const ActionPlan &p,
                                     const IConstraintManager &cm) {
    if (!valid_action_plan(p))
        return Result<void>::failure(err("invalid plan schema"));
    std::map<ActionId, std::vector<ActionId>> edges;
    for (const auto &n : p.actions)
        edges[n.action.id] = n.dependencies;
    std::set<ActionId> visiting, done;
    std::function<bool(ActionId)> cycle = [&](ActionId id) {
        if (visiting.contains(id))
            return true;
        if (done.contains(id))
            return false;
        visiting.insert(id);
        for (auto d : edges[id])
            if (cycle(d))
                return true;
        visiting.erase(id);
        done.insert(id);
        return false;
    };
    for (const auto &[id, deps] : edges) {
        (void)deps;
        if (cycle(id))
            return Result<void>::failure(err("cyclic dependency"));
    }
    if (!cm.evaluate(p).allowed)
        return Result<void>::failure(err("critical constraint violation"));
    return Result<void>::success();
}
RuleBasedPlanner::RuleBasedPlanner(const IConstraintManager &c)
    : _constraints(c) {}
Result<PlanningResult> RuleBasedPlanner::plan(const PlanningContext &c) {
    if (!valid_goal(c.goal))
        return Result<PlanningResult>::failure(err("invalid goal"));
    if (c.policies.empty())
        return Result<PlanningResult>::failure(err("no applicable policy"));
    const auto &p = c.policies.front();
    ActionPlan plan;
    plan.id = _nextPlanId++;
    plan.goalId = c.goal.id;
    plan.createdAt = steady_now();
    plan.worldStateVersion = c.worldState.version;
    plan.version = 1;
    ActionId prev = 0;
    for (auto action : p.actionTemplate) {
        if (!std::ranges::all_of(
                action.preconditions, [&](const auto &condition) {
                    return precondition_satisfied(condition, c.worldState);
                }))
            return Result<PlanningResult>::failure(
                err("action precondition not satisfied"));
        action.id = _nextActionId++;
        if (action.target == std::nullopt)
            action.target = c.goal.target;
        if (action.timeout.count() <= 0)
            action.timeout = Duration{1000};
        plan.actions.push_back(
            {std::move(action),
             prev ? std::vector<ActionId>{prev} : std::vector<ActionId>{}});
        prev = plan.actions.back().action.id;
    }
    auto valid = PlanValidator::validate(plan, _constraints);
    if (!valid)
        return Result<PlanningResult>::failure(valid.error());
    auto eval = _constraints.evaluate(plan);
    plan.constraints = eval.applied;
    return Result<PlanningResult>::success(
        {std::move(plan),
         double(p.priority) + p.successRate - eval.softPenalty,
         {}});
}
Result<PlanningResult> TransformerPlanner::plan(const PlanningContext &) {
    return Result<PlanningResult>::failure(
        err("TransformerPlanner stub: no model bundle loaded"));
}
} // namespace ai::brain
