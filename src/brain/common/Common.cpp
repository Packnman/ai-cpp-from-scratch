#include "brain/common/Dtos.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace ai::brain {

bool valid_confidence(float confidence) noexcept {
    return std::isfinite(confidence) && confidence >= 0.0F &&
           confidence <= 1.0F;
}

bool valid_semantic_item(const SemanticItem &item) noexcept {
    return valid_id(item.id) && item.type != SemanticType::Unknown &&
           valid_confidence(item.confidence) &&
           (!item.target || valid_id(*item.target));
}

bool valid_brain_input(const BrainInput &input) noexcept {
    return valid_id(input.id) && input.status != InputStatus::Invalid;
}

bool valid_goal(const Goal &goal) noexcept {
    return valid_id(goal.id) && !goal.type.empty() &&
           (!goal.target || valid_id(*goal.target)) &&
           (!goal.parentGoal ||
            (valid_id(*goal.parentGoal) && *goal.parentGoal != goal.id));
}

bool valid_constraint(const Constraint &constraint) noexcept {
    if (!valid_id(constraint.id) || constraint.type.empty() ||
        constraint.expression.expression.empty())
        return false;
    if (constraint.scope.type == ConstraintScopeType::Global)
        return !constraint.scope.targetId.has_value();
    return constraint.scope.targetId && valid_id(*constraint.scope.targetId);
}

bool valid_action(const Action &action) noexcept {
    if (!valid_id(action.id) || action.type.empty() ||
        action.timeout.count() < 0)
        return false;
    std::unordered_set<ResourceId> resources;
    return std::all_of(action.resources.begin(), action.resources.end(),
                       [&](const ResourceRequest &request) {
                           return valid_id(request.id) &&
                                  resources.insert(request.id).second;
                       });
}

bool valid_action_plan(const ActionPlan &plan) noexcept {
    if (!valid_id(plan.id) || !valid_id(plan.goalId) || plan.actions.empty())
        return false;
    std::unordered_set<ActionId> actions;
    for (const auto &node : plan.actions)
        if (!valid_action(node.action) ||
            !actions.insert(node.action.id).second)
            return false;
    for (const auto &node : plan.actions)
        for (const auto dependency : node.dependencies)
            if (dependency == node.action.id || !actions.contains(dependency))
                return false;
    return std::all_of(plan.constraints.begin(), plan.constraints.end(),
                       [](ConstraintId id) { return valid_id(id); });
}

bool valid_action_result(const ActionResult &result) noexcept {
    return valid_id(result.actionId) && result.endTime >= result.startTime;
}

} // namespace ai::brain
