#include "brain/constraint/ConstraintManager.hpp"
#include <algorithm>
#include <mutex>
namespace ai::brain {
namespace {
bool applies(const Constraint &c, const ActionPlan &p) {
    switch (c.scope.type) {
    case ConstraintScopeType::Global:
        return true;
    case ConstraintScopeType::Goal:
        return c.scope.targetId == p.goalId;
    case ConstraintScopeType::Plan:
        return c.scope.targetId == p.id;
    case ConstraintScopeType::Action:
        return c.scope.targetId &&
               std::ranges::any_of(p.actions, [&](const ActionNode &n) {
                   return n.action.id == *c.scope.targetId;
               });
    default:
        return true;
    }
}
bool violated(const Constraint &c, const ActionPlan &p) {
    auto &name = c.expression.expression;
    if (name == "deny_all")
        return true;
    if (name == "deny_action_type") {
        auto i = c.expression.arguments.find("type");
        if (i == c.expression.arguments.end() ||
            !std::holds_alternative<std::string>(i->second))
            return true;
        auto &type = std::get<std::string>(i->second);
        return std::ranges::any_of(p.actions, [&](const ActionNode &n) {
            return n.action.type == type;
        });
    }
    if (name == "max_action_priority") {
        auto i = c.expression.arguments.find("value");
        if (i == c.expression.arguments.end() ||
            !std::holds_alternative<std::int64_t>(i->second))
            return true;
        auto max = std::get<std::int64_t>(i->second);
        return std::ranges::any_of(p.actions, [&](const ActionNode &n) {
            return n.action.priority > max;
        });
    }
    return false;
}
} // namespace
bool ConstraintManager::add(const Constraint &c) {
    if (!valid_constraint(c))
        return false;
    std::unique_lock lock(_mutex);
    return _constraints.emplace(c.id, c).second;
}
bool ConstraintManager::update(const Constraint &c) {
    if (!valid_constraint(c))
        return false;
    std::unique_lock lock(_mutex);
    auto f = _constraints.find(c.id);
    if (f == _constraints.end())
        return false;
    f->second = c;
    return true;
}
bool ConstraintManager::remove(ConstraintId id) {
    std::unique_lock lock(_mutex);
    return _constraints.erase(id) != 0;
}
std::vector<Constraint> ConstraintManager::active(TimePoint now) const {
    std::shared_lock lock(_mutex);
    std::vector<Constraint> r;
    for (const auto &[id, c] : _constraints) {
        (void)id;
        if (c.active && (!c.expiresAt || now < *c.expiresAt))
            r.push_back(c);
    }
    return r;
}
ConstraintEvaluation ConstraintManager::evaluate(const ActionPlan &p,
                                                 TimePoint now) const {
    ConstraintEvaluation r;
    for (const auto &c : active(now)) {
        if (!applies(c, p))
            continue;
        r.applied.push_back(c.id);
        if (!violated(c, p))
            continue;
        r.violated.push_back(c.id);
        if (c.critical)
            r.allowed = false;
        else
            r.softPenalty += 1;
    }
    return r;
}
} // namespace ai::brain
