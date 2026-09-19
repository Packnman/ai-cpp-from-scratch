#include "brain/execution/ExecutionManager.hpp"
#include <algorithm>
namespace ai::brain {
namespace {
BrainError error(std::string s) {
    return {1,  ErrorLevel::Error,     9, steady_now(), std::move(s),
            {}, RecoveryAction::Cancel};
}
} // namespace
std::optional<ActionResult>
MockControlDispatcher::dispatch(const ActionCommand &c) {
    _commands.push_back(c);
    if (!_immediate)
        return {};
    auto now = steady_now();
    return ActionResult{c.id,
                        _result,
                        _result == ActionResultCode::Succeeded ? "mock success"
                                                               : "mock failure",
                        now,
                        now,
                        {}};
}
void MockControlDispatcher::cancel(ActionId id) {
    auto now = steady_now();
    _commands.push_back({id, "cancel", {}, {}, Duration{0}, 0});
    (void)now;
}
ExecutionManager::ExecutionManager(std::shared_ptr<IControlDispatcher> c)
    : _control(std::move(c)) {}
Result<void> ExecutionManager::submit(const ActionPlan &p) {
    if (_status == ExecutionStatus::EmergencyStopped)
        return Result<void>::failure(error("emergency stop active"));
    if (std::ranges::any_of(_runtime, [](const auto &entry) {
            return entry.second.state == ActionState::Running;
        }))
        return Result<void>::failure(
            error("cancelled action has not acknowledged stop"));
    if (!valid_action_plan(p) || !_control)
        return Result<void>::failure(error("invalid execution submission"));
    _plan = p;
    _runtime.clear();
    _resources.clear();
    for (const auto &n : p.actions)
        _runtime[n.action.id] = {};
    _status = ExecutionStatus::Running;
    return Result<void>::success();
}
bool ExecutionManager::dependencies_done(const ActionNode &n) const {
    return std::ranges::all_of(n.dependencies, [&](ActionId id) {
        auto i = _runtime.find(id);
        return i != _runtime.end() && i->second.state == ActionState::Succeeded;
    });
}
bool ExecutionManager::resources_available(const Action &a) const {
    for (const auto &r : a.resources) {
        auto i = _resources.find(r.id);
        if (i == _resources.end())
            continue;
        if (r.access == ResourceAccess::Exclusive ||
            i->second.first == ResourceAccess::Exclusive)
            return false;
    }
    return true;
}
void ExecutionManager::acquire(const Action &a) {
    for (const auto &r : a.resources) {
        auto &v = _resources[r.id];
        v.first = r.access;
        v.second.insert(a.id);
    }
}
void ExecutionManager::release(const Action &a) {
    for (const auto &r : a.resources) {
        auto i = _resources.find(r.id);
        if (i != _resources.end() && i->second.second.erase(a.id) &&
            i->second.second.empty())
            _resources.erase(i);
    }
}
const Action *ExecutionManager::find_action(ActionId id) const {
    for (const auto &n : _plan.actions)
        if (n.action.id == id)
            return &n.action;
    return nullptr;
}
void ExecutionManager::finish(const ActionResult &r) {
    auto i = _runtime.find(r.actionId);
    auto *a = find_action(r.actionId);
    if (i == _runtime.end() || !a)
        return;
    i->second.result = r;
    i->second.state =
        r.result == ActionResultCode::Succeeded   ? ActionState::Succeeded
        : r.result == ActionResultCode::Timeout   ? ActionState::Timeout
        : r.result == ActionResultCode::Cancelled ? ActionState::Cancelled
                                                  : ActionState::Failed;
    release(*a);
    if (_status == ExecutionStatus::Cancelled ||
        _status == ExecutionStatus::EmergencyStopped)
        return;
    if (i->second.state != ActionState::Succeeded)
        _status = ExecutionStatus::Failed;
    else if (std::ranges::all_of(_runtime, [](const auto &v) {
                 return v.second.state == ActionState::Succeeded;
             }))
        _status = ExecutionStatus::Succeeded;
}
std::vector<ActionResult> ExecutionManager::tick(TimePoint now) {
    std::vector<ActionResult> out;
    if (_status != ExecutionStatus::Running)
        return out;
    for (const auto &n : _plan.actions) {
        auto &r = _runtime[n.action.id];
        if (r.state == ActionState::Running &&
            now - r.startedAt >= n.action.timeout) {
            _control->cancel(n.action.id);
            ActionResult x{n.action.id, ActionResultCode::Timeout,
                           "timeout",   r.startedAt,
                           now,         {}};
            finish(x);
            out.push_back(x);
        }
    }
    if (_status != ExecutionStatus::Running)
        return out;
    for (const auto &n : _plan.actions) {
        auto &r = _runtime[n.action.id];
        if (r.state != ActionState::Pending || !dependencies_done(n) ||
            !resources_available(n.action))
            continue;
        r.state = ActionState::Running;
        r.startedAt = now;
        acquire(n.action);
        auto result = _control->dispatch({n.action.id, n.action.type,
                                          n.action.target, n.action.parameters,
                                          n.action.timeout, n.action.priority});
        if (result) {
            if (result->startTime == TimePoint{})
                result->startTime = now;
            if (result->endTime == TimePoint{})
                result->endTime = now;
            finish(*result);
            out.push_back(*result);
        }
    }
    return out;
}
bool ExecutionManager::update(const ActionResult &r) {
    if (!valid_action_result(r) || !_runtime.contains(r.actionId) ||
        _runtime[r.actionId].state != ActionState::Running)
        return false;
    finish(r);
    return true;
}
void ExecutionManager::cancel(PlanId id) {
    if (id != _plan.id)
        return;
    for (auto &[aid, r] : _runtime)
        if (r.state == ActionState::Running)
            _control->cancel(aid);
    for (auto &[aid, r] : _runtime) {
        (void)aid;
        if (r.state == ActionState::Pending)
            r.state = ActionState::Cancelled;
    }
    _status = ExecutionStatus::Cancelled;
}
void ExecutionManager::emergencyStop() {
    if (_plan.id)
        cancel(_plan.id);
    _status = ExecutionStatus::EmergencyStopped;
}
} // namespace ai::brain
