#include "brain/goal/GoalManager.hpp"
#include <algorithm>
namespace ai::brain {
int GoalManager::score(const Goal &g) noexcept {
    return g.priority + (g.source == GoalSource::Safety ? 1000000 : 0);
}
void GoalManager::select_active_locked() {
    Goal *cur = nullptr, *best = nullptr;
    for (auto &[id, g] : _goals) {
        (void)id;
        if (g.status == GoalStatus::Active)
            cur = &g;
        if (g.status == GoalStatus::Pending ||
            g.status == GoalStatus::Suspended)
            if (!best || score(g) > score(*best) ||
                (score(g) == score(*best) && g.id < best->id))
                best = &g;
    }
    if (cur && (!best || score(*cur) >= score(*best)))
        return;
    auto now = steady_now();
    if (cur) {
        cur->status = GoalStatus::Suspended;
        cur->updatedAt = now;
    }
    if (best) {
        best->status = GoalStatus::Active;
        best->updatedAt = now;
    }
}
GoalId GoalManager::add(Goal g) {
    std::lock_guard lock(_mutex);
    if (!g.id)
        g.id = _nextId++;
    else
        _nextId = std::max(_nextId, g.id + 1);
    if (!valid_goal(g) || _goals.contains(g.id) ||
        (g.parentGoal && !_goals.contains(*g.parentGoal)))
        return 0;
    const auto id = g.id;
    const auto now = steady_now();
    if (g.createdAt == TimePoint{})
        g.createdAt = now;
    g.updatedAt = now;
    g.status = GoalStatus::Pending;
    _goals.emplace(id, std::move(g));
    select_active_locked();
    return id;
}
bool GoalManager::update(const Goal &g) {
    if (!valid_goal(g))
        return false;
    std::lock_guard lock(_mutex);
    auto f = _goals.find(g.id);
    if (f == _goals.end())
        return false;
    f->second = g;
    f->second.updatedAt = steady_now();
    select_active_locked();
    return true;
}
bool GoalManager::cancel(GoalId id) {
    std::lock_guard lock(_mutex);
    auto f = _goals.find(id);
    if (f == _goals.end())
        return false;
    f->second.status = GoalStatus::Cancelled;
    f->second.updatedAt = steady_now();
    select_active_locked();
    return true;
}
bool GoalManager::complete(GoalId id, bool ok) {
    std::lock_guard lock(_mutex);
    auto f = _goals.find(id);
    if (f == _goals.end())
        return false;
    f->second.status = ok ? GoalStatus::Achieved : GoalStatus::Failed;
    f->second.updatedAt = steady_now();
    select_active_locked();
    return true;
}
std::optional<Goal> GoalManager::activeGoal() const {
    std::lock_guard lock(_mutex);
    for (const auto &[id, g] : _goals) {
        (void)id;
        if (g.status == GoalStatus::Active)
            return g;
    }
    return {};
}
std::vector<Goal> GoalManager::pendingGoals() const {
    std::lock_guard lock(_mutex);
    std::vector<Goal> r;
    for (const auto &[id, g] : _goals) {
        (void)id;
        if (g.status == GoalStatus::Pending ||
            g.status == GoalStatus::Suspended)
            r.push_back(g);
    }
    std::ranges::sort(r, [](const Goal &a, const Goal &b) {
        return score(a) > score(b) || (score(a) == score(b) && a.id < b.id);
    });
    return r;
}
std::optional<Goal> GoalManager::get(GoalId id) const {
    std::lock_guard lock(_mutex);
    auto f = _goals.find(id);
    return f == _goals.end() ? std::optional<Goal>{} : f->second;
}
} // namespace ai::brain
