#include "brain/policy/PolicyManager.hpp"
#include <algorithm>
#include <mutex>
namespace ai::brain {
bool PolicyManager::add(const Policy &p) {
    if (!p.id || p.name.empty() || p.goalType.empty() ||
        p.actionTemplate.empty() || !valid_confidence(p.confidence) ||
        !valid_confidence(p.successRate))
        return false;
    std::unique_lock lock(_mutex);
    return _policies.emplace(p.id, p).second;
}
bool PolicyManager::update(const Policy &p) {
    std::unique_lock lock(_mutex);
    auto i = _policies.find(p.id);
    if (i == _policies.end())
        return false;
    i->second = p;
    ++i->second.version;
    return true;
}
std::vector<Policy> PolicyManager::findApplicable(const Goal &g,
                                                  const WorldState &) const {
    std::shared_lock lock(_mutex);
    std::vector<Policy> r;
    for (const auto &[id, p] : _policies) {
        (void)id;
        if (p.goalType == g.type || p.goalType == "*")
            r.push_back(p);
    }
    std::ranges::sort(r, [](const Policy &a, const Policy &b) {
        auto as = a.priority + a.confidence + a.successRate,
             bs = b.priority + b.confidence + b.successRate;
        return as > bs || (as == bs && a.id < b.id);
    });
    return r;
}
bool PolicyManager::recordResult(PolicyId id, bool ok) {
    std::unique_lock lock(_mutex);
    auto i = _policies.find(id);
    if (i == _policies.end())
        return false;
    if (ok)
        ++i->second.successCount;
    else
        ++i->second.failureCount;
    auto total = i->second.successCount + i->second.failureCount;
    i->second.successRate = float(i->second.successCount) / float(total);
    ++i->second.version;
    return true;
}
} // namespace ai::brain
