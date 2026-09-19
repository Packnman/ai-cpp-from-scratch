#include "brain/memory/MemoryManager.hpp"
#include <algorithm>
namespace ai::brain {
MemoryManager::MemoryManager(std::unique_ptr<IMemoryBackend> b, std::size_t cap)
    : _backend(std::move(b)), _capacity(std::max<std::size_t>(1, cap)) {}
bool MemoryManager::matches(const MemoryItem &i, const MemoryQuery &q) const {
    if (q.type && i.type != *q.type)
        return false;
    if (q.target && i.content.target != q.target)
        return false;
    for (const auto &t : q.tags)
        if (std::ranges::find(i.tags, t) == i.tags.end())
            return false;
    return true;
}
MemoryId MemoryManager::remember(MemoryItem i) {
    std::lock_guard lock(_mutex);
    auto now = steady_now();
    if (!i.id)
        i.id = _nextFallbackId++;
    else
        _nextFallbackId = std::max(_nextFallbackId, i.id + 1);
    if (i.createdAt == TimePoint{})
        i.createdAt = now;
    i.updatedAt = now;
    i.lastAccessed = now;
    i.confidence = std::clamp(i.confidence, 0.F, 1.F);
    i.importance = std::clamp(i.importance, 0.F, 1.F);
    if (_backend && _backend->available()) {
        auto id = _backend->store(i);
        if (id)
            i.id = id;
    }
    if (_shortTerm.size() >= _capacity)
        _shortTerm.pop_front();
    _shortTerm.push_back(i);
    return i.id;
}
std::vector<MemoryItem> MemoryManager::recall(const MemoryQuery &q) {
    std::lock_guard lock(_mutex);
    std::vector<MemoryItem> out;
    if (_backend && _backend->available())
        out = _backend->search(q);
    if (out.empty())
        for (auto i = _shortTerm.rbegin();
             i != _shortTerm.rend() && out.size() < q.maxResults; ++i)
            if (matches(*i, q))
                out.push_back(*i);
    return out;
}
bool MemoryManager::persistent() const noexcept {
    return _backend && _backend->available();
}
std::size_t MemoryManager::shortTermSize() const {
    std::lock_guard lock(_mutex);
    return _shortTerm.size();
}
} // namespace ai::brain
