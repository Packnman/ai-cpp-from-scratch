#include "brain/world/WorldStateManager.hpp"
#include <mutex>
namespace ai::brain {
WorldStateManager::WorldStateManager() {
    _state.timestamp = steady_now();
    _state.safetyState.level = SafetyLevel::SafeStop;
}
bool WorldStateManager::update(const SemanticItem &i) {
    if (!valid_semantic_item(i))
        return false;
    std::unique_lock lock(_mutex);
    bool changed = false;
    if (i.type == SemanticType::Perception) {
        auto f = _state.perceptions.find(i.id);
        if (f == _state.perceptions.end() ||
            i.timestamp >= f->second.timestamp) {
            _state.perceptions[i.id] = {i.id,         "semantic",  {},
                                        i.confidence, i.timestamp, i.valid,
                                        i.attributes};
            changed = true;
        }
    } else if (i.type == SemanticType::Condition) {
        auto f = _state.conditions.find(i.id);
        if (f == _state.conditions.end() ||
            i.timestamp >= f->second.timestamp) {
            _state.conditions[i.id] = {i.id,         "semantic",  i.valid,
                                       i.confidence, i.timestamp, i.attributes};
            changed = true;
        }
    } else if (i.type == SemanticType::RobotState &&
               i.timestamp >= _state.robotState.timestamp) {
        _state.robotState = {i.timestamp, i.valid, i.attributes};
        changed = true;
    }
    if (changed) {
        ++_state.version;
        _state.timestamp = i.timestamp;
    }
    return changed;
}
WorldState WorldStateManager::snapshot() const {
    std::shared_lock lock(_mutex);
    return _state;
}
std::uint64_t WorldStateManager::version() const {
    std::shared_lock lock(_mutex);
    return _state.version;
}
std::size_t WorldStateManager::expire(TimePoint now, Duration ttl) {
    std::unique_lock lock(_mutex);
    std::size_t n = 0;
    for (auto &[id, v] : _state.perceptions) {
        (void)id;
        if (v.valid && now >= v.timestamp && now - v.timestamp > ttl) {
            v.valid = false;
            ++n;
        }
    }
    for (auto &[id, v] : _state.conditions) {
        (void)id;
        if (v.active && now >= v.timestamp && now - v.timestamp > ttl) {
            v.active = false;
            ++n;
        }
    }
    if (n) {
        ++_state.version;
        _state.timestamp = now;
    }
    return n;
}
bool WorldStateManager::updateSafety(const SafetyState &state) {
    std::unique_lock lock(_mutex);
    if (state.timestamp < _state.safetyState.timestamp)
        return false;
    _state.safetyState = state;
    _state.timestamp = state.timestamp;
    ++_state.version;
    return true;
}
bool WorldStateManager::updateCommunication(const CommunicationState &state) {
    std::unique_lock lock(_mutex);
    if (state.timestamp < _state.communicationState.timestamp)
        return false;
    _state.communicationState = state;
    _state.timestamp = state.timestamp;
    ++_state.version;
    return true;
}
} // namespace ai::brain
