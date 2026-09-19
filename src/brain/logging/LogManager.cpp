#include "brain/logging/LogManager.hpp"
namespace ai::brain {
void LogManager::log(const TraceEvent &event) noexcept {
    try {
        if (_backend)
            _backend->write(event);
        else
            ++_dropped;
    } catch (...) {
        ++_dropped;
    }
}
void LogManager::report(const BrainError &error) noexcept {
    try {
        if (_backend)
            _backend->write(error);
        else
            ++_dropped;
    } catch (...) {
        ++_dropped;
    }
}
void LogManager::flush() noexcept {
    try {
        if (_backend)
            _backend->flush();
    } catch (...) {
        ++_dropped;
    }
}
void InMemoryLogManager::log(const TraceEvent &e) {
    std::lock_guard lock(_mutex);
    _events.push_back(e);
}
void InMemoryLogManager::report(const BrainError &e) {
    std::lock_guard lock(_mutex);
    _errors.push_back(e);
}
std::vector<TraceEvent> InMemoryLogManager::events() const {
    std::lock_guard lock(_mutex);
    return _events;
}
std::vector<BrainError> InMemoryLogManager::errors() const {
    std::lock_guard lock(_mutex);
    return _errors;
}
} // namespace ai::brain
