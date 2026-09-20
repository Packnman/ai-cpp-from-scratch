#include "brain/input/InputAdapter.hpp"
#include <numeric>
namespace ai::brain {
namespace {
BrainError input_error(std::string text,
                       RecoveryAction action = RecoveryAction::Ignore) {
    return {1,     ErrorLevel::Warning, 1, steady_now(), std::move(text), {},
            action};
}
} // namespace
InputAdapter::InputAdapter(InputAdapterConfig c, std::shared_ptr<ILogManager> l)
    : _config(c), _logger(std::move(l)) {
    if (!_config.queueCapacity)
        _config.queueCapacity = 1;
}
std::size_t InputAdapter::queue_index(InputSource s) noexcept {
    switch (s) {
    case InputSource::Safety:
        return 0;
    case InputSource::Control:
        return 1;
    case InputSource::Sensor:
        return 2;
    case InputSource::HumanInterface:
        return 3;
    default:
        return 4;
    }
}
InputStatus InputAdapter::validate(const ExternalMessage &m,
                                   TimePoint now) const {
    const auto source = static_cast<unsigned>(m.source);
    const auto type = static_cast<unsigned>(m.type);
    if (source > static_cast<unsigned>(InputSource::ExternalAI) ||
        type > static_cast<unsigned>(BrainInputType::ExternalAIResult) ||
        m.timestamp == TimePoint{} ||
        m.schemaVersion != _config.schemaVersion || !m.sequence ||
        std::holds_alternative<std::monostate>(m.payload))
        return InputStatus::Invalid;
    const auto age = m.source == InputSource::Safety ? _config.safetyMaxAge
                                                     : _config.defaultMaxAge;
    return m.timestamp > now || now - m.timestamp > age ? InputStatus::Stale
                                                        : InputStatus::Valid;
}
BrainInput InputAdapter::normalize(const ExternalMessage &m, InputStatus s) {
    return {_nextId++, m.source, m.timestamp, s, m.type, m.payload};
}
Result<void> InputAdapter::push(const ExternalMessage &m) {
    const auto now = steady_now();
    const auto status = validate(m, now);
    if (status != InputStatus::Valid) {
        auto e = input_error(
            status == InputStatus::Stale ? "stale external message"
                                         : "invalid external message",
            m.source == InputSource::Safety ? RecoveryAction::SafeStop
                                            : RecoveryAction::Ignore);
        if (_logger)
            _logger->report(e);
        return Result<void>::failure(std::move(e));
    }
    std::lock_guard lock(_mutex);
    auto &q = _queues[queue_index(m.source)];
    if (q.size() >= _config.queueCapacity) {
        auto e = input_error("input queue overflow", RecoveryAction::Retry);
        if (_logger)
            _logger->report(e);
        return Result<void>::failure(std::move(e));
    }
    auto in = normalize(m, status);
    q.push_back(in);
    if (_logger) {
        TraceEvent e;
        e.id = in.id;
        e.type = TraceType::Input;
        e.timestamp = now;
        e.source = 1;
        e.data.emplace("accepted", true);
        _logger->log(e);
    }
    return Result<void>::success();
}
std::vector<BrainInput> InputAdapter::poll() {
    std::lock_guard lock(_mutex);
    std::vector<BrainInput> out;
    for (auto &q : _queues)
        while (!q.empty()) {
            out.push_back(std::move(q.front()));
            q.pop_front();
        }
    return out;
}
std::size_t InputAdapter::size() const {
    std::lock_guard lock(_mutex);
    return std::accumulate(_queues.begin(), _queues.end(), std::size_t{},
                           [](auto n, const auto &q) { return n + q.size(); });
}
} // namespace ai::brain
