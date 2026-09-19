#include "brain/external/ExternalAIAdapter.hpp"
namespace ai::brain {
RequestId FakeExternalAI::submit(ExternalAIRequest r) {
    if (!r.id)
        r.id = _nextId++;
    if (r.createdAt == TimePoint{})
        r.createdAt = steady_now();
    if (!r.goalId || r.schemaVersion != 1 || r.timeout.count() < 0)
        return 0;
    if (_pending.contains(r.id))
        return 0;
    _pending.emplace(r.id,
                     Pending{std::move(r), ExternalRequestState::Waiting});
    return r.id;
}
std::optional<ExternalAIResponse> FakeExternalAI::poll(RequestId id,
                                                       TimePoint now) {
    auto i = _pending.find(id);
    if (i == _pending.end() ||
        i->second.state == ExternalRequestState::Cancelled)
        return {};
    auto &r = i->second.request;
    if (now - r.createdAt >= r.timeout) {
        i->second.state = ExternalRequestState::Timeout;
        return ExternalAIResponse{
            id, ResponseStatus::Timeout, now, {}, {}, "fake", "1"};
    }
    if (now - r.createdAt < _latency)
        return {};
    i->second.state =
        _fail ? ExternalRequestState::Failed : ExternalRequestState::Succeeded;
    return ExternalAIResponse{
        id,
        _fail ? ResponseStatus::Failed : ResponseStatus::Succeeded,
        now,
        r.payload,
        _fail ? std::optional<float>{} : std::optional<float>{1.F},
        "fake",
        "1"};
}
void FakeExternalAI::cancel(RequestId id) {
    auto i = _pending.find(id);
    if (i != _pending.end())
        i->second.state = ExternalRequestState::Cancelled;
}
} // namespace ai::brain
