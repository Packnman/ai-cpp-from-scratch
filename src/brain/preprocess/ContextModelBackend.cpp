#include "brain/preprocess/ContextModelBackend.hpp"

#include <utility>

namespace ai::brain {

FakeContextModelBackend::FakeContextModelBackend(ContextModelResponse response)
    : _defaultResponse(std::move(response)) {}

ContextModelResponse
FakeContextModelBackend::infer(const ContextModelRequest &request) {
    ++_calls;
    _lastRequest = request;
    if (_throwOnInfer)
        throw std::runtime_error("injected context backend failure");
    if (_responses.empty())
        return _defaultResponse;
    auto response = std::move(_responses.front());
    _responses.pop_front();
    return response;
}

void FakeContextModelBackend::enqueue(ContextModelResponse response) {
    _responses.push_back(std::move(response));
}

} // namespace ai::brain
