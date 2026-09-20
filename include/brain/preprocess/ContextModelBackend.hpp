#pragma once

#include "brain/common/Time.hpp"

#include <cstddef>
#include <deque>
#include <stdexcept>
#include <string>

namespace ai::brain {

enum class ContextModelStatus {
    Succeeded,
    Timeout,
    ConnectionFailure,
    HttpError,
    InvalidResponse,
    Failed
};

struct ContextModelRequest {
        std::string text;
};

struct ContextModelResponse {
        ContextModelStatus status{ContextModelStatus::Failed};
        std::string content;
        std::string error;
};

class IContextModelBackend {
    public:
        virtual ~IContextModelBackend() = default;
        virtual ContextModelResponse infer(const ContextModelRequest &) = 0;
};

class FakeContextModelBackend final : public IContextModelBackend {
    public:
        explicit FakeContextModelBackend(ContextModelResponse response = {});
        ContextModelResponse infer(const ContextModelRequest &) override;
        void enqueue(ContextModelResponse);
        void throwOnInfer(bool enabled) noexcept { _throwOnInfer = enabled; }
        std::size_t calls() const noexcept { return _calls; }
        const ContextModelRequest &lastRequest() const noexcept {
            return _lastRequest;
        }

    private:
        ContextModelResponse _defaultResponse;
        std::deque<ContextModelResponse> _responses;
        ContextModelRequest _lastRequest;
        std::size_t _calls{};
        bool _throwOnInfer{};
};

struct QwenContextBackendConfig {
        std::string endpoint{"http://127.0.0.1:8000/v1/chat/completions"};
        std::string model{"Qwen/Qwen3-4B-Instruct-2507"};
        std::string apiKey;
        Duration timeout{Duration{5'000}};
        double temperature{0.0};
        std::size_t maxTokens{512};
        std::size_t maxResponseBytes{65'536};
};

class QwenContextBackend final : public IContextModelBackend {
    public:
        explicit QwenContextBackend(QwenContextBackendConfig = {});
        ContextModelResponse infer(const ContextModelRequest &) override;
        const QwenContextBackendConfig &config() const noexcept {
            return _config;
        }

    private:
        QwenContextBackendConfig _config;
};

} // namespace ai::brain
