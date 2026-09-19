#pragma once
#include "brain/common/Dtos.hpp"
#include "brain/common/Result.hpp"
#include <map>
#include <optional>
namespace ai::brain {
enum class ExternalTaskType {
    HighLevelRecognition,
    LanguageInference,
    KnowledgeQuery,
    HeavyPlanning,
    Embedding,
    ModelSpecificTask
};
enum class ResponseStatus { Succeeded, Failed, Timeout, Cancelled };
enum class ExternalRequestState {
    Created,
    Waiting,
    Succeeded,
    Timeout,
    Failed,
    Cancelled
};
struct ExternalAIRequest {
        RequestId id{};
        ExternalTaskType type{ExternalTaskType::ModelSpecificTask};
        TimePoint createdAt{};
        Duration timeout{1000};
        Payload payload;
        GoalId goalId{};
        std::optional<PlanId> planId;
        std::uint64_t worldStateVersion{};
        std::uint32_t schemaVersion{1};
        bool idempotent{true};
};
struct ExternalAIResponse {
        RequestId requestId{};
        ResponseStatus status{ResponseStatus::Failed};
        TimePoint receivedAt{};
        Payload payload;
        std::optional<float> confidence;
        std::string modelId;
        std::string modelVersion;
};
class IExternalAI {
    public:
        virtual ~IExternalAI() = default;
        virtual RequestId submit(ExternalAIRequest) = 0;
        virtual std::optional<ExternalAIResponse>
        poll(RequestId, TimePoint now = steady_now()) = 0;
        virtual void cancel(RequestId) = 0;
};
class FakeExternalAI final : public IExternalAI {
    public:
        explicit FakeExternalAI(Duration latency = Duration{0},
                                bool fail = false)
            : _latency(latency), _fail(fail) {}
        RequestId submit(ExternalAIRequest) override;
        std::optional<ExternalAIResponse>
        poll(RequestId, TimePoint now = steady_now()) override;
        void cancel(RequestId) override;

    private:
        struct Pending {
                ExternalAIRequest request;
                ExternalRequestState state;
        };
        Duration _latency;
        bool _fail;
        RequestId _nextId{1};
        std::map<RequestId, Pending> _pending;
};
} // namespace ai::brain
