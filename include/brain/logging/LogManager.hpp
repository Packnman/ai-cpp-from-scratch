#pragma once
#include "brain/common/Attribute.hpp"
#include "brain/common/Error.hpp"
#include "brain/common/Time.hpp"
#include "brain/common/Types.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
namespace ai::brain {
enum class TraceType {
    Input,
    State,
    Goal,
    Constraint,
    Memory,
    Policy,
    Plan,
    Execution,
    ExternalAI,
    Error
};
struct TraceEvent {
        TraceId id{};
        TraceType type{TraceType::State};
        TimePoint timestamp{};
        std::optional<GoalId> goalId;
        std::optional<PlanId> planId;
        std::optional<ActionId> actionId;
        ModuleId source{};
        AttributeMap data;
};
class ILogManager {
    public:
        virtual ~ILogManager() = default;
        virtual void log(const TraceEvent &) = 0;
        virtual void report(const BrainError &) = 0;
};
class ILogBackend {
    public:
        virtual ~ILogBackend() = default;
        virtual void write(const TraceEvent &) = 0;
        virtual void write(const BrainError &) = 0;
        virtual void flush() = 0;
};
class LogManager final : public ILogManager {
    public:
        explicit LogManager(std::shared_ptr<ILogBackend> backend)
            : _backend(std::move(backend)) {}
        void log(const TraceEvent &) noexcept override;
        void report(const BrainError &) noexcept override;
        void flush() noexcept;
        std::size_t dropped() const noexcept { return _dropped.load(); }

    private:
        std::shared_ptr<ILogBackend> _backend;
        std::atomic_size_t _dropped{};
};
class InMemoryLogManager final : public ILogManager {
    public:
        void log(const TraceEvent &) override;
        void report(const BrainError &) override;
        std::vector<TraceEvent> events() const;
        std::vector<BrainError> errors() const;

    private:
        mutable std::mutex _mutex;
        std::vector<TraceEvent> _events;
        std::vector<BrainError> _errors;
};
} // namespace ai::brain
