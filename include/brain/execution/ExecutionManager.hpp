#pragma once
#include "brain/common/Dtos.hpp"
#include "brain/common/Result.hpp"
#include <map>
#include <memory>
#include <optional>
#include <set>
namespace ai::brain {
enum class ActionState {
    Pending,
    Ready,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Timeout
};
enum class ExecutionStatus {
    Idle,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    EmergencyStopped
};
struct ActionCommand {
        ActionId id{};
        ActionType type;
        std::optional<SemanticId> target;
        AttributeMap parameters;
        Duration timeout{};
        int priority{};
};
struct ActionRuntimeState {
        ActionState state{ActionState::Pending};
        TimePoint startedAt{};
        std::optional<ActionResult> result;
};
class IControlDispatcher {
    public:
        virtual ~IControlDispatcher() = default;
        virtual std::optional<ActionResult> dispatch(const ActionCommand &) = 0;
        virtual void cancel(ActionId) = 0;
};
class MockControlDispatcher final : public IControlDispatcher {
    public:
        explicit MockControlDispatcher(
            bool immediate = true,
            ActionResultCode result = ActionResultCode::Succeeded)
            : _immediate(immediate), _result(result) {}
        std::optional<ActionResult> dispatch(const ActionCommand &) override;
        void cancel(ActionId) override;
        const std::vector<ActionCommand> &commands() const noexcept {
            return _commands;
        }

    private:
        bool _immediate;
        ActionResultCode _result;
        std::vector<ActionCommand> _commands;
};
class IExecutionManager {
    public:
        virtual ~IExecutionManager() = default;
        virtual Result<void> submit(const ActionPlan &) = 0;
        virtual std::vector<ActionResult>
        tick(TimePoint now = steady_now()) = 0;
        virtual bool update(const ActionResult &) = 0;
        virtual void cancel(PlanId) = 0;
        virtual void emergencyStop() = 0;
        virtual ExecutionStatus status() const noexcept = 0;
};
class ExecutionManager final : public IExecutionManager {
    public:
        explicit ExecutionManager(std::shared_ptr<IControlDispatcher>);
        Result<void> submit(const ActionPlan &) override;
        std::vector<ActionResult> tick(TimePoint now = steady_now()) override;
        bool update(const ActionResult &) override;
        void cancel(PlanId) override;
        void emergencyStop() override;
        ExecutionStatus status() const noexcept override { return _status; }
        const std::map<ActionId, ActionRuntimeState> &actions() const noexcept {
            return _runtime;
        }

    private:
        bool dependencies_done(const ActionNode &) const;
        bool resources_available(const Action &) const;
        void acquire(const Action &);
        void release(const Action &);
        const Action *find_action(ActionId) const;
        void finish(const ActionResult &);
        ActionPlan _plan;
        std::shared_ptr<IControlDispatcher> _control;
        std::map<ActionId, ActionRuntimeState> _runtime;
        std::map<ResourceId, std::pair<ResourceAccess, std::set<ActionId>>>
            _resources;
        ExecutionStatus _status{ExecutionStatus::Idle};
};
} // namespace ai::brain
