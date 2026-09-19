#pragma once
#include "brain/execution/ExecutionManager.hpp"
#include "brain/external/ExternalAIAdapter.hpp"
#include "brain/input/InputAdapter.hpp"
#include "brain/postprocess/PostProcessor.hpp"
#include "brain/preprocess/Preprocessor.hpp"
#include "brain/world/WorldStateManager.hpp"
#include <memory>
namespace ai::brain {
struct BrainCycleResult {
        std::size_t inputs{};
        std::size_t semantics{};
        std::size_t actionResults{};
        std::optional<PlanId> planId;
        std::vector<BrainError> errors;
};
class BrainSystem final {
    public:
        explicit BrainSystem(std::string memoryPath = ":memory:");
        Result<void> push(const ExternalMessage &);
        BrainCycleResult runOnce(TimePoint now = steady_now());
        IWorldStateManager &world() noexcept { return _world; }
        IGoalManager &goals() noexcept { return _goals; }
        IMemoryManager &memory() noexcept { return _memory; }
        IPolicyManager &policies() noexcept { return _policies; }
        IConstraintManager &constraints() noexcept { return _constraints; }
        IExecutionManager &execution() noexcept { return _execution; }

    private:
        std::shared_ptr<InMemoryLogManager> _log;
        InputAdapter _input;
        Preprocessor _preprocessor;
        WorldStateManager _world;
        GoalManager _goals;
        ConstraintManager _constraints;
        MemoryManager _memory;
        PolicyManager _policies;
        RuleBasedPlanner _planner;
        std::shared_ptr<MockControlDispatcher> _control;
        ExecutionManager _execution;
        PostProcessor _post;
        std::optional<ActionPlan> _currentPlan;
        std::optional<PolicyId> _currentPolicy;
};
} // namespace ai::brain
