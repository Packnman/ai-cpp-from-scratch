#pragma once
#include "brain/common/Result.hpp"
#include "brain/constraint/ConstraintManager.hpp"
#include "brain/memory/MemoryTypes.hpp"
#include "brain/policy/PolicyManager.hpp"
namespace ai::brain {
struct PlanningContext {
        Goal goal;
        WorldState worldState;
        std::vector<Constraint> constraints;
        std::vector<MemoryItem> memories;
        std::vector<Policy> policies;
        std::optional<ActionResult> previousResult;
};
struct PlanningResult {
        ActionPlan plan;
        double score{};
        std::vector<std::string> warnings;
};
class IPlanner {
    public:
        virtual ~IPlanner() = default;
        virtual Result<PlanningResult> plan(const PlanningContext &) = 0;
};
class PlanValidator {
    public:
        static Result<void> validate(const ActionPlan &,
                                     const IConstraintManager &);
};
class RuleBasedPlanner final : public IPlanner {
    public:
        explicit RuleBasedPlanner(const IConstraintManager &);
        Result<PlanningResult> plan(const PlanningContext &) override;

    private:
        const IConstraintManager &_constraints;
        PlanId _nextPlanId{1};
        ActionId _nextActionId{1};
};
class TransformerPlanner final : public IPlanner {
    public:
        Result<PlanningResult> plan(const PlanningContext &) override;
        bool modelLoaded() const noexcept { return false; }
};
} // namespace ai::brain
