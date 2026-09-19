#pragma once
#include "brain/goal/GoalManager.hpp"
#include "brain/memory/MemoryManager.hpp"
#include "brain/planning/Planner.hpp"
#include "brain/policy/PolicyManager.hpp"
namespace ai::brain {
struct LearningSample {
        PlanningContext context;
        GoalId goalId{};
        PlanId planId{};
        Action action;
        ActionResult result;
        float evaluation{};
        TimePoint timestamp{};
};
struct PostProcessInput {
        Goal goal;
        ActionPlan plan;
        ActionResult actionResult;
        WorldState worldState;
        std::optional<PolicyId> policyId;
        bool planComplete{};
};
struct PostProcessResult {
        bool goalUpdated{};
        MemoryId memoryId{};
        bool policyUpdated{};
        bool replanRequired{};
        LearningSample sample;
};
class IPostProcessor {
    public:
        virtual ~IPostProcessor() = default;
        virtual PostProcessResult process(const PostProcessInput &) = 0;
};
class PostProcessor final : public IPostProcessor {
    public:
        PostProcessor(IGoalManager &, IMemoryManager &, IPolicyManager &);
        PostProcessResult process(const PostProcessInput &) override;

    private:
        IGoalManager &_goals;
        IMemoryManager &_memory;
        IPolicyManager &_policies;
};
} // namespace ai::brain
