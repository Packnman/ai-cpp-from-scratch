#pragma once
#include "brain/common/Dtos.hpp"
#include <map>
#include <mutex>
namespace ai::brain {
class IGoalManager {
    public:
        virtual ~IGoalManager() = default;
        virtual GoalId add(Goal) = 0;
        virtual bool update(const Goal &) = 0;
        virtual bool cancel(GoalId) = 0;
        virtual bool complete(GoalId, bool) = 0;
        virtual std::optional<Goal> activeGoal() const = 0;
        virtual std::vector<Goal> pendingGoals() const = 0;
        virtual std::optional<Goal> get(GoalId) const = 0;
};
class GoalManager final : public IGoalManager {
    public:
        GoalId add(Goal) override;
        bool update(const Goal &) override;
        bool cancel(GoalId) override;
        bool complete(GoalId, bool) override;
        std::optional<Goal> activeGoal() const override;
        std::vector<Goal> pendingGoals() const override;
        std::optional<Goal> get(GoalId) const override;

    private:
        void select_active_locked();
        static int score(const Goal &) noexcept;
        mutable std::mutex _mutex;
        std::map<GoalId, Goal> _goals;
        GoalId _nextId{1};
};
} // namespace ai::brain
