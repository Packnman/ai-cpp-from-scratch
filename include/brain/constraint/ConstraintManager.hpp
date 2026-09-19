#pragma once
#include "brain/common/Dtos.hpp"
#include <map>
#include <shared_mutex>
namespace ai::brain {
struct ConstraintEvaluation {
        bool allowed{true};
        double softPenalty{};
        std::vector<ConstraintId> applied;
        std::vector<ConstraintId> violated;
};
class IConstraintManager {
    public:
        virtual ~IConstraintManager() = default;
        virtual bool add(const Constraint &) = 0;
        virtual bool update(const Constraint &) = 0;
        virtual bool remove(ConstraintId) = 0;
        virtual std::vector<Constraint>
        active(TimePoint now = TimePoint::clock::now()) const = 0;
        virtual ConstraintEvaluation
        evaluate(const ActionPlan &,
                 TimePoint now = TimePoint::clock::now()) const = 0;
};
class ConstraintManager final : public IConstraintManager {
    public:
        bool add(const Constraint &) override;
        bool update(const Constraint &) override;
        bool remove(ConstraintId) override;
        std::vector<Constraint>
        active(TimePoint now = TimePoint::clock::now()) const override;
        ConstraintEvaluation
        evaluate(const ActionPlan &,
                 TimePoint now = TimePoint::clock::now()) const override;

    private:
        mutable std::shared_mutex _mutex;
        std::map<ConstraintId, Constraint> _constraints;
};
} // namespace ai::brain
