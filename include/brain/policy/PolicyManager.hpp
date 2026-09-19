#pragma once
#include "brain/common/Dtos.hpp"
#include <map>
#include <shared_mutex>
namespace ai::brain {
enum class PolicySource {
    Predefined,
    Learned,
    ExperienceDerived,
    DomainSpecific
};
struct Policy {
        PolicyId id{};
        std::string name;
        GoalType goalType;
        ConditionExpression applicableCondition;
        std::vector<Action> actionTemplate;
        int priority{};
        float confidence{1};
        float successRate{.5F};
        PolicySource source{PolicySource::Predefined};
        std::uint32_t version{1};
        std::uint64_t successCount{};
        std::uint64_t failureCount{};
};
class IPolicyManager {
    public:
        virtual ~IPolicyManager() = default;
        virtual bool add(const Policy &) = 0;
        virtual bool update(const Policy &) = 0;
        virtual std::vector<Policy>
        findApplicable(const Goal &, const WorldState &) const = 0;
        virtual bool recordResult(PolicyId, bool) = 0;
};
class PolicyManager final : public IPolicyManager {
    public:
        bool add(const Policy &) override;
        bool update(const Policy &) override;
        std::vector<Policy> findApplicable(const Goal &,
                                           const WorldState &) const override;
        bool recordResult(PolicyId, bool) override;

    private:
        mutable std::shared_mutex _mutex;
        std::map<PolicyId, Policy> _policies;
};
} // namespace ai::brain
