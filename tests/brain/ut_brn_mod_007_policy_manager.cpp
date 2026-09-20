#include "TestSupport.hpp"
#include "brain/policy/PolicyManager.hpp"

using namespace ai::brain;

namespace {
Policy policy(PolicyId id, std::string goalType, int priority) {
    Action action;
    action.type = "act";
    action.timeout = Duration{10};
    return {id,
            "policy-" + std::to_string(id),
            std::move(goalType),
            {},
            {action},
            priority};
}
} // namespace

int main() {
    brain_test::Suite t{"UT-BRN-MOD-007"};
    PolicyManager policies;
    t.expect(policies.add(policy(1, "inspect", 1)), "UT-007-001",
             "policy added");
    Goal inspect{1, "inspect"};
    t.expect(policies.findApplicable(inspect, {}).size() == 1, "UT-007-002",
             "matching policy returned");
    Goal other{2, "other"};
    t.expect(policies.findApplicable(other, {}).empty(), "UT-007-003",
             "nonmatching policy excluded");
    policies.add(policy(2, "inspect", 10));
    t.expect(policies.findApplicable(inspect, {}).front().id == 2, "UT-007-004",
             "priority ordering");
    t.expect(policies.recordResult(1, true), "UT-007-005",
             "success statistics updated");
    auto before = policies.findApplicable(inspect, {});
    t.expect(policies.recordResult(1, false), "UT-007-006",
             "failure statistics updated");
    auto changed = policy(1, "inspect", 20);
    changed.version = before.back().version;
    t.expect(policies.update(changed) &&
                 policies.findApplicable(inspect, {}).front().version ==
                     changed.version + 1,
             "UT-007-007", "version incremented on update");
    t.skip("UT-007-008", "PolicyManager has no ConstraintManager dependency");
    return t.finish();
}
