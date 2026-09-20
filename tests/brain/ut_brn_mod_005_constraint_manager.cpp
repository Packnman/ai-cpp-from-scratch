#include "TestSupport.hpp"
#include "brain/constraint/ConstraintManager.hpp"

using namespace ai::brain;

namespace {
ActionPlan plan(std::string type = "move") {
    Action action{1, std::move(type)};
    action.timeout = Duration{10};
    return {1, 1, {{action, {}}}};
}
Constraint constraint(ConstraintId id, bool critical, TimePoint now) {
    return {id,
            "rule",
            critical,
            true,
            {},
            {"deny_action_type", {{"type", std::string("move")}}},
            ConstraintSource::Safety,
            {},
            now};
}
} // namespace

int main() {
    brain_test::Suite t{"UT-BRN-MOD-005"};
    const auto now = steady_now();
    ConstraintManager hard;
    t.expect(hard.add(constraint(1, true, now)) && hard.active(now).size() == 1,
             "UT-005-001", "hard constraint added");
    t.expect(!hard.evaluate(plan(), now).allowed, "UT-005-003",
             "hard violation rejects plan");
    ConstraintManager soft;
    t.expect(soft.add(constraint(2, false, now)), "UT-005-002",
             "soft constraint added");
    auto softResult = soft.evaluate(plan(), now);
    t.expect(softResult.allowed && softResult.softPenalty == 1, "UT-005-004",
             "soft violation adds penalty");
    auto scoped = constraint(3, true, now);
    scoped.scope = {ConstraintScopeType::Goal, 99};
    ConstraintManager scopes;
    scopes.add(scoped);
    t.expect(scopes.evaluate(plan(), now).allowed, "UT-005-005",
             "nonmatching scope ignored");
    auto expiring = constraint(4, true, now);
    expiring.expiresAt = now + Duration{5};
    ConstraintManager expiry;
    expiry.add(expiring);
    t.expect(expiry.active(now + Duration{6}).empty(), "UT-005-006",
             "expired constraint inactive");
    t.skip("UT-005-007/008", "constraint conflict resolution API is absent");
    t.skip("UT-005-009", "replan event channel is not exposed");
    return t.finish();
}
