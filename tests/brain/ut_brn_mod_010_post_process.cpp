#include "TestSupport.hpp"
#include "brain/postprocess/PostProcessor.hpp"

using namespace ai::brain;

namespace {
Goal active_goal(GoalManager &goals, GoalId id) {
    Goal goal{id, "test"};
    goals.add(goal);
    return *goals.get(id);
}
Policy policy() {
    Action action{1, "act"};
    action.timeout = Duration{10};
    return {1, "policy", "test", {}, {action}, 1};
}
} // namespace

int main() {
    brain_test::Suite t{"UT-BRN-MOD-010"};
    GoalManager goals;
    MemoryManager memory(std::make_unique<SQLiteMemoryBackend>(":memory:"));
    PolicyManager policies;
    policies.add(policy());
    PostProcessor post(goals, memory, policies);
    auto goal = active_goal(goals, 1);
    Action action{1, "act"};
    action.timeout = Duration{10};
    ActionPlan plan{1, goal.id, {{action, {}}}};
    const auto now = steady_now();
    ActionResult success{1, ActionResultCode::Succeeded, "ok", now, now, {}};
    auto successResult = post.process({goal, plan, success, {}, 1, true});
    t.skip("UT-010-001", "PostProcessor has no raw-result logger dependency");
    t.expect(successResult.memoryId != 0, "UT-010-002",
             "success stored in memory");
    t.expect(goals.get(1)->status == GoalStatus::Achieved, "UT-010-004",
             "completed plan achieves goal");
    t.skip("UT-010-005", "recognition-result LTM promotion is not exposed");
    t.expect(successResult.policyUpdated, "UT-010-006",
             "policy statistics updated");
    t.expect(successResult.sample.result.actionId == 1 &&
                 successResult.sample.evaluation == 1.F,
             "UT-010-007", "learning sample generated");

    auto failedGoal = active_goal(goals, 2);
    ActionResult failed{1, ActionResultCode::Failed, "jammed", now, now, {}};
    auto failedResult = post.process(
        {failedGoal, ActionPlan{2, 2, {{action, {}}}}, failed, {}, 1, false});
    t.expect(failedResult.memoryId != 0 && failedResult.replanRequired,
             "UT-010-003/008", "failure stored and requests replan");
    t.skip("UT-010-009", "evaluation exception injection is not exposed");
    return t.finish();
}
