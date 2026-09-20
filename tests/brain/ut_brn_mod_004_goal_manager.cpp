#include "TestSupport.hpp"
#include "brain/goal/GoalManager.hpp"

using namespace ai::brain;

namespace {
Goal goal(GoalId id, int priority, GoalSource source = GoalSource::Human) {
    Goal value;
    value.id = id;
    value.type = "test";
    value.priority = priority;
    value.source = source;
    return value;
}
} // namespace

int main() {
    brain_test::Suite t{"UT-BRN-MOD-004"};
    GoalManager goals;
    const auto first = goals.add(goal(1, 1));
    t.expect(first == 1 && goals.activeGoal()->id == 1, "UT-004-001",
             "first goal registered and selected");
    goals.add(goal(2, 10));
    t.expect(goals.activeGoal()->id == 2, "UT-004-002/004",
             "higher priority preempts current goal");
    t.expect(goals.get(1)->status == GoalStatus::Suspended, "UT-004-004",
             "preempted goal suspended");
    goals.add(goal(3, -100, GoalSource::Safety));
    t.expect(goals.activeGoal()->id == 3, "UT-004-003",
             "safety goal takes precedence");
    t.expect(goals.complete(3, true) &&
                 goals.get(3)->status == GoalStatus::Achieved,
             "UT-004-005", "goal completed");
    t.expect(goals.complete(2, false) &&
                 goals.get(2)->status == GoalStatus::Failed,
             "UT-004-006", "goal failed");
    t.expect(goals.cancel(1) && goals.get(1)->status == GoalStatus::Cancelled,
             "UT-004-007", "active goal cancelled");
    Goal parent = goal(4, 1);
    goals.add(parent);
    Goal child = goal(5, 2, GoalSource::SubGoal);
    child.parentGoal = 4;
    t.expect(goals.add(child) == 5 && goals.get(5)->parentGoal == 4,
             "UT-004-008", "parent relationship retained");
    t.expect(goals.add(goal(5, 3)) == 0, "UT-004-009", "duplicate ID rejected");
    return t.finish();
}
