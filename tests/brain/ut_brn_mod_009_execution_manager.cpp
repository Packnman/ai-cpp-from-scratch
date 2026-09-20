#include "TestSupport.hpp"
#include "brain/execution/ExecutionManager.hpp"

using namespace ai::brain;

namespace {
Action action(ActionId id, ResourceAccess access = ResourceAccess::Exclusive) {
    Action value{id, "action"};
    value.timeout = Duration{10};
    value.resources = {{1, access}};
    return value;
}
ActionResult result(ActionId id, ActionResultCode code, TimePoint now) {
    return {id, code, "result", now, now, {}};
}
} // namespace

int main() {
    brain_test::Suite t{"UT-BRN-MOD-009"};
    const auto now = steady_now();
    auto sequentialControl = std::make_shared<MockControlDispatcher>(false);
    ExecutionManager sequential(sequentialControl);
    ActionPlan sequence{1, 1, {{action(1), {}}, {action(2), {1}}}};
    sequential.submit(sequence);
    sequential.tick(now);
    sequential.update(result(1, ActionResultCode::Succeeded, now));
    sequential.tick(now);
    t.expect(sequentialControl->commands().size() == 2 &&
                 sequentialControl->commands()[0].id == 1 &&
                 sequentialControl->commands()[1].id == 2,
             "UT-009-001", "dependent actions dispatched in order");

    auto parallelControl = std::make_shared<MockControlDispatcher>(false);
    ExecutionManager parallel(parallelControl);
    auto p1 = action(3);
    p1.resources = {{1, ResourceAccess::Exclusive}};
    auto p2 = action(4);
    p2.resources = {{2, ResourceAccess::Exclusive}};
    parallel.submit({2, 1, {{p1, {}}, {p2, {}}}});
    parallel.tick(now);
    t.expect(parallelControl->commands().size() == 2, "UT-009-002",
             "nonconflicting actions dispatched together");

    auto exclusiveControl = std::make_shared<MockControlDispatcher>(false);
    ExecutionManager exclusive(exclusiveControl);
    exclusive.submit({3, 1, {{action(5), {}}, {action(6), {}}}});
    exclusive.tick(now);
    t.expect(exclusiveControl->commands().size() == 1, "UT-009-003",
             "exclusive resource collision prevented");

    auto sharedControl = std::make_shared<MockControlDispatcher>(false);
    ExecutionManager shared(sharedControl);
    shared.submit({4,
                   1,
                   {{action(7, ResourceAccess::Shared), {}},
                    {action(8, ResourceAccess::Shared), {}}}});
    shared.tick(now);
    t.expect(sharedControl->commands().size() == 2, "UT-009-004",
             "shared resource permits concurrent dispatch");

    auto immediate = std::make_shared<MockControlDispatcher>();
    ExecutionManager success(immediate);
    success.submit({5, 1, {{action(9), {}}}});
    success.tick(now);
    t.expect(success.status() == ExecutionStatus::Succeeded, "UT-009-005",
             "success reaches terminal state");

    ExecutionManager failure(std::make_shared<MockControlDispatcher>(
        true, ActionResultCode::Failed));
    failure.submit({6, 1, {{action(10), {}}}});
    failure.tick(now);
    t.expect(failure.status() == ExecutionStatus::Failed, "UT-009-006",
             "failure reaches failed state");

    auto timeoutControl = std::make_shared<MockControlDispatcher>(false);
    ExecutionManager timeout(timeoutControl);
    timeout.submit({7, 1, {{action(11), {}}}});
    timeout.tick(now);
    auto timeoutResults = timeout.tick(now + Duration{11});
    t.expect(timeoutResults.size() == 1 &&
                 timeoutResults[0].result == ActionResultCode::Timeout,
             "UT-009-007", "timeout cancels action");

    auto cancelControl = std::make_shared<MockControlDispatcher>(false);
    ExecutionManager cancel(cancelControl);
    cancel.submit({8, 1, {{action(12), {}}}});
    cancel.tick(now);
    cancel.cancel(8);
    t.expect(!cancel.submit({9, 1, {{action(13), {}}}}) &&
                 cancel.update(result(12, ActionResultCode::Cancelled, now)),
             "UT-009-008", "resource retained until stop acknowledgement");
    cancel.emergencyStop();
    t.expect(cancel.status() == ExecutionStatus::EmergencyStopped &&
                 !cancel.submit({10, 1, {{action(14), {}}}}),
             "UT-009-009", "emergency stop blocks dispatch");
    t.expect(!cancel.update(result(12, ActionResultCode::Cancelled, now)),
             "UT-009-010", "duplicate result ignored");
    return t.finish();
}
