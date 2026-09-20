#include "TestSupport.hpp"
#include "brain/BrainSystem.hpp"

using namespace ai::brain;

int main() {
    brain_test::Suite t{"IT-BRAIN"};
    const auto now = steady_now();
    BrainSystem brain;
    t.expect(bool(brain.push({InputSource::HumanInterface, BrainInputType::Text,
                              1, now, std::string("goal:AcquireObject"), 1})),
             "IT-001", "goal input accepted");
    auto cycle = brain.runOnce(now);
    t.expect(cycle.planId && cycle.actionResults == 1 &&
                 !brain.goals().activeGoal(),
             "IT-001/030/050", "input-to-memory success path completes");

    BrainSystem constrained;
    Constraint deny{
        1,
        "safety",
        true,
        true,
        {},
        {"deny_action_type", {{"type", std::string("acknowledge")}}},
        ConstraintSource::Safety,
        {},
        now};
    constrained.constraints().add(deny);
    constrained.push({InputSource::HumanInterface, BrainInputType::Text, 1, now,
                      std::string("goal:MoveObject"), 1});
    cycle = constrained.runOnce(now);
    t.expect(!cycle.errors.empty() && cycle.actionResults == 0, "IT-010",
             "critical constraint prevents dispatch");

    BrainSystem fallback{"/path/that/does/not/exist/brain.sqlite3"};
    fallback.push({InputSource::HumanInterface, BrainInputType::Text, 1, now,
                   std::string("goal:AnswerQuestion"), 1});
    cycle = fallback.runOnce(now);
    t.expect(cycle.actionResults == 1 && !fallback.memory().persistent(),
             "IT-052", "STM-only mode continues end-to-end");

    BrainSystem stopped;
    stopped.push({InputSource::Safety, BrainInputType::StopRequest, 1, now,
                  std::string("stop"), 1});
    stopped.push({InputSource::HumanInterface, BrainInputType::Text, 1, now,
                  std::string("goal:FindObject"), 2});
    cycle = stopped.runOnce(now);
    t.expect(stopped.execution().status() ==
                     ExecutionStatus::EmergencyStopped &&
                 cycle.actionResults == 0,
             "IT-080/090", "safety stop prevents new dispatch");

    t.skip("IT-002/003",
           "visibility-aware AcquireObject policy is not implemented");
    t.skip("IT-011/012",
           "soft constraint and live replan are unit-tested only");
    t.skip("IT-020/031/032/033", "covered by module-level deterministic tests");
    t.skip("IT-040/041/042", "External AI is not wired into BrainSystem");
    t.skip("IT-051/060/061", "memory/stale-triggered replanning is not wired");
    t.skip("IT-070/071/072", "failure injection is not exposed by BrainSystem");
    t.skip("IT-081/091",
           "running EStop and queue stress require injectable doubles");
    t.skip("IT-100", "BrainSystem does not expose its trace sink");

    BrainSystem same;
    same.push({InputSource::HumanInterface, BrainInputType::Text, 1, now,
               std::string("goal:AcquireObject"), 1});
    auto sameCycle = same.runOnce(now);
    t.expect(sameCycle.actionResults == 1 && sameCycle.errors.empty(), "IT-110",
             "same input follows same successful path");
    return t.finish();
}
