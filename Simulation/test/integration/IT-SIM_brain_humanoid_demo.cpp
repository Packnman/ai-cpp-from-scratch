#include "../TestSupport.hpp"
#include "simulation/HumanoidDemo.hpp"

#include <string>

using namespace ai::simulation::demo;

int main() {
    simulation_test::Suite test{"IT-SIM-004 Brain Humanoid Demo"};
    HumanoidDemo demo{simulation_test::model("demo/humanoid_supported.xml")};
    demo.initialize();

    const auto raised = demo.command("右手を上げて");
    test.expect(raised.recognized && raised.planned && raised.dispatched &&
                    raised.goalType == "RaiseRightArm",
                "IT-SIM-004-001",
                "Japanese command reaches the Control dispatcher");
    demo.runSteps(2'000);
    const double raisedElbow = demo.rightElbowPosition();
    test.expect(raisedElbow > -0.05 && raisedElbow < 0.05, "IT-SIM-004-002",
                "raised pose keeps the elbow approximately straight");
    const double raisedShoulder =
        demo.jointPosition("right_shoulder_abduction");
    test.expect(raisedShoulder < -1.8, "IT-SIM-004-009",
                "deltoid group raises the right arm toward vertical");
    test.expect(demo.jointPosition("right_scapula_rotation") > 0.05,
                "IT-SIM-004-010",
                "serratus and trapezius groups rotate the scapula");

    const auto lowered = demo.command("右腕を下げて");
    test.expect(lowered.recognized && lowered.planned && lowered.dispatched &&
                    lowered.goalType == "LowerRightArm",
                "IT-SIM-004-003", "lower command selects the inverse skill");
    demo.runSteps(2'000);
    test.expect(demo.rightElbowPosition() > -0.05 &&
                    demo.rightElbowPosition() < 0.05,
                "IT-SIM-004-004", "lower pose keeps the elbow straight");
    test.expect(demo.jointPosition("right_shoulder_abduction") > raisedShoulder,
                "IT-SIM-004-011", "lower skill returns the right shoulder");

    const auto unsupported = demo.command("こんにちは");
    test.expect(!unsupported.recognized && !unsupported.planned &&
                    !unsupported.dispatched,
                "IT-SIM-004-005", "unsupported conversation is not motion");

    const auto replayTarget = demo.command("右手を上げて");
    test.expect(replayTarget.dispatched, "IT-SIM-004-006",
                "raise command is available for replay");
    demo.runSteps(2'000);
    const double beforeReplay =
        demo.jointPosition("right_shoulder_abduction");
    demo.replay();
    test.expect(demo.jointPosition("right_shoulder_abduction") > beforeReplay,
                "IT-SIM-004-007",
                "replay restores the initial humanoid pose");
    demo.runSteps(2'000);
    test.expect(demo.jointPosition("right_shoulder_abduction") < -1.8,
                "IT-SIM-004-008",
                "replay runs the retained command again");

    HumanoidDemo tendonDemo{
        simulation_test::model("upper_body_structure/upper_body.xml")};
    tendonDemo.initialize();
    const auto tendonRaise = tendonDemo.command("右手を上げて");
    test.expect(tendonRaise.dispatched, "IT-SIM-004-012",
                "raise command reaches the native tendon model");
    tendonDemo.runSteps(8'000);
    test.expect(tendonDemo.jointPosition("right_shoulder_abduction") < -2.8 &&
                    tendonDemo.jointPosition("right_scapula_rotation") > 0.15,
                "IT-SIM-004-013",
                "pull-only muscles raise the arm against gravity");
    const auto *nativeModel = tendonDemo.plant().model().model();
    const auto *nativeData = tendonDemo.plant().model().data();
    bool hasTension{};
    bool allPullOnly{true};
    for (int id = 0; id < nativeModel->nu; ++id) {
        hasTension = hasTension || nativeData->actuator_force[id] > 0.01;
        allPullOnly =
            allPullOnly && nativeData->actuator_force[id] >= -1e-9;
    }
    test.expect(hasTension && allPullOnly, "IT-SIM-004-014",
                "native muscles apply non-negative tension only");
    return test.finish();
}
