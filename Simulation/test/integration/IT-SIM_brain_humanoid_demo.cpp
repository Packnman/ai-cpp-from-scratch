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
    const double raisedPosition = demo.rightElbowPosition();
    test.expect(raisedPosition < -0.2, "IT-SIM-004-002",
                "Actuator moves the supported humanoid right arm");

    const auto lowered = demo.command("右腕を下げて");
    test.expect(lowered.recognized && lowered.planned && lowered.dispatched &&
                    lowered.goalType == "LowerRightArm",
                "IT-SIM-004-003", "lower command selects the inverse skill");
    demo.runSteps(2'000);
    test.expect(demo.rightElbowPosition() > raisedPosition, "IT-SIM-004-004",
                "lower skill reverses the elbow motion");

    const auto unsupported = demo.command("こんにちは");
    test.expect(!unsupported.recognized && !unsupported.planned &&
                    !unsupported.dispatched,
                "IT-SIM-004-005", "unsupported conversation is not motion");
    return test.finish();
}
