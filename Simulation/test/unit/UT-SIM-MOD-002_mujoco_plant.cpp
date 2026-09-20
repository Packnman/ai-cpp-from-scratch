#include "../TestSupport.hpp"
#include "simulation/MuJoCoPlant.hpp"

using namespace ai::simulation;

int main() {
    simulation_test::Suite test{"SIM-MOD-002 MuJoCo Plant"};
    MuJoCoPlant plant{simulation_test::model("test/single_joint.xml")};
    plant.initialize();
    test.expect(plant.initialized(), "UT-SIM-002-001",
                "MJCF model loads through RAII wrapper");
    const auto &joint = plant.model().jointBinding("elbow");
    const auto &body = plant.model().bodyBinding("forearm");
    test.expect(joint.jointId >= 0 && joint.qposAddress >= 0 &&
                    joint.dofAddress >= 0,
                "UT-SIM-002-002", "joint mapping records all addresses");
    test.expect(body.bodyId >= 0, "UT-SIM-002-003",
                "body mapping resolves logical name");
    plant.applyJointTorque("elbow", 0.2);
    plant.step(0.001);
    const auto state = plant.getState();
    test.expect(simulation_test::near(state.simulationTime, 0.001, 1e-12),
                "UT-SIM-002-004", "step advances explicit simulation dt");
    test.expect(!state.joints.empty() &&
                    std::abs(state.joints.front().velocity) > 0.0,
                "UT-SIM-002-005", "applied torque changes joint velocity");
    plant.reset();
    test.expect(simulation_test::near(plant.getState().simulationTime, 0.0),
                "UT-SIM-002-006", "reset restores MuJoCo data");
    return test.finish();
}
