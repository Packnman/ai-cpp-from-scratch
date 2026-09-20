#include "../TestSupport.hpp"
#include "simulation/GroundTruthSensorSimulator.hpp"
#include "simulation/MuJoCoPlant.hpp"

using namespace ai::simulation;

int main() {
    simulation_test::Suite test{"SIM-MOD-005 State Adapter"};
    MuJoCoPlant plant{simulation_test::model("test/simple_leg.xml")};
    plant.initialize();
    const auto state = plant.getState();
    test.expect(!state.joints.empty() && !state.bodies.empty(),
                "UT-SIM-005-001",
                "qpos/qvel and body transforms are converted");
    test.expect(state.base.id == "pelvis", "UT-SIM-005-002",
                "floating pelvis is selected as base state");
    test.expect(state.sensors.size() >= 5, "UT-SIM-005-003",
                "MuJoCo sensors are converted without native types");
    GroundTruthSensorSimulator sensors;
    const auto samples = sensors.sample(state);
    test.expect(samples.size() > state.sensors.size(), "UT-SIM-005-004",
                "ground-truth IMU and joint feedback are available");
    return test.finish();
}
