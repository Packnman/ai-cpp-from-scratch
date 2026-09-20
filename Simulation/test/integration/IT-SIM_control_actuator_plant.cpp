#include "../TestSupport.hpp"
#include "actuator/driver/sim/SimActuatorDriver.hpp"
#include "simulation/MuJoCoActuatorAdapter.hpp"
#include "simulation/MuJoCoPlant.hpp"
#include "simulation/SimulationManager.hpp"

#include <algorithm>
#include <memory>

using namespace ai::actuator;
using namespace ai::actuator::sim;
using namespace ai::simulation;

namespace {

JointState elbow(const RobotPlantState &state) {
    const auto found = std::find_if(
        state.joints.begin(), state.joints.end(),
        [](const JointState &joint) { return joint.id == "elbow"; });
    return found == state.joints.end() ? JointState{} : *found;
}

} // namespace

int main() {
    simulation_test::Suite test{"IT-SIM-003 Control Actuator Plant"};
    ActuatorDescriptor descriptor;
    descriptor.actuatorId = "right_elbow_biceps";
    descriptor.componentId = std::string{muscleComponentId};
    descriptor.type = ActuatorType::Muscle;
    descriptor.positionLimit = {-1.5, 1.5};
    descriptor.velocityLimit = {-10.0, 10.0};
    descriptor.torqueLimit = {-1.0, 1.0};
    descriptor.currentLimit = {-0.7, 0.7};
    descriptor.temperatureLimit = {-30.0, 125.0};
    descriptor.supportedModes = {ControlMode::Torque, ControlMode::Stop,
                                 ControlMode::Disable};
    SimActuatorDriver driver{defaultElbowMuscleConfig()};
    driver.configure(descriptor);
    driver.enable();
    MuJoCoActuatorAdapter adapter;
    adapter.bind({descriptor.actuatorId, "elbow", {}, {}, {}, {}});

    auto plant = std::make_unique<MuJoCoPlant>(
        simulation_test::model("test/single_joint.xml"));
    SimulationManager manager{std::move(plant), {0.001, 0.001, 0.01, 60.0}};
    constexpr double target = 0.5;
    manager.setControlCallback([&](double, const RobotPlantState &state,
                                   IPlant &) {
        const auto joint = elbow(state);
        const double torque = std::clamp(
            4.0 * (target - joint.position) - 0.5 * joint.velocity, -0.8, 0.8);
        driver.command({1,
                        descriptor.actuatorId,
                        ControlMode::Torque,
                        torque,
                        {},
                        TimePoint{},
                        Duration{100}});
    });
    manager.setActuatorCallback([&](double dt, const RobotPlantState &state,
                                    IPlant &plantRef) {
        const auto joint = elbow(state);
        driver.advance(dt, {joint.position, joint.velocity, 0.0, true});
        adapter.apply(descriptor.actuatorId, driver.plantOutput(), plantRef);
    });
    manager.initialize();
    manager.runSteps(2'000);
    const auto finalJoint = elbow(manager.plant().getState());
    test.expect(finalJoint.position > 0.05, "IT-SIM-003-001",
                "control-actuator-plant loop moves toward target");
    test.expect(manager.plant().getState().simulationTime >= 1.999,
                "IT-SIM-003-002",
                "headless multirate loop reaches requested time");
    return test.finish();
}
