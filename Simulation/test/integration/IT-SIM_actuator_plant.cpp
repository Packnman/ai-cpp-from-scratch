#include "../TestSupport.hpp"
#include "actuator/driver/sim/SimActuatorDriver.hpp"
#include "simulation/MuJoCoActuatorAdapter.hpp"
#include "simulation/MuJoCoPlant.hpp"

#include <algorithm>

using namespace ai::actuator;
using namespace ai::actuator::sim;
using namespace ai::simulation;

namespace {

const JointState &elbow(const RobotPlantState &state) {
    const auto found = std::find_if(
        state.joints.begin(), state.joints.end(),
        [](const JointState &joint) { return joint.id == "elbow"; });
    if (found == state.joints.end())
        throw std::runtime_error("elbow state missing");
    return *found;
}

} // namespace

int main() {
    simulation_test::Suite test{"IT-SIM-002 Actuator Plant"};
    ActuatorDescriptor descriptor;
    descriptor.actuatorId = "right_elbow_biceps";
    descriptor.componentId = std::string{muscleComponentId};
    descriptor.type = ActuatorType::Muscle;
    descriptor.positionLimit = {-1.5, 1.5};
    descriptor.velocityLimit = {-10.0, 10.0};
    descriptor.torqueLimit = {-2.0, 2.0};
    descriptor.currentLimit = {-0.7, 0.7};
    descriptor.temperatureLimit = {-30.0, 125.0};
    descriptor.supportedModes = {ControlMode::Torque, ControlMode::Stop,
                                 ControlMode::Disable};
    SimActuatorDriver driver{defaultElbowMuscleConfig()};
    driver.configure(descriptor);
    driver.enable();
    driver.command({1,
                    descriptor.actuatorId,
                    ControlMode::Torque,
                    0.3,
                    {},
                    TimePoint{},
                    Duration{1'000}});

    MuJoCoPlant plant{simulation_test::model("test/single_joint.xml")};
    plant.initialize();
    MuJoCoActuatorAdapter adapter;
    adapter.bind({descriptor.actuatorId, "elbow", {}, {}, {}, {}});
    double maxCurrent{};
    double maxForce{};
    double maxJointVelocity{};
    for (int i = 0; i < 1'000; ++i) {
        const auto joint = elbow(plant.getState());
        maxJointVelocity = std::max(maxJointVelocity, std::abs(joint.velocity));
        driver.advance(0.001, {joint.position, joint.velocity, 0.0, true});
        adapter.apply(descriptor.actuatorId, driver.plantOutput(), plant);
        plant.step(0.001);
        const auto actuator = driver.readState();
        maxCurrent = std::max(maxCurrent, std::abs(actuator.current));
        maxForce = std::max(maxForce, std::abs(actuator.tendonTension));
    }
    const auto state = elbow(plant.getState());
    test.expect(maxCurrent > 0.0 && maxForce > 0.0, "IT-SIM-002-001",
                "motor current produces tendon force");
    test.expect(std::abs(state.position) > 1e-4 && maxJointVelocity > 1e-4,
                "IT-SIM-002-002", "actuator effort moved MuJoCo elbow");
    test.expect(simulation_test::near(driver.readState().position,
                                      state.position, 0.02),
                "IT-SIM-002-003", "MuJoCo feedback returns to actuator state");
    return test.finish();
}
