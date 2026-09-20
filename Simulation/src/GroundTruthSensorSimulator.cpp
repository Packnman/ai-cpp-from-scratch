#include "simulation/GroundTruthSensorSimulator.hpp"

namespace ai::simulation {

std::vector<SensorState>
GroundTruthSensorSimulator::sample(const RobotPlantState &state) const {
    // Add ideal channels without altering sensor values already supplied by
    // MJCF.
    std::vector<SensorState> sensors = state.sensors;
    sensors.push_back({"ground_truth_imu_orientation",
                       SensorKind::Orientation,
                       {state.base.orientation.w, state.base.orientation.x,
                        state.base.orientation.y, state.base.orientation.z}});
    sensors.push_back(
        {"ground_truth_imu_angular_velocity",
         SensorKind::AngularVelocity,
         {state.base.angularVelocity.x, state.base.angularVelocity.y,
          state.base.angularVelocity.z}});
    for (const auto &joint : state.joints) {
        sensors.push_back({joint.id + "_position",
                           SensorKind::JointPosition,
                           {joint.position}});
        sensors.push_back({joint.id + "_velocity",
                           SensorKind::JointVelocity,
                           {joint.velocity}});
    }
    sensors.push_back({"ground_truth_foot_contact",
                       SensorKind::Contact,
                       {state.contacts.empty() ? 0.0 : 1.0}});
    return sensors;
}

} // namespace ai::simulation
