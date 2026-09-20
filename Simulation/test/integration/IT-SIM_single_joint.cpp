#include "../TestSupport.hpp"
#include "simulation/MuJoCoPlant.hpp"

#include <cmath>

using namespace ai::simulation;

namespace {

double jointPosition(const RobotPlantState &state) {
    for (const auto &joint : state.joints)
        if (joint.id == "elbow")
            return joint.position;
    return 0.0;
}

double jointVelocity(const RobotPlantState &state) {
    for (const auto &joint : state.joints)
        if (joint.id == "elbow")
            return joint.velocity;
    return 0.0;
}

} // namespace

int main() {
    simulation_test::Suite test{"IT-SIM-001 Single Joint"};
    constexpr double dt = 0.001;
    constexpr int steps = 100;
    constexpr double torque = 0.2;
    MuJoCoPlant plant{simulation_test::model("test/single_joint.xml")};
    plant.initialize();
    auto *model = plant.model().model();
    auto *data = plant.model().data();
    const auto binding = plant.model().jointBinding("elbow");
    model->opt.gravity[0] = 0.0;
    model->opt.gravity[1] = 0.0;
    model->opt.gravity[2] = 0.0;
    model->dof_damping[binding.dofAddress] = 0.0;
    mj_forward(model, data);
    const double inertia = data->M[model->dof_Madr[binding.dofAddress]];
    for (int i = 0; i < steps; ++i) {
        plant.applyJointTorque("elbow", torque);
        plant.step(dt);
    }
    const auto driven = plant.getState();
    const double acceleration = torque / inertia;
    const double expectedVelocity = acceleration * steps * dt;
    const double expectedPosition =
        acceleration * dt * dt * steps * (steps + 1) * 0.5;
    test.expect(jointPosition(driven) > 0.0 && jointVelocity(driven) > 0.0,
                "IT-SIM-001-001", "positive torque has positive sign");
    test.expect(
        simulation_test::near(jointVelocity(driven), expectedVelocity, 3e-3) &&
            simulation_test::near(jointPosition(driven), expectedPosition,
                                  2e-4),
        "IT-SIM-001-002",
        "constant torque agrees with rigid-body analytical result");

    plant.reset();
    model->opt.gravity[2] = -9.81;
    mj_forward(model, data);
    for (int i = 0; i < steps; ++i)
        plant.step(dt);
    test.expect(std::abs(jointPosition(plant.getState())) > 1e-3,
                "IT-SIM-001-003", "gravity moves horizontal forearm");

    MuJoCoPlant first{simulation_test::model("test/single_joint.xml")};
    MuJoCoPlant second{simulation_test::model("test/single_joint.xml")};
    first.initialize();
    second.initialize();
    for (int i = 0; i < steps; ++i) {
        first.applyJointTorque("elbow", 0.1);
        second.applyJointTorque("elbow", 0.1);
        first.step(dt);
        second.step(dt);
    }
    test.expect(simulation_test::near(jointPosition(first.getState()),
                                      jointPosition(second.getState()), 1e-12),
                "IT-SIM-001-004", "headless repeated runs are deterministic");
    return test.finish();
}
