#include "../TestSupport.hpp"
#include "simulation/MuJoCoActuatorAdapter.hpp"

#include <optional>

using namespace ai::simulation;

namespace {

class RecordingPlant final : public IPlant {
    public:
        void initialize() override {}
        void reset() override {}
        void step(double) override {}
        void applyJointTorque(const JointId &id, double value) override {
            joint = id;
            torque = value;
        }
        void applyLinearForce(const BodyId &id, const Vec3 &,
                              const Vec3 &value) override {
            body = id;
            force = value;
        }
        void setConstraintState(const ConstraintId &id,
                                const ConstraintState &value) override {
            constraint = id;
            constraintState = value;
        }
        RobotPlantState getState() const override { return {}; }

        JointId joint;
        BodyId body;
        ConstraintId constraint;
        double torque{};
        Vec3 force;
        ConstraintState constraintState;
};

} // namespace

int main() {
    simulation_test::Suite test{"SIM-MOD-004 Actuator Adapter"};
    MuJoCoActuatorAdapter adapter;
    adapter.bind({"biceps",
                  "elbow",
                  BodyId{"forearm"},
                  {0.1, 0.0, 0.0},
                  {1.0, 0.0, 0.0},
                  ConstraintId{"elbow"}});
    ai::actuator::PlantOutput output;
    output.jointTorque = 0.3;
    output.linearForce = 12.0;
    output.passiveReactionTorque = -0.05;
    output.constraintActive = true;
    output.constraintPosition = 0.4;
    output.constraintStiffness = 1'000.0;
    output.constraintTorqueLimit = 50.0;
    RecordingPlant plant;
    adapter.apply("biceps", output, plant);
    test.expect(plant.joint == "elbow" &&
                    simulation_test::near(plant.torque, 0.25),
                "UT-SIM-004-001", "active and passive torque map once");
    test.expect(plant.body == "forearm" &&
                    simulation_test::near(plant.force.x, 12.0),
                "UT-SIM-004-002", "linear force maps to body vector");
    test.expect(
        plant.constraint == "elbow" && plant.constraintState.enabled &&
            simulation_test::near(plant.constraintState.maxTorque, 50.0),
        "UT-SIM-004-003", "finite lock constraint maps to plant");
    return test.finish();
}
