#include "../TestSupport.hpp"
#include "simulation/SimulationManager.hpp"

#include <memory>

using namespace ai::simulation;

namespace {

class FakePlant final : public IPlant {
    public:
        void initialize() override { initialized = true; }
        void reset() override {
            state = {};
            ++resets;
        }
        void step(double dt) override {
            state.simulationTime += dt;
            ++steps;
        }
        void applyJointTorque(const JointId &, double) override {}
        void applyLinearForce(const BodyId &, const Vec3 &,
                              const Vec3 &) override {}
        void setConstraintState(const ConstraintId &,
                                const ConstraintState &) override {}
        RobotPlantState getState() const override { return state; }

        RobotPlantState state;
        std::size_t steps{};
        std::size_t resets{};
        bool initialized{};
};

} // namespace

int main() {
    simulation_test::Suite test{"SIM-MOD-001 Simulation Manager"};
    auto plant = std::make_unique<FakePlant>();
    auto *rawPlant = plant.get();
    SimulationManager manager{std::move(plant), {0.001, 0.001, 0.01, 60.0}};
    std::size_t controlCycles{};
    std::size_t actuatorCycles{};
    std::size_t logs{};
    manager.setControlCallback(
        [&](double, const RobotPlantState &, IPlant &) { ++controlCycles; });
    manager.setActuatorCallback(
        [&](double, const RobotPlantState &, IPlant &) { ++actuatorCycles; });
    manager.setLogCallback([&](const RobotPlantState &) { ++logs; });
    manager.initialize();
    manager.runSteps(100);
    test.expect(rawPlant->initialized && rawPlant->steps == 100,
                "UT-SIM-001-001", "manager initializes and advances plant");
    test.expect(controlCycles == 10 && actuatorCycles == 100, "UT-SIM-001-002",
                "control and actuator rates are separated");
    test.expect(logs == 100 &&
                    simulation_test::near(rawPlant->state.simulationTime, 0.1),
                "UT-SIM-001-003", "logging follows deterministic physics time");
    manager.setPaused(true);
    manager.step();
    test.expect(rawPlant->steps == 100, "UT-SIM-001-004",
                "paused manager does not advance physics");
    manager.reset();
    test.expect(rawPlant->resets == 1 &&
                    simulation_test::near(rawPlant->state.simulationTime, 0.0),
                "UT-SIM-001-005", "reset restores initial simulation time");
    return test.finish();
}
