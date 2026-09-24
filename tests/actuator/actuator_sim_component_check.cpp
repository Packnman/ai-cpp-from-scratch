#include "TestSupport.hpp"
#include "actuator/component/sim/SimComponents.hpp"

#include <cmath>
#include <memory>

using namespace ai::actuator;
using namespace ai::actuator::sim;

namespace {

ScrewTransmissionParameters transmission(double minimum = -1.0,
                                         double maximum = 1.0) {
    return {std::string{transmissionComponentId},
            0.01,
            1.0,
            2.0,
            0.0,
            minimum,
            maximum};
}

MuscleActuatorParameters muscle(std::size_t motorCount, double minimum = -1.0,
                                double maximum = 1.0) {
    MuscleActuatorParameters parameters;
    parameters.motors.assign(motorCount,
                             motorParameters(MotorPreset::Ect35_80));
    parameters.transmission = transmission(minimum, maximum);
    parameters.maxTendonForce = 1'000.0;
    return parameters;
}

void run(SimBLDCMotor &motor, double voltage, int steps = 500,
         double loadTorque = 0.0) {
    motor.setVoltage(voltage);
    for (int i = 0; i < steps; ++i)
        motor.step(1e-5, loadTorque);
}

void run(SimMuscleActuator &actuator, double voltage, int steps = 500) {
    actuator.commandVoltage(voltage);
    for (int i = 0; i < steps; ++i)
        actuator.step(1e-5, 0.0, 0.0);
}

} // namespace

int main() {
    actuator_test::Suite test{"Actuator simulation components"};

    SimBLDCMotor zero{motorParameters(MotorPreset::Ect35_80)};
    zero.step(1e-5, 0.0);
    test.expect(actuator_test::near(zero.state().current, 0.0) &&
                    actuator_test::near(zero.state().angularVelocity, 0.0),
                "MOT-001", "zero voltage preserves zero state");

    SimBLDCMotor positive{motorParameters(MotorPreset::Ect35_80)};
    run(positive, 6.0);
    test.expect(positive.state().angularVelocity > 0.0 &&
                    positive.state().position > 0.0,
                "MOT-002", "positive voltage accelerates motor");

    const double speedBeforeBackEmf = positive.state().angularVelocity;
    positive.setVoltage(0.0);
    for (int i = 0; i < 100; ++i)
        positive.step(1e-5, 0.0);
    test.expect(positive.state().current < 0.0 &&
                    positive.state().angularVelocity < speedBeforeBackEmf,
                "MOT-003", "back EMF produces braking current");

    SimBLDCMotor limited{motorParameters(MotorPreset::Ect35_80)};
    limited.setVoltage(24.0);
    limited.step(1e-3, 0.0);
    test.expect(limited.state().currentLimited &&
                    std::abs(limited.state().current) <= 0.7,
                "MOT-004", "continuous current limit is enforced");

    auto lowSpeedParameters = motorParameters(MotorPreset::Ect35_80);
    lowSpeedParameters.maxSpeed = 2.0;
    SimBLDCMotor speedLimited{lowSpeedParameters};
    run(speedLimited, 24.0, 100);
    test.expect(speedLimited.state().speedLimited &&
                    std::abs(speedLimited.state().angularVelocity) <= 2.0,
                "MOT-005", "speed limit is enforced");

    SimBLDCMotor unloaded{motorParameters(MotorPreset::Ect35_80)};
    SimBLDCMotor loaded{motorParameters(MotorPreset::Ect35_80)};
    run(unloaded, 6.0, 200);
    run(loaded, 6.0, 200, 0.01);
    test.expect(loaded.state().angularVelocity <
                    unloaded.state().angularVelocity,
                "MOT-006", "load torque reduces acceleration");

    SimBLDCMotor deterministicA{motorParameters(MotorPreset::Ect48_35)};
    SimBLDCMotor deterministicB{motorParameters(MotorPreset::Ect48_35)};
    run(deterministicA, 5.0, 100);
    run(deterministicB, 5.0, 100);
    test.expect(actuator_test::near(deterministicA.state().position,
                                    deterministicB.state().position) &&
                    actuator_test::near(deterministicA.state().current,
                                        deterministicB.state().current),
                "MOT-007", "explicit dt produces deterministic state");
    deterministicA.injectFault({.overCurrent = true});
    deterministicA.step(1e-5, 0.0);
    test.expect(deterministicA.state().fault == ActuatorFaultCode::OverCurrent,
                "MOT-008", "injected motor fault is observable");
    deterministicA.injectFault({.stalledMotor = true});
    deterministicA.step(1e-5, 0.0);
    test.expect(deterministicA.state().fault == ActuatorFaultCode::StalledMotor,
                "MOT-009", "stall injection is observable");

    SimTransmission screw{transmission(-0.01, 0.01)};
    const auto converted = screw.evaluate(4.0 * std::numbers::pi, 2.0, 0.5);
    test.expect(actuator_test::near(converted.displacement, 0.01), "TRN-001",
                "rotation maps to screw displacement");
    test.expect(actuator_test::near(converted.force,
                                    2.0 * std::numbers::pi * 0.5 * 2.0 / 0.01),
                "TRN-002", "motor torque maps to linear force");
    const auto ratioOne = SimTransmission{
        {std::string{transmissionComponentId}, 0.01, 1.0, 1.0, 0.0, -1.0,
         1.0}}.evaluate(1.0, 1.0, 1.0);
    test.expect(converted.velocity < ratioOne.velocity, "TRN-003",
                "gear ratio reduces linear velocity");
    const auto stop = screw.evaluate(100.0, 10.0, 0.1);
    test.expect(stop.upperLimit &&
                    actuator_test::near(stop.displacement, 0.01) &&
                    actuator_test::near(stop.velocity, 0.0),
                "TRN-004", "mechanical stop clamps travel and velocity");

    SimMuscleActuator single{muscle(1),
                             std::make_unique<ConstantMomentArmModel>(0.03)};
    SimMuscleActuator multiple{muscle(2),
                               std::make_unique<ConstantMomentArmModel>(0.03)};
    run(single, 6.0);
    run(multiple, 6.0);
    test.expect(single.state().motors.size() == 1 &&
                    single.state().tendonForce > 0.0,
                "MUS-001", "single motor muscle generates force");
    test.expect(multiple.state().motors.size() == 2 &&
                    multiple.state().tendonForce > single.state().tendonForce,
                "MUS-002", "multi motor ideal load sharing adds torque");
    test.expect(multiple.state().estimatedJointTorque > 0.0, "MUS-003",
                "tendon force maps through moment arm");

    SimMuscleActuator limitedMuscle{
        muscle(1, -1e-6, 1e-6), std::make_unique<ConstantMomentArmModel>(0.03)};
    run(limitedMuscle, 24.0, 2'000);
    test.expect(
        limitedMuscle.state().upperLimit &&
            actuator_test::near(
                limitedMuscle.state().motors.front().angularVelocity, 0.0) &&
            limitedMuscle.state().fault == ActuatorFaultCode::OverTravel,
        "MUS-004", "muscle stroke limit reports over-travel");
    multiple.injectFault({.tendonBreak = true});
    multiple.step(1e-5, 0.0, 0.0);
    test.expect(actuator_test::near(multiple.state().tendonForce, 0.0) &&
                    multiple.state().fault == ActuatorFaultCode::TendonBreak,
                "MUS-005", "tendon break removes transmitted force");

    WaistLinearCylinderParameters cylinderParameters{
        std::string{linearCylinderComponentId},
        motorParameters(MotorPreset::Ect60_21), transmission(-0.002, 0.002),
        2'000.0};
    SimWaistLinearCylinder cylinder{cylinderParameters};
    cylinder.setVoltage(8.0);
    for (int i = 0; i < 500; ++i)
        cylinder.step(1e-5, 0.0);
    test.expect(cylinder.state().length > 0.0 &&
                    cylinder.state().velocity > 0.0,
                "LIN-001", "positive voltage extends cylinder");
    test.expect(cylinder.state().estimatedForce > 0.0, "LIN-002",
                "cylinder exposes estimated force");
    SimWaistLinearCylinder retract{cylinderParameters};
    retract.setVoltage(-8.0);
    for (int i = 0; i < 500; ++i)
        retract.step(1e-5, 0.0);
    test.expect(retract.state().length < 0.0 && retract.state().velocity < 0.0,
                "LIN-003", "negative voltage retracts cylinder");
    auto shortCylinderParameters = cylinderParameters;
    shortCylinderParameters.transmission.minPosition = -1e-6;
    shortCylinderParameters.transmission.maxPosition = 1e-6;
    SimWaistLinearCylinder shortCylinder{shortCylinderParameters};
    shortCylinder.setVoltage(24.0);
    for (int i = 0; i < 2'000; ++i)
        shortCylinder.step(1e-5, 0.0);
    test.expect(shortCylinder.state().upperLimit, "LIN-004",
                "cylinder enforces stroke limit");

    KneeLockParameters lockParameters{
        std::string{kneeLockComponentId}, 0.2, 0.05, 0.02, 80.0, 5'000.0};
    SimKneeLock lock{lockParameters};
    lock.lock(0.4, 0.0);
    lock.step(0.06, 0.4, 0.0);
    test.expect(lock.state() == KneeLockState::Locked &&
                    lock.constraint().active &&
                    actuator_test::near(lock.constraint().position, 0.4),
                "LCK-001", "lock exposes finite plant constraint");
    lock.release();
    lock.step(0.03, 0.4, 0.0);
    test.expect(lock.state() == KneeLockState::Released &&
                    !lock.constraint().active,
                "LCK-002", "release removes plant constraint");
    lock.lock(0.0, 1.0);
    test.expect(lock.state() == KneeLockState::Fault, "LCK-003",
                "high velocity engage fails safe");
    lock.resetFault();
    lock.injectFault({.lockFailure = true});
    lock.lock(0.0, 0.0);
    test.expect(lock.state() == KneeLockState::Fault, "LCK-004",
                "lock fault injection is observable");

    SimSpringDamper spring{{std::string{springDamperComponentId}, 100.0, 5.0,
                            0.1, -0.5, 0.5, 1'000.0}};
    test.expect(
        actuator_test::near(spring.evaluate(0.1, 0.0).reactionTorque, 0.0),
        "SPR-001", "neutral position has zero reaction");
    test.expect(
        actuator_test::near(spring.evaluate(0.2, 0.0).reactionTorque, -10.0),
        "SPR-002", "spring produces restoring torque");
    test.expect(
        actuator_test::near(spring.evaluate(0.1, 2.0).reactionTorque, -10.0),
        "SPR-003", "damper opposes velocity");
    const auto beyondLimit = spring.evaluate(0.6, 0.0);
    test.expect(beyondLimit.upperLimit && beyondLimit.reactionTorque < -50.0,
                "SPR-004", "mechanical limit adds finite stop torque");

    return test.finish();
}
