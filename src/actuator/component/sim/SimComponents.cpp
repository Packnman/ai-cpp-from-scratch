#include "actuator/component/sim/SimComponents.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace ai::actuator::sim {
namespace {

constexpr double rpmToRadiansPerSecond = 2.0 * std::numbers::pi / 60.0;

// Rejects non-finite and non-positive physical parameters.
void requirePositive(double value, const char *name) {
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(std::string{name} + " must be positive");
}

// Validates the motor model before any numerical integration.
void validateMotor(const MotorParameters &parameters) {
    requirePositive(parameters.resistance, "motor resistance");
    requirePositive(parameters.inductance, "motor inductance");
    requirePositive(parameters.torqueConstant, "motor torque constant");
    requirePositive(parameters.backEmfConstant, "motor back EMF constant");
    requirePositive(parameters.rotorInertia, "motor rotor inertia");
    if (parameters.viscousFriction < 0.0)
        throw std::invalid_argument(
            "motor viscous friction must be non-negative");
    requirePositive(parameters.nominalVoltage, "motor nominal voltage");
    requirePositive(parameters.continuousCurrent, "motor current limit");
    requirePositive(parameters.maxSpeed, "motor speed limit");
    requirePositive(parameters.thermalResistance, "motor thermal resistance");
    requirePositive(parameters.thermalTimeConstant,
                    "motor thermal time constant");
}

// Validates screw geometry, efficiency and stroke limits.
void validateTransmission(const ScrewTransmissionParameters &parameters) {
    requirePositive(parameters.lead, "screw lead");
    requirePositive(parameters.efficiency, "transmission efficiency");
    if (parameters.efficiency > 1.0)
        throw std::invalid_argument(
            "transmission efficiency must not exceed one");
    requirePositive(parameters.gearRatio, "transmission gear ratio");
    if (parameters.backlash < 0.0)
        throw std::invalid_argument(
            "transmission backlash must be non-negative");
    if (parameters.minPosition > parameters.maxPosition)
        throw std::invalid_argument("invalid transmission position limits");
}

// Applies a symmetric deadband around zero displacement.
double applyBacklash(double displacement, double backlash) {
    const double half = backlash * 0.5;
    if (displacement > half)
        return displacement - half;
    if (displacement < -half)
        return displacement + half;
    return 0.0;
}

} // namespace

MotorParameters motorParameters(MotorPreset preset) {
    // Physical values are SI conversions of CMP-MOT-001 detailed motor data.
    // Viscous friction is not specified by the component data and therefore
    // remains an explicit zero-valued default model parameter.
    switch (preset) {
    case MotorPreset::Ect35_80:
        return {std::string{motorComponentId},
                "22ECT35-80",
                9.23,
                0.00075,
                0.02731,
                2.86 / (1000.0 * rpmToRadiansPerSecond),
                3.6e-7,
                0.0,
                24.0,
                0.7,
                20'000.0 * rpmToRadiansPerSecond,
                125.0,
                13.0,
                829.0,
                25.0};
    case MotorPreset::Ect48_35:
        return {std::string{motorComponentId},
                "22ECT48-35",
                2.43,
                0.00024,
                0.02808,
                2.94 / (1000.0 * rpmToRadiansPerSecond),
                6.3e-7,
                0.0,
                24.0,
                1.5,
                20'000.0 * rpmToRadiansPerSecond,
                125.0,
                12.0,
                962.0,
                25.0};
    case MotorPreset::Ect60_21:
        return {std::string{motorComponentId},
                "22ECT60-21",
                1.11,
                0.000123,
                0.02597,
                2.72 / (1000.0 * rpmToRadiansPerSecond),
                8.71e-7,
                0.0,
                24.0,
                2.6,
                20'000.0 * rpmToRadiansPerSecond,
                125.0,
                8.8,
                980.0,
                25.0};
    }
    throw std::invalid_argument("unknown motor preset");
}

SimBLDCMotor::SimBLDCMotor(MotorParameters parameters)
    : _parameters(std::move(parameters)) {
    validateMotor(_parameters);
    _state.windingTemperature = _parameters.ambientTemperature;
}

void SimBLDCMotor::setVoltage(double voltage) noexcept {
    _requestedVoltage = std::isfinite(voltage) ? voltage : 0.0;
}

void SimBLDCMotor::step(double dtSeconds, double loadTorque) {
    requirePositive(dtSeconds, "motor dt");
    if (!std::isfinite(loadTorque))
        throw std::invalid_argument("motor load torque must be finite");

    const double voltage =
        std::clamp(_requestedVoltage, -_parameters.nominalVoltage,
                   _parameters.nominalVoltage);
    // Solve the RL winding response exactly over this step while voltage and
    // back-EMF are held constant. This stays stable when the simulation step
    // is longer than the electrical time constant.
    const double steadyCurrent =
        (voltage - _parameters.backEmfConstant * _state.angularVelocity) /
        _parameters.resistance;
    const double decay =
        std::exp(-_parameters.resistance * dtSeconds / _parameters.inductance);
    const double rawCurrent =
        steadyCurrent + (_state.current - steadyCurrent) * decay;
    _state.current = std::clamp(rawCurrent, -_parameters.continuousCurrent,
                                _parameters.continuousCurrent);
    _state.currentLimited = rawCurrent != _state.current;
    _state.torque = _parameters.torqueConstant * _state.current;

    // Semi-implicit Euler updates speed before rotor position.
    const double acceleration =
        (_state.torque - loadTorque -
         _parameters.viscousFriction * _state.angularVelocity) /
        _parameters.rotorInertia;
    const double rawSpeed = _state.angularVelocity + acceleration * dtSeconds;
    _state.angularVelocity =
        std::clamp(rawSpeed, -_parameters.maxSpeed, _parameters.maxSpeed);
    _state.speedLimited = rawSpeed != _state.angularVelocity;
    _state.position += _state.angularVelocity * dtSeconds;
    _state.appliedVoltage = voltage;

    // A first-order thermal network converts copper loss into winding heat.
    const double thermalCapacitance =
        _parameters.thermalTimeConstant / _parameters.thermalResistance;
    const double copperLoss =
        _state.current * _state.current * _parameters.resistance;
    const double cooling =
        (_state.windingTemperature - _parameters.ambientTemperature) /
        _parameters.thermalResistance;
    _state.windingTemperature +=
        (copperLoss - cooling) / thermalCapacitance * dtSeconds;

    _state.fault = ActuatorFaultCode::None;
    if (_faultInjection.overCurrent)
        _state.fault = ActuatorFaultCode::OverCurrent;
    else if (_faultInjection.overSpeed)
        _state.fault = ActuatorFaultCode::OverSpeed;
    else if (_faultInjection.overTemperature)
        _state.fault = ActuatorFaultCode::OverTemperature;
    else if (_faultInjection.stalledMotor)
        _state.fault = ActuatorFaultCode::StalledMotor;
    else if (_state.windingTemperature > _parameters.maxWindingTemperature)
        _state.fault = ActuatorFaultCode::OverTemperature;
    else if (_faultInjection.hallFailure ||
             _faultInjection.currentSensorFailure ||
             _faultInjection.temperatureSensorFailure)
        _state.fault = ActuatorFaultCode::SensorInvalid;
    else if (_state.speedLimited)
        _state.fault = ActuatorFaultCode::OverSpeed;
}

void SimBLDCMotor::reset() noexcept {
    _state = {};
    _state.windingTemperature = _parameters.ambientTemperature;
    _requestedVoltage = 0.0;
    _faultInjection = {};
}

void SimBLDCMotor::injectFault(const SimFaultInjection &fault) noexcept {
    _faultInjection = fault;
}

const MotorParameters &SimBLDCMotor::parameters() const noexcept {
    return _parameters;
}

const MotorState &SimBLDCMotor::state() const noexcept { return _state; }

SimTransmission::SimTransmission(ScrewTransmissionParameters parameters)
    : _parameters(std::move(parameters)) {
    validateTransmission(_parameters);
}

TransmissionState SimTransmission::evaluate(double motorPosition,
                                            double motorVelocity,
                                            double motorTorque) const {
    // Apply gear ratio, screw lead and backlash before enforcing stroke limits.
    const double screwPosition = motorPosition / _parameters.gearRatio;
    const double rawDisplacement =
        screwPosition * _parameters.lead / (2.0 * std::numbers::pi);
    const double displacement =
        applyBacklash(rawDisplacement, _parameters.backlash);
    TransmissionState state;
    state.displacement = std::clamp(displacement, _parameters.minPosition,
                                    _parameters.maxPosition);
    state.lowerLimit = displacement <= _parameters.minPosition;
    state.upperLimit = displacement >= _parameters.maxPosition;
    state.velocity = (state.lowerLimit && motorVelocity < 0.0) ||
                             (state.upperLimit && motorVelocity > 0.0)
                         ? 0.0
                         : motorVelocity / _parameters.gearRatio *
                               _parameters.lead / (2.0 * std::numbers::pi);
    const double screwTorque = motorTorque * _parameters.gearRatio;
    state.force = 2.0 * std::numbers::pi * _parameters.efficiency *
                  screwTorque / _parameters.lead;
    return state;
}

double SimTransmission::motorLoadTorque(double linearForce) const {
    return linearForce * _parameters.lead /
           (2.0 * std::numbers::pi * _parameters.efficiency *
            _parameters.gearRatio);
}

const ScrewTransmissionParameters &
SimTransmission::parameters() const noexcept {
    return _parameters;
}

ConstantMomentArmModel::ConstantMomentArmModel(double momentArm)
    : _momentArm(momentArm) {
    requirePositive(std::abs(_momentArm), "moment arm magnitude");
}

double ConstantMomentArmModel::momentArm(double) const { return _momentArm; }

SimMuscleActuator::SimMuscleActuator(MuscleActuatorParameters parameters,
                                     std::unique_ptr<IMomentArmModel> momentArm)
    : _parameters(std::move(parameters)),
      _transmission(_parameters.transmission),
      _momentArm(std::move(momentArm)) {
    if (_parameters.motors.empty())
        throw std::invalid_argument("muscle actuator needs at least one motor");
    if (!_momentArm)
        throw std::invalid_argument("muscle actuator needs a moment arm model");
    requirePositive(_parameters.maxTendonForce, "maximum tendon force");
    _motors.reserve(_parameters.motors.size());
    for (const auto &parameters : _parameters.motors)
        _motors.emplace_back(parameters);
}

void SimMuscleActuator::commandVoltage(double voltage) noexcept {
    _requestedVoltage = voltage;
}

void SimMuscleActuator::commandForce(double force) noexcept {
    force = std::clamp(force, -_parameters.maxTendonForce,
                       _parameters.maxTendonForce);
    // Split the requested tendon load equally across parallel motors.
    const double totalMotorTorque = _transmission.motorLoadTorque(force);
    const double torquePerMotor =
        totalMotorTorque / static_cast<double>(_motors.size());
    double voltageSum = 0.0;
    for (const auto &motor : _motors) {
        const auto &parameters = motor.parameters();
        const auto &state = motor.state();
        const double desiredCurrent =
            torquePerMotor / parameters.torqueConstant;
        voltageSum += parameters.resistance * desiredCurrent +
                      parameters.backEmfConstant * state.angularVelocity;
    }
    _requestedVoltage = voltageSum / static_cast<double>(_motors.size());
}

void SimMuscleActuator::commandJointTorque(double jointTorque,
                                           double jointPosition) noexcept {
    const double arm = _momentArm->momentArm(jointPosition);
    commandForce(arm == 0.0 ? 0.0 : jointTorque / arm);
}

void SimMuscleActuator::step(double dtSeconds, double jointPosition,
                             double externalLoadTorque) {
    requirePositive(dtSeconds, "muscle actuator dt");
    const double arm = _momentArm->momentArm(jointPosition);
    // Reflect external joint load through moment arm and transmission.
    const double externalForce = arm == 0.0 ? 0.0 : externalLoadTorque / arm;
    const double loadPerMotor = _transmission.motorLoadTorque(externalForce) /
                                static_cast<double>(_motors.size());
    for (auto &motor : _motors) {
        motor.setVoltage(_requestedVoltage);
        motor.injectFault(_faultInjection);
        motor.step(dtSeconds, loadPerMotor);
    }

    double position{};
    double velocity{};
    double torque{};
    _state.motors.clear();
    for (const auto &motor : _motors) {
        const auto &motorState = motor.state();
        position += motorState.position;
        velocity += motorState.angularVelocity;
        torque += motorState.torque;
        _state.motors.push_back(motorState);
    }
    // Average kinematics but sum torque for motors acting in parallel.
    position /= static_cast<double>(_motors.size());
    velocity /= static_cast<double>(_motors.size());
    const auto transmission =
        _transmission.evaluate(position, velocity, torque);
    const bool stoppedAtLimit = (transmission.lowerLimit && velocity < 0.0) ||
                                (transmission.upperLimit && velocity > 0.0);
    if (stoppedAtLimit) {
        // The screw cannot keep accelerating through its finite travel stop.
        // Retaining current preserves the available holding force.
        for (auto &motor : _motors)
            motor._state.angularVelocity = 0.0;
        for (auto &motor : _state.motors)
            motor.angularVelocity = 0.0;
    }
    _state.displacement = transmission.displacement;
    _state.velocity = transmission.velocity;
    _state.tendonForce = _faultInjection.tendonBreak ? 0.0 : transmission.force;
    _state.estimatedJointTorque = _state.tendonForce * arm;
    _state.lowerLimit = transmission.lowerLimit;
    _state.upperLimit = transmission.upperLimit;
    _state.fault = ActuatorFaultCode::None;
    if (_faultInjection.tendonBreak)
        _state.fault = ActuatorFaultCode::TendonBreak;
    else if (std::abs(transmission.force) > _parameters.maxTendonForce) {
        _state.tendonForce =
            std::clamp(transmission.force, -_parameters.maxTendonForce,
                       _parameters.maxTendonForce);
        _state.estimatedJointTorque = _state.tendonForce * arm;
        _state.fault = ActuatorFaultCode::TendonOverload;
    } else if (_state.lowerLimit || _state.upperLimit)
        _state.fault = ActuatorFaultCode::OverTravel;
    else
        for (const auto &motor : _motors)
            if (motor.state().fault != ActuatorFaultCode::None) {
                _state.fault = motor.state().fault;
                break;
            }
}

void SimMuscleActuator::reset() noexcept {
    for (auto &motor : _motors)
        motor.reset();
    _state = {};
    _requestedVoltage = 0.0;
    _faultInjection = {};
}

void SimMuscleActuator::injectFault(const SimFaultInjection &fault) noexcept {
    _faultInjection = fault;
}

const MuscleActuatorState &SimMuscleActuator::state() const noexcept {
    return _state;
}

const MuscleActuatorParameters &SimMuscleActuator::parameters() const noexcept {
    return _parameters;
}

double SimMuscleActuator::momentArm(double jointPosition) const {
    return _momentArm->momentArm(jointPosition);
}

SimWaistLinearCylinder::SimWaistLinearCylinder(
    WaistLinearCylinderParameters parameters)
    : _parameters(std::move(parameters)), _motor(_parameters.motor),
      _transmission(_parameters.transmission) {
    requirePositive(_parameters.maxForce, "maximum cylinder force");
}

void SimWaistLinearCylinder::setVoltage(double voltage) noexcept {
    _motor.setVoltage(voltage);
}

void SimWaistLinearCylinder::step(double dtSeconds, double externalForce) {
    _motor.injectFault(_faultInjection);
    _motor.step(dtSeconds, _transmission.motorLoadTorque(externalForce));
    // Convert motor state once, then clamp the Plant-facing cylinder force.
    auto state = _transmission.evaluate(_motor.state().position,
                                        _motor.state().angularVelocity,
                                        _motor.state().torque);
    _state.length = state.displacement;
    _state.velocity = state.velocity;
    _state.estimatedForce =
        std::clamp(state.force, -_parameters.maxForce, _parameters.maxForce);
    _state.current = _motor.state().current;
    _state.temperature = _motor.state().windingTemperature;
    _state.lowerLimit = state.lowerLimit;
    _state.upperLimit = state.upperLimit;
    _state.fault = _motor.state().fault;
    if (std::abs(state.force) > _parameters.maxForce)
        _state.fault = ActuatorFaultCode::TendonOverload;
    else if (state.lowerLimit || state.upperLimit)
        _state.fault = ActuatorFaultCode::OverTravel;
}

void SimWaistLinearCylinder::reset() noexcept {
    _motor.reset();
    _state = {};
    _state.temperature = _parameters.motor.ambientTemperature;
    _faultInjection = {};
}

void SimWaistLinearCylinder::injectFault(
    const SimFaultInjection &fault) noexcept {
    _faultInjection = fault;
}

const LinearCylinderState &SimWaistLinearCylinder::state() const noexcept {
    return _state;
}

SimKneeLock::SimKneeLock(KneeLockParameters parameters)
    : _parameters(std::move(parameters)) {
    requirePositive(_parameters.maxEngageVelocity,
                    "maximum lock engage velocity");
    requirePositive(_parameters.engageTime, "lock engage time");
    requirePositive(_parameters.releaseTime, "lock release time");
    requirePositive(_parameters.holdingTorqueLimit, "lock torque limit");
    requirePositive(_parameters.constraintStiffness, "lock stiffness");
}

// Lock engagement captures position only below the safe velocity threshold.
void SimKneeLock::lock(double jointPosition, double jointVelocity) {
    if (_state == KneeLockState::Locked || _state == KneeLockState::Engaging)
        return;
    if (_faultInjection.lockFailure ||
        std::abs(jointVelocity) > _parameters.maxEngageVelocity) {
        _state = KneeLockState::Fault;
        return;
    }
    _lockedPosition = jointPosition;
    _transitionRemaining = _parameters.engageTime;
    _state = KneeLockState::Engaging;
}

void SimKneeLock::release() {
    if (_state == KneeLockState::Released || _state == KneeLockState::Releasing)
        return;
    if (_faultInjection.lockFailure) {
        _state = KneeLockState::Fault;
        return;
    }
    _transitionRemaining = _parameters.releaseTime;
    _state = KneeLockState::Releasing;
}

void SimKneeLock::step(double dtSeconds, double, double) {
    requirePositive(dtSeconds, "knee lock dt");
    if (_faultInjection.lockFailure && _state != KneeLockState::Released) {
        _state = KneeLockState::Fault;
        return;
    }
    if (_state != KneeLockState::Engaging && _state != KneeLockState::Releasing)
        return;
    // Transition time models finite mechanical engage and release latency.
    _transitionRemaining -= dtSeconds;
    if (_transitionRemaining <= 0.0)
        _state = _state == KneeLockState::Engaging ? KneeLockState::Locked
                                                   : KneeLockState::Released;
}

void SimKneeLock::resetFault() noexcept {
    if (_state == KneeLockState::Fault)
        _state = KneeLockState::Released;
    _faultInjection = {};
}

void SimKneeLock::injectFault(const SimFaultInjection &fault) noexcept {
    _faultInjection = fault;
}

KneeLockState SimKneeLock::state() const noexcept { return _state; }

KneeLockConstraint SimKneeLock::constraint() const noexcept {
    const bool active = _state == KneeLockState::Locked;
    return {active, _lockedPosition,
            active ? _parameters.constraintStiffness : 0.0,
            active ? _parameters.holdingTorqueLimit : 0.0};
}

SimSpringDamper::SimSpringDamper(SpringDamperParameters parameters)
    : _parameters(std::move(parameters)) {
    if (_parameters.springConstant < 0.0 ||
        _parameters.dampingCoefficient < 0.0 ||
        _parameters.mechanicalStopStiffness < 0.0)
        throw std::invalid_argument(
            "spring-damper coefficients must be non-negative");
    if (_parameters.minPosition > _parameters.maxPosition)
        throw std::invalid_argument("invalid spring-damper limits");
}

SpringDamperState SimSpringDamper::evaluate(double position,
                                            double velocity) const {
    SpringDamperState state;
    state.position = position;
    state.velocity = velocity;
    state.deflection = position - _parameters.neutralPosition;
    // Passive torque opposes both elastic deflection and joint velocity.
    state.reactionTorque = -_parameters.springConstant * state.deflection -
                           _parameters.dampingCoefficient * velocity;
    state.lowerLimit = position < _parameters.minPosition;
    state.upperLimit = position > _parameters.maxPosition;
    if (state.lowerLimit)
        state.reactionTorque += _parameters.mechanicalStopStiffness *
                                (_parameters.minPosition - position);
    else if (state.upperLimit)
        state.reactionTorque -= _parameters.mechanicalStopStiffness *
                                (position - _parameters.maxPosition);
    return state;
}

} // namespace ai::actuator::sim
