#include "actuator/driver/sim/SimActuatorDriver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ai::actuator::sim {
namespace {

// Applies descriptor limits when they form a valid inclusive range.
double clampToRange(double value, const Range &range) {
    return range.valid() ? std::clamp(value, range.minimum, range.maximum)
                         : value;
}

} // namespace

// Provides an explicit prototype default, not an identified robot value.
SimActuatorDriverConfig defaultElbowMuscleConfig() {
    SimActuatorDriverConfig config;
    config.muscle.motors = {motorParameters(MotorPreset::Ect35_80),
                            motorParameters(MotorPreset::Ect35_80)};
    // These are explicitly configurable simulation defaults because the
    // component documents leave screw selection and elbow geometry open.
    config.muscle.transmission = {std::string{transmissionComponentId},
                                  0.005,
                                  0.85,
                                  4.0,
                                  0.0001,
                                  -0.05,
                                  0.05};
    config.muscle.maxTendonForce = 500.0;
    config.momentArm = 0.03;
    config.positionGain = 8.0;
    config.velocityGain = 1.5;
    return config;
}

SimActuatorDriver::SimActuatorDriver(SimActuatorDriverConfig config)
    : _config(std::move(config)),
      _actuator(_config.muscle,
                std::make_unique<ConstantMomentArmModel>(_config.momentArm)) {
    if (_config.positionGain < 0.0 || _config.velocityGain < 0.0)
        throw std::invalid_argument("driver gains must be non-negative");
}

void SimActuatorDriver::configure(const ActuatorDescriptor &descriptor) {
    if (descriptor.actuatorId.empty())
        throw std::invalid_argument("actuator id is empty");
    if (descriptor.type != ActuatorType::Muscle)
        throw std::invalid_argument("SimActuatorDriver expects muscle type");
    _descriptor = descriptor;
    _state = {};
    _state.actuatorId = descriptor.actuatorId;
    _state.status = ActuatorStatus::Disabled;
    _state.temperature = _config.muscle.motors.front().ambientTemperature;
    _configured = true;
    _enabled = false;
}

void SimActuatorDriver::enable() {
    if (!_configured)
        throw std::logic_error("driver is not configured");
    if (_state.fault != ActuatorFaultCode::None)
        throw std::logic_error("driver has an active fault");
    _enabled = true;
    _state.status = ActuatorStatus::Ready;
}

void SimActuatorDriver::disable() {
    _enabled = false;
    _actuator.commandVoltage(0.0);
    _state.status = ActuatorStatus::Disabled;
}

bool SimActuatorDriver::supports(ControlMode mode) const {
    return std::find(_descriptor.supportedModes.begin(),
                     _descriptor.supportedModes.end(),
                     mode) != _descriptor.supportedModes.end();
}

void SimActuatorDriver::command(const DriveCommand &command) {
    if (!_configured)
        throw std::logic_error("driver is not configured");
    if (command.actuatorId != _descriptor.actuatorId)
        throw std::invalid_argument("command actuator id mismatch");
    if (!supports(command.mode))
        throw std::invalid_argument("unsupported control mode");
    if (command.mode == ControlMode::Disable) {
        disable();
        _command = command;
        return;
    }
    if (!_enabled)
        throw std::logic_error("driver is disabled");
    _command = command;
    if (command.mode == ControlMode::Stop)
        stop(StopMode::ControlledStop);
}

void SimActuatorDriver::stop(StopMode mode) {
    _command.mode = ControlMode::Stop;
    _command.target = 0.0;
    _actuator.commandVoltage(0.0);
    _state.status = mode == StopMode::EmergencyOff
                        ? ActuatorStatus::EmergencyStop
                        : ActuatorStatus::Stopping;
    if (mode == StopMode::EmergencyOff)
        _enabled = false;
}

ActuatorState SimActuatorDriver::readState() { return _state; }

void SimActuatorDriver::resetFault() {
    _actuator.reset();
    _state.fault = ActuatorFaultCode::None;
    _state.status = _enabled ? ActuatorStatus::Ready : ActuatorStatus::Disabled;
    _output = {};
}

void SimActuatorDriver::advance(double dtSeconds,
                                const PlantFeedback &feedback) {
    if (!std::isfinite(dtSeconds) || dtSeconds <= 0.0)
        throw std::invalid_argument("driver dt must be positive");
    if (!_configured)
        throw std::logic_error("driver is not configured");
    _elapsedSeconds += dtSeconds;
    if (!_enabled) {
        _output = {};
        return;
    }

    // Convert each external control mode to one common joint-torque demand.
    switch (_command.mode) {
    case ControlMode::Torque:
        _actuator.commandJointTorque(
            clampToRange(_command.target, _descriptor.torqueLimit),
            feedback.jointPosition);
        break;
    case ControlMode::Current: {
        const double current =
            clampToRange(_command.target, _descriptor.currentLimit);
        double totalMotorTorque{};
        for (const auto &motor : _config.muscle.motors)
            totalMotorTorque += current * motor.torqueConstant;
        const SimTransmission transmission{_config.muscle.transmission};
        const double force =
            transmission.evaluate(0.0, 0.0, totalMotorTorque).force;
        _actuator.commandForce(force);
        break;
    }
    case ControlMode::Position: {
        const double target =
            clampToRange(_command.target, _descriptor.positionLimit);
        const double torque =
            _config.positionGain * (target - feedback.jointPosition) -
            _config.velocityGain * feedback.jointVelocity;
        _actuator.commandJointTorque(
            clampToRange(torque, _descriptor.torqueLimit),
            feedback.jointPosition);
        break;
    }
    case ControlMode::Velocity: {
        const double target =
            clampToRange(_command.target, _descriptor.velocityLimit);
        const double torque =
            _config.velocityGain * (target - feedback.jointVelocity);
        _actuator.commandJointTorque(
            clampToRange(torque, _descriptor.torqueLimit),
            feedback.jointPosition);
        break;
    }
    case ControlMode::Stop:
        _actuator.commandVoltage(0.0);
        break;
    case ControlMode::Disable:
        disable();
        return;
    case ControlMode::Lock:
    case ControlMode::Release:
        throw std::invalid_argument("lock mode requires a brake driver");
    }

    // Component simulation receives explicit dt and Plant load feedback only.
    _actuator.step(dtSeconds, feedback.jointPosition,
                   feedback.loadValid ? feedback.externalLoadTorque : 0.0);
    const auto &simulation = _actuator.state();
    // Publish mechanical output once, after every child component has advanced.
    _output = {};
    _output.jointTorque = simulation.estimatedJointTorque;
    _output.linearForce = simulation.tendonForce;
    _state.position = feedback.jointPosition;
    _state.velocity = feedback.jointVelocity;
    _state.torque = simulation.estimatedJointTorque;
    _state.linearPosition = simulation.displacement;
    _state.linearForce = simulation.tendonForce;
    _state.tendonTension = simulation.tendonForce;
    _state.lowerLimit = simulation.lowerLimit;
    _state.upperLimit = simulation.upperLimit;
    _state.fault = simulation.fault;
    if (!simulation.motors.empty()) {
        double current{};
        double maxTemperature = simulation.motors.front().windingTemperature;
        for (const auto &motor : simulation.motors) {
            current += motor.current;
            maxTemperature = std::max(maxTemperature, motor.windingTemperature);
        }
        _state.current =
            current / static_cast<double>(simulation.motors.size());
        _state.temperature = maxTemperature;
    }
    _state.timestamp =
        TimePoint{} + std::chrono::duration_cast<Duration>(
                          std::chrono::duration<double>{_elapsedSeconds});
    if (_state.fault == ActuatorFaultCode::None)
        _state.status = _command.mode == ControlMode::Stop
                            ? ActuatorStatus::Ready
                            : ActuatorStatus::Running;
    else if (_state.fault == ActuatorFaultCode::OverTravel ||
             _state.fault == ActuatorFaultCode::TendonOverload)
        _state.status = ActuatorStatus::Limited;
    else
        _state.status = ActuatorStatus::Fault;
}

PlantOutput SimActuatorDriver::plantOutput() const { return _output; }

void SimActuatorDriver::injectFault(const SimFaultInjection &fault) {
    _actuator.injectFault(fault);
}

std::unique_ptr<IActuatorDriver>
createActuatorDriver(const ActuatorDescriptor &descriptor, BackendType backend,
                     SimActuatorDriverConfig simulationConfig,
                     PhysicalDriverFactory physicalFactory) {
    if (backend == BackendType::Physical) {
        if (!physicalFactory)
            throw std::invalid_argument(
                "physical driver factory is not configured");
        auto driver = physicalFactory(descriptor);
        if (!driver)
            throw std::runtime_error("physical driver factory returned null");
        driver->configure(descriptor);
        return driver;
    }
    auto driver =
        std::make_unique<SimActuatorDriver>(std::move(simulationConfig));
    driver->configure(descriptor);
    return driver;
}

} // namespace ai::actuator::sim
