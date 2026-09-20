#pragma once

#include "actuator/component/sim/SimComponents.hpp"
#include "actuator/driver/IActuatorDriver.hpp"

#include <functional>
#include <memory>

namespace ai::actuator::sim {

/// Parameters used to assemble a muscle-style simulation driver.
struct SimActuatorDriverConfig {
        MuscleActuatorParameters muscle; ///< Motor/tendon component model.
        double momentArm{};    ///< Constant joint moment arm in metres.
        double positionGain{}; ///< Position-to-torque proportional gain.
        double velocityGain{}; ///< Velocity-to-torque proportional gain.
};

/// Returns a documented elbow configuration intended for tests and prototypes.
[[nodiscard]] SimActuatorDriverConfig defaultElbowMuscleConfig();

/// Deterministic Actuator driver backed by component-level physical models.
class SimActuatorDriver final : public ISimActuatorDriver {
    public:
        /// Constructs the driver from explicit simulation parameters.
        explicit SimActuatorDriver(SimActuatorDriverConfig);
        /// Validates and stores the Actuator descriptor.
        void configure(const ActuatorDescriptor &) override;
        /// Enables command execution when configured and fault-free.
        void enable() override;
        /// Disables drive output and clears pending mechanical output.
        void disable() override;
        /// Stores a supported command for subsequent advance calls.
        void command(const DriveCommand &) override;
        /// Applies the requested stopping policy.
        void stop(StopMode) override;
        /// Returns the latest synthesized driver and component state.
        ActuatorState readState() override;
        /// Clears recoverable faults and returns to standby.
        void resetFault() override;
        /// Advances motor/transmission dynamics by explicit simulation time.
        void advance(double dtSeconds, const PlantFeedback &) override;
        /// Returns the most recent Plant-facing mechanical output.
        [[nodiscard]] PlantOutput plantOutput() const override;
        /// Activates deterministic component and sensor faults.
        void injectFault(const SimFaultInjection &) override;

    private:
        /// Reports whether the configured descriptor supports a mode.
        [[nodiscard]] bool supports(ControlMode) const;
        SimActuatorDriverConfig _config; ///< Immutable model configuration.
        SimMuscleActuator _actuator;     ///< Owned muscle component model.
        ActuatorDescriptor _descriptor;  ///< Active capabilities and limits.
        DriveCommand _command;           ///< Latest accepted command.
        ActuatorState _state;            ///< Latest externally visible state.
        PlantOutput _output;             ///< Latest Plant-facing output.
        double _elapsedSeconds{}; ///< Deterministic elapsed simulation time.
        bool _configured{};       ///< True after configure succeeds.
        bool _enabled{};          ///< True while drive output is permitted.
};

/// Factory callback used to inject a platform-specific physical driver.
using PhysicalDriverFactory =
    std::function<std::unique_ptr<IActuatorDriver>(const ActuatorDescriptor &)>;

/// Creates either the built-in simulation backend or an injected physical one.
[[nodiscard]] std::unique_ptr<IActuatorDriver>
createActuatorDriver(const ActuatorDescriptor &, BackendType,
                     SimActuatorDriverConfig simulationConfig,
                     PhysicalDriverFactory physicalFactory = {});

} // namespace ai::actuator::sim
