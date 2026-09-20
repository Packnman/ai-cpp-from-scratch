#pragma once

#include "actuator/common/ActuatorTypes.hpp"

namespace ai::actuator {

/// Common lifecycle and command interface for physical and simulated drivers.
class IActuatorDriver {
    public:
        virtual ~IActuatorDriver() = default;
        /// Applies static descriptor and safety limits.
        virtual void configure(const ActuatorDescriptor &) = 0;
        /// Enables drive output.
        virtual void enable() = 0;
        /// Disables drive output.
        virtual void disable() = 0;
        /// Accepts a validated mode-specific command.
        virtual void command(const DriveCommand &) = 0;
        /// Starts the requested stop behavior.
        virtual void stop(StopMode) = 0;
        /// Returns the latest observable state.
        virtual ActuatorState readState() = 0;
        /// Clears recoverable driver faults.
        virtual void resetFault() = 0;
};

/// Simulation-only extension driven by explicit deterministic time.
class ISimActuatorDriver : public IActuatorDriver {
    public:
        /// Advances component dynamics using Plant feedback.
        virtual void advance(double dtSeconds, const PlantFeedback &) = 0;
        /// Returns the mechanical output to apply to an IPlant.
        [[nodiscard]] virtual PlantOutput plantOutput() const = 0;
        /// Sets deterministic fault-injection switches.
        virtual void injectFault(const SimFaultInjection &) = 0;
};

} // namespace ai::actuator
