#pragma once

#include "actuator/driver/IActuatorDriver.hpp"

#include <memory>
#include <string_view>
#include <unordered_map>

namespace ai::actuator {

/// Owns Actuator drivers and routes lifecycle, command and state operations.
class ActuatorManager {
    public:
        /// Registers one descriptor and its exclusively owned driver.
        void registerActuator(ActuatorDescriptor,
                              std::unique_ptr<IActuatorDriver>);
        /// Reports whether an Actuator ID has been registered.
        [[nodiscard]] bool contains(std::string_view actuatorId) const;
        /// Enables one registered Actuator.
        void enable(std::string_view actuatorId);
        /// Disables one registered Actuator.
        void disable(std::string_view actuatorId);
        /// Routes a command to its target Actuator.
        void command(const DriveCommand &);
        /// Requests the specified stop behavior.
        void stop(std::string_view actuatorId, StopMode);
        /// Reads the latest state from one driver.
        [[nodiscard]] ActuatorState readState(std::string_view actuatorId);
        /// Returns the static descriptor for one Actuator.
        [[nodiscard]] const ActuatorDescriptor &
        descriptor(std::string_view actuatorId) const;
        /// Returns a driver for backend-specific operations such as simulation
        /// advance.
        [[nodiscard]] IActuatorDriver &driver(std::string_view actuatorId);

    private:
        /// Descriptor and driver stored for one registered Actuator.
        struct Entry {
                ActuatorDescriptor descriptor; ///< Validated static metadata.
                std::unique_ptr<IActuatorDriver> driver; ///< Owned backend.
        };
        /// Resolves a mutable entry or throws for an unknown ID.
        [[nodiscard]] Entry &entry(std::string_view actuatorId);
        /// Resolves a read-only entry or throws for an unknown ID.
        [[nodiscard]] const Entry &entry(std::string_view actuatorId) const;
        std::unordered_map<ActuatorId, Entry> _entries; ///< Registry by ID.
};

} // namespace ai::actuator
