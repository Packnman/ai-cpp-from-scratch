#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ai::actuator {

/// Stable logical Actuator identifier.
using ActuatorId = std::string;
/// Monotonic identifier attached to one command.
using CommandId = std::uint64_t;
/// Host timestamp used for command freshness, not physics integration.
using TimePoint = std::chrono::steady_clock::time_point;
/// Command timeout duration.
using Duration = std::chrono::milliseconds;

/// Inclusive scalar operating range.
struct Range {
        double minimum{}; ///< Lowest permitted value.
        double maximum{}; ///< Highest permitted value.

        /// Reports whether the range ordering is valid.
        [[nodiscard]] bool valid() const noexcept { return minimum <= maximum; }
        /// Reports whether a value lies inside the inclusive range.
        [[nodiscard]] bool contains(double value) const noexcept {
            return value >= minimum && value <= maximum;
        }
};

/// Command interpretation supported by an Actuator driver.
enum class ControlMode {
    Position,
    Velocity,
    Torque,
    Current,
    Stop,
    Disable,
    Lock,
    Release
};

/// Requested behavior while stopping an Actuator.
enum class StopMode { Coast, Brake, Hold, ControlledStop, EmergencyOff };
/// Selects the deterministic simulation or injected hardware backend.
enum class BackendType { Simulation, Physical };
/// Mechanical Actuator category.
enum class ActuatorType {
    Muscle,
    LinearCylinder,
    KneeLock,
    PassiveSpringDamper
};
/// Driver lifecycle and safety state.
enum class ActuatorStatus {
    Disabled,
    Standby,
    Ready,
    Running,
    Limited,
    Stopping,
    Fault,
    EmergencyStop
};
/// Normalized fault codes shared by simulation and physical drivers.
enum class ActuatorFaultCode {
    None,
    OverCurrent,
    OverSpeed,
    OverTravel,
    OverTemperature,
    StalledMotor,
    TendonOverload,
    TendonBreak,
    LockFailure,
    SensorInvalid,
    InvalidCommand
};

/// Static capabilities and safety limits for one Actuator.
struct ActuatorDescriptor {
        ActuatorId actuatorId;   ///< Logical Actuator identifier.
        std::string componentId; ///< Mechanical component specification ID.
        ActuatorType type{ActuatorType::Muscle}; ///< Mechanical category.
        Range positionLimit{};    ///< Allowed position in rad or m.
        Range velocityLimit{};    ///< Allowed velocity in rad/s or m/s.
        Range torqueLimit{};      ///< Allowed rotary effort in Nm.
        Range currentLimit{};     ///< Allowed motor current in A.
        Range temperatureLimit{}; ///< Allowed temperature in degrees Celsius.
        std::vector<ControlMode> supportedModes; ///< Accepted control modes.
        std::uint32_t version{1}; ///< Descriptor schema/configuration version.
};

/// One timestamped command delivered to an Actuator driver.
struct DriveCommand {
        CommandId commandId{};               ///< Unique command identifier.
        ActuatorId actuatorId;               ///< Intended target Actuator.
        ControlMode mode{ControlMode::Stop}; ///< Requested control mode.
        double target{};                     ///< Mode-specific SI target value.
        std::optional<double> limit; ///< Optional mode-specific effort limit.
        TimePoint timestamp{}; ///< Host issue time for timeout validation.
        Duration timeout{};    ///< Maximum command lifetime.
};

/// Observable state returned by a simulation or physical driver.
struct ActuatorState {
        ActuatorId actuatorId;    ///< Logical Actuator identifier.
        double position{};        ///< Rotary position in radians.
        double velocity{};        ///< Rotary velocity in rad/s.
        double torque{};          ///< Estimated output torque in Nm.
        double current{};         ///< Motor current in A.
        double temperature{25.0}; ///< Winding temperature in degrees Celsius.
        double linearPosition{};  ///< Linear displacement in metres.
        double linearForce{};     ///< Linear output force in N.
        double tendonTension{};   ///< Tendon tension in N.
        ActuatorStatus status{ActuatorStatus::Disabled};  ///< Lifecycle state.
        ActuatorFaultCode fault{ActuatorFaultCode::None}; ///< Active fault.
        bool lowerLimit{};     ///< True at the lower mechanical stop.
        bool upperLimit{};     ///< True at the upper mechanical stop.
        TimePoint timestamp{}; ///< Host observation time.
};

/// Plant feedback consumed by the component-level Actuator simulation.
struct PlantFeedback {
        double jointPosition{};      ///< Joint position in radians.
        double jointVelocity{};      ///< Joint velocity in rad/s.
        double externalLoadTorque{}; ///< External joint load in Nm.
        bool loadValid{true};        ///< Whether the load estimate may be used.
};

/// Engine-independent mechanical output sent from Actuator to Plant.
struct PlantOutput {
        double jointTorque{};           ///< Active joint torque in Nm.
        double linearForce{};           ///< Linear Actuator force in N.
        double passiveReactionTorque{}; ///< Spring/damper reaction in Nm.
        bool constraintActive{}; ///< Whether a lock constraint is requested.
        double constraintPosition{};    ///< Lock target position in radians.
        double constraintStiffness{};   ///< Finite lock stiffness.
        double constraintTorqueLimit{}; ///< Lock holding limit in Nm.
};

/// Deterministic fault switches used only by simulation drivers.
struct SimFaultInjection {
        bool hallFailure{};          ///< Invalidates motor position sensing.
        bool currentSensorFailure{}; ///< Invalidates measured current.
        bool tendonBreak{};          ///< Removes tendon force transmission.
        bool lockFailure{};          ///< Prevents or releases knee locking.
        bool temperatureSensorFailure{}; ///< Invalidates temperature sensing.
        bool overCurrent{};              ///< Forces an over-current fault.
        bool overSpeed{};                ///< Forces an over-speed fault.
        bool overTemperature{};          ///< Forces an over-temperature fault.
        bool stalledMotor{}; ///< Forces zero motor motion under command.
};

} // namespace ai::actuator
