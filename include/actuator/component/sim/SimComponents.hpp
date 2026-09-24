#pragma once

#include "actuator/common/ActuatorTypes.hpp"

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

namespace ai::actuator::sim {

/// Traceability IDs for the modeled mechanical components.
inline constexpr std::string_view motorComponentId{"CMP-MOT-001"};
inline constexpr std::string_view transmissionComponentId{"CMP-TRN-001"};
inline constexpr std::string_view muscleComponentId{"CMP-MUS-001"};
inline constexpr std::string_view linearCylinderComponentId{"CMP-LIN-001"};
inline constexpr std::string_view kneeLockComponentId{"CMP-LCK-001"};
inline constexpr std::string_view springDamperComponentId{"CMP-SPR-001"};

/// Available documented BLDC motor parameter presets.
enum class MotorPreset { Ect35_80, Ect48_35, Ect60_21 };

/// Electrical, mechanical and first-order thermal motor parameters.
struct MotorParameters {
        std::string componentId{motorComponentId}; ///< Traceability ID.
        std::string article;        ///< Human-readable motor variant name.
        double resistance{};        ///< Phase resistance in ohms.
        double inductance{};        ///< Phase inductance in henries.
        double torqueConstant{};    ///< Torque constant in Nm/A.
        double backEmfConstant{};   ///< Back-EMF constant in V/(rad/s).
        double rotorInertia{};      ///< Rotor inertia in kg m^2.
        double viscousFriction{};   ///< Viscous drag in Nm/(rad/s).
        double nominalVoltage{};    ///< Nominal supply voltage in V.
        double continuousCurrent{}; ///< Continuous current limit in A.
        double maxSpeed{};          ///< Absolute rotor speed limit in rad/s.
        double maxWindingTemperature{}; ///< Winding limit in degrees Celsius.
        double thermalResistance{}; ///< Winding-to-ambient thermal resistance.
        double
            thermalTimeConstant{}; ///< First-order thermal time constant in s.
        double ambientTemperature{
            25.0}; ///< Ambient temperature in degrees Celsius.
};

/// Returns the SI parameter set for a documented motor preset.
[[nodiscard]] MotorParameters motorParameters(MotorPreset preset);

/// Dynamic state of one simulated BLDC motor.
struct MotorState {
        double current{};                ///< Winding current in A.
        double angularVelocity{};        ///< Rotor speed in rad/s.
        double position{};               ///< Rotor position in radians.
        double torque{};                 ///< Electromagnetic torque in Nm.
        double windingTemperature{25.0}; ///< Estimated winding temperature.
        double appliedVoltage{};         ///< Clamped terminal voltage in V.
        bool currentLimited{}; ///< True when current limiting is active.
        bool speedLimited{};   ///< True when the speed limit is active.
        ActuatorFaultCode fault{ActuatorFaultCode::None}; ///< Active fault.
};

/// Lumped electrical, rotor and thermal BLDC motor simulation.
class SimBLDCMotor {
    public:
        /// Constructs a motor and validates its parameters.
        explicit SimBLDCMotor(MotorParameters);
        /// Sets requested voltage for the next integration step.
        void setVoltage(double voltage) noexcept;
        /// Advances electrical, mechanical and thermal state.
        void step(double dtSeconds, double loadTorque);
        /// Restores ambient, stationary state.
        void reset() noexcept;
        /// Updates deterministic fault switches.
        void injectFault(const SimFaultInjection &) noexcept;
        /// Returns immutable motor parameters.
        [[nodiscard]] const MotorParameters &parameters() const noexcept;
        /// Returns the latest dynamic state.
        [[nodiscard]] const MotorState &state() const noexcept;

    private:
        // The owning transmission clamps rotor motion at a mechanical stop.
        friend class SimMuscleActuator;
        MotorParameters _parameters; ///< Validated physical parameters.
        MotorState _state;           ///< Integrated motor state.
        double _requestedVoltage{};  ///< Unclamped requested terminal voltage.
        SimFaultInjection _faultInjection; ///< Active deterministic faults.
};

/// Screw and gear transmission parameters.
struct ScrewTransmissionParameters {
        std::string componentId{transmissionComponentId}; ///< Traceability ID.
        double lead{};         ///< Linear travel per screw revolution in m/rev.
        double efficiency{};   ///< Power conversion efficiency from 0 to 1.
        double gearRatio{1.0}; ///< Motor-to-screw reduction ratio.
        double backlash{};     ///< Total deadband in metres.
        double minPosition{};  ///< Lower stroke limit in metres.
        double maxPosition{};  ///< Upper stroke limit in metres.
};

/// Derived linear transmission state.
struct TransmissionState {
        double displacement{}; ///< Clamped output displacement in metres.
        double velocity{};     ///< Linear output velocity in m/s.
        double force{};        ///< Ideal efficiency-adjusted output force in N.
        bool lowerLimit{};     ///< True at the lower stroke stop.
        bool upperLimit{};     ///< True at the upper stroke stop.
};

/// Converts rotary motor state and load to a linear screw output.
class SimTransmission {
    public:
        /// Constructs and validates a transmission.
        explicit SimTransmission(ScrewTransmissionParameters);
        /// Computes linear displacement, speed, force and limit state.
        [[nodiscard]] TransmissionState evaluate(double motorPosition,
                                                 double motorVelocity,
                                                 double motorTorque) const;
        /// Reflects an external linear force back to the motor shaft.
        [[nodiscard]] double motorLoadTorque(double linearForce) const;
        /// Returns immutable transmission parameters.
        [[nodiscard]] const ScrewTransmissionParameters &
        parameters() const noexcept;

    private:
        ScrewTransmissionParameters _parameters; ///< Validated conversion data.
};

/// Strategy interface for joint-position-dependent tendon moment arms.
class IMomentArmModel {
    public:
        virtual ~IMomentArmModel() = default;
        /// Returns signed moment arm in metres at a joint position.
        [[nodiscard]] virtual double momentArm(double jointPosition) const = 0;
};

/// Position-independent moment-arm approximation.
class ConstantMomentArmModel final : public IMomentArmModel {
    public:
        /// Creates a model with one signed moment arm in metres.
        explicit ConstantMomentArmModel(double momentArm);
        /// Returns the configured moment arm.
        [[nodiscard]] double momentArm(double jointPosition) const override;

    private:
        double _momentArm; ///< Constant signed moment arm in metres.
};

/// Configuration for a parallel-motor tendon Actuator.
struct MuscleActuatorParameters {
        std::string componentId{muscleComponentId}; ///< Traceability ID.
        std::vector<MotorParameters> motors; ///< Parallel motor parameter sets.
        ScrewTransmissionParameters transmission; ///< Shared transmission.
        double maxTendonForce{}; ///< Absolute tendon force limit in N.
};

/// Aggregated dynamic state of a muscle-style Actuator.
struct MuscleActuatorState {
        double displacement{};          ///< Tendon displacement in metres.
        double velocity{};              ///< Tendon velocity in m/s.
        double tendonForce{};           ///< Total tendon tension in N.
        double estimatedJointTorque{};  ///< Moment-arm torque estimate in Nm.
        std::vector<MotorState> motors; ///< State of every parallel motor.
        bool lowerLimit{}; ///< True at the lower transmission stop.
        bool upperLimit{}; ///< True at the upper transmission stop.
        ActuatorFaultCode fault{ActuatorFaultCode::None}; ///< Aggregated fault.
};

/// Motor, transmission and tendon model for a muscle-style Actuator.
class SimMuscleActuator {
    public:
        /// Builds component models and takes ownership of the moment-arm model.
        SimMuscleActuator(MuscleActuatorParameters,
                          std::unique_ptr<IMomentArmModel>);
        /// Commands identical motor terminal voltage.
        void commandVoltage(double voltage) noexcept;
        /// Converts desired tendon force into motor voltage demand.
        void commandForce(double force) noexcept;
        /// Converts desired joint torque using the current moment arm.
        void commandJointTorque(double jointTorque,
                                double jointPosition) noexcept;
        /// Advances every component using explicit simulation time.
        void step(double dtSeconds, double jointPosition,
                  double externalLoadTorque);
        /// Restores all component state.
        void reset() noexcept;
        /// Applies deterministic faults to the aggregate and child motors.
        void injectFault(const SimFaultInjection &) noexcept;
        /// Returns the latest aggregate state.
        [[nodiscard]] const MuscleActuatorState &state() const noexcept;
        /// Returns immutable aggregate parameters.
        [[nodiscard]] const MuscleActuatorParameters &
        parameters() const noexcept;
        /// Returns the signed moment arm for a joint position.
        [[nodiscard]] double momentArm(double jointPosition) const;

    private:
        MuscleActuatorParameters _parameters; ///< Aggregate configuration.
        std::vector<SimBLDCMotor> _motors;    ///< Parallel simulated motors.
        SimTransmission _transmission;        ///< Rotary-to-linear conversion.
        std::unique_ptr<IMomentArmModel> _momentArm; ///< Joint geometry model.
        MuscleActuatorState _state;        ///< Latest aggregate state.
        SimFaultInjection _faultInjection; ///< Active aggregate faults.
        double _requestedVoltage{}; ///< Voltage applied on the next step.
};

/// Motor and transmission configuration for a waist cylinder.
struct WaistLinearCylinderParameters {
        std::string componentId{
            linearCylinderComponentId};           ///< Traceability ID.
        MotorParameters motor;                    ///< Driving motor parameters.
        ScrewTransmissionParameters transmission; ///< Screw conversion data.
        double maxForce{}; ///< Absolute cylinder force limit in N.
};

/// Observable state of a linear cylinder.
struct LinearCylinderState {
        double length{};          ///< Current stroke position in metres.
        double velocity{};        ///< Current stroke speed in m/s.
        double estimatedForce{};  ///< Estimated output force in N.
        double current{};         ///< Driving motor current in A.
        double temperature{25.0}; ///< Driving motor winding temperature.
        bool lowerLimit{};        ///< True at the lower stroke stop.
        bool upperLimit{};        ///< True at the upper stroke stop.
        ActuatorFaultCode fault{ActuatorFaultCode::None}; ///< Active fault.
};

/// Component simulation of a motor-driven waist linear cylinder.
class SimWaistLinearCylinder {
    public:
        /// Constructs a cylinder from explicit physical parameters.
        explicit SimWaistLinearCylinder(WaistLinearCylinderParameters);
        /// Sets motor voltage for the next simulation step.
        void setVoltage(double voltage) noexcept;
        /// Advances motor and transmission against an external force.
        void step(double dtSeconds, double externalForce);
        /// Restores a stationary initial state.
        void reset() noexcept;
        /// Applies deterministic motor and sensor faults.
        void injectFault(const SimFaultInjection &) noexcept;
        /// Returns the latest cylinder state.
        [[nodiscard]] const LinearCylinderState &state() const noexcept;

    private:
        WaistLinearCylinderParameters _parameters; ///< Cylinder configuration.
        SimBLDCMotor _motor;               ///< Driving motor simulation.
        SimTransmission _transmission;     ///< Rotary-to-linear conversion.
        LinearCylinderState _state;        ///< Latest derived cylinder state.
        SimFaultInjection _faultInjection; ///< Active deterministic faults.
};

/// State machine states for a finite-stiffness knee lock.
enum class KneeLockState { Released, Engaging, Locked, Releasing, Fault };

/// Timing and effort limits for the knee lock.
struct KneeLockParameters {
        std::string componentId{kneeLockComponentId}; ///< Traceability ID.
        double maxEngageVelocity{};   ///< Maximum safe lock speed in rad/s.
        double engageTime{};          ///< Engage transition duration in s.
        double releaseTime{};         ///< Release transition duration in s.
        double holdingTorqueLimit{};  ///< Absolute holding limit in Nm.
        double constraintStiffness{}; ///< Finite lock stiffness in Nm/rad.
};

/// Plant-facing constraint produced by the knee-lock state machine.
struct KneeLockConstraint {
        bool active{};        ///< True while a lock constraint must be applied.
        double position{};    ///< Captured lock position in radians.
        double stiffness{};   ///< Finite lock stiffness in Nm/rad.
        double torqueLimit{}; ///< Absolute holding limit in Nm.
};

/// Deterministic engage/release state machine for a knee lock.
class SimKneeLock {
    public:
        /// Constructs and validates lock parameters.
        explicit SimKneeLock(KneeLockParameters);
        /// Begins engagement if joint velocity is within its safe limit.
        void lock(double jointPosition, double jointVelocity);
        /// Begins the release transition.
        void release();
        /// Advances the lock transition using explicit simulation time.
        void step(double dtSeconds, double jointPosition, double jointVelocity);
        /// Clears a recoverable lock fault.
        void resetFault() noexcept;
        /// Applies deterministic lock failure state.
        void injectFault(const SimFaultInjection &) noexcept;
        /// Returns the current lock state-machine state.
        [[nodiscard]] KneeLockState state() const noexcept;
        /// Returns the constraint currently requested from the Plant.
        [[nodiscard]] KneeLockConstraint constraint() const noexcept;

    private:
        KneeLockParameters _parameters; ///< Validated lock configuration.
        KneeLockState _state{KneeLockState::Released}; ///< Current state.
        SimFaultInjection _faultInjection; ///< Active deterministic faults.
        double _transitionRemaining{};     ///< Remaining transition time in s.
        double _lockedPosition{}; ///< Joint position captured at engagement.
};

/// Passive joint spring, damper and mechanical-stop parameters.
struct SpringDamperParameters {
        std::string componentId{springDamperComponentId}; ///< Traceability ID.
        double springConstant{};     ///< Linear stiffness in Nm/rad.
        double dampingCoefficient{}; ///< Viscous damping in Nm/(rad/s).
        double neutralPosition{}; ///< Zero-spring-torque position in radians.
        double minPosition{};     ///< Lower mechanical limit in radians.
        double maxPosition{};     ///< Upper mechanical limit in radians.
        double mechanicalStopStiffness{}; ///< Stop stiffness in Nm/rad.
};

/// Result of evaluating a passive spring-damper element.
struct SpringDamperState {
        double position{};       ///< Input joint position in radians.
        double velocity{};       ///< Input joint velocity in rad/s.
        double deflection{};     ///< Position relative to neutral in radians.
        double reactionTorque{}; ///< Opposing passive torque in Nm.
        bool lowerLimit{};       ///< True beyond the lower mechanical stop.
        bool upperLimit{};       ///< True beyond the upper mechanical stop.
};

/// Stateless evaluator for passive elasticity, damping and joint stops.
class SimSpringDamper {
    public:
        /// Constructs and validates passive-element parameters.
        explicit SimSpringDamper(SpringDamperParameters);
        /// Evaluates reaction torque without advancing hidden state.
        [[nodiscard]] SpringDamperState evaluate(double position,
                                                 double velocity) const;

    private:
        SpringDamperParameters _parameters; ///< Validated passive parameters.
};

} // namespace ai::actuator::sim
