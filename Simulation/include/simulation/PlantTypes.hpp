#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ai::simulation {

/// Logical joint name shared across subsystem boundaries.
using JointId = std::string;
/// Logical rigid-body name shared across subsystem boundaries.
using BodyId = std::string;
/// Logical MuJoCo site name.
using SiteId = std::string;
/// Logical sensor name.
using SensorId = std::string;
/// Logical equality or finite-lock constraint name.
using ConstraintId = std::string;

/// Three-dimensional vector expressed in SI units.
struct Vec3 {
        double x{}; ///< X component.
        double y{}; ///< Y component.
        double z{}; ///< Z component.
};

/// Unit quaternion using MuJoCo's w, x, y, z ordering.
struct Quaternion {
        double w{1.0}; ///< Scalar component.
        double x{};    ///< X imaginary component.
        double y{};    ///< Y imaginary component.
        double z{};    ///< Z imaginary component.
};

/// Pose and world-frame velocity of one rigid body.
struct RigidBodyState {
        BodyId id;              ///< Logical body name.
        Vec3 position;          ///< World position in metres.
        Quaternion orientation; ///< World orientation.
        Vec3 linearVelocity;    ///< World linear velocity in m/s.
        Vec3 angularVelocity;   ///< World angular velocity in rad/s.
};

/// Scalar state exposed for a hinge or slide joint.
struct JointState {
        JointId id;             ///< Logical joint name.
        double position{};      ///< Position in rad or m.
        double velocity{};      ///< Velocity in rad/s or m/s.
        double acceleration{};  ///< Acceleration in rad/s^2 or m/s^2.
        double appliedTorque{}; ///< Applied generalized effort in Nm or N.
};

/// One active contact and its force in the world frame.
struct ContactState {
        BodyId bodyA;         ///< First contacting body.
        BodyId bodyB;         ///< Second contacting body.
        Vec3 position;        ///< World contact point in metres.
        Vec3 normal;          ///< Contact normal from MuJoCo.
        Vec3 force;           ///< World-frame contact force in newtons.
        double normalForce{}; ///< Normal component magnitude in newtons.
        double distance{};    ///< Signed contact distance in metres.
};

/// Semantic category used to interpret sensor values.
enum class SensorKind {
    JointPosition,
    JointVelocity,
    Orientation,
    AngularVelocity,
    Contact,
    Other
};

/// Values produced by one named simulated sensor.
struct SensorState {
        SensorId id;                        ///< Logical sensor name.
        SensorKind kind{SensorKind::Other}; ///< Sensor value category.
        std::vector<double> values;         ///< Raw SI values in model order.
};

/// Immutable snapshot returned by the plant at one simulation time.
struct RobotPlantState {
        double simulationTime{};        ///< MuJoCo simulation time in seconds.
        RigidBodyState base;            ///< Selected base or pelvis state.
        std::vector<JointState> joints; ///< All mapped joint states.
        std::vector<RigidBodyState> bodies; ///< All mapped rigid-body states.
        std::vector<ContactState> contacts; ///< Contacts active in this step.
        std::vector<SensorState> sensors;   ///< Sensor ground-truth readings.
};

/// Finite constraint request, also used to model a joint lock.
struct ConstraintState {
        bool enabled{};          ///< Whether the constraint is active.
        double targetPosition{}; ///< Requested locked position in rad or m.
        double stiffness{};      ///< Proportional stiffness in SI units.
        double damping{};        ///< Velocity damping in SI units.
        double maxTorque{}; ///< Absolute effort limit; zero means unlimited.
};

/// Linear force queued for application at a world-space point.
struct LinearForceApplication {
        BodyId body; ///< Body receiving the force.
        Vec3 point;  ///< World-space application point in metres.
        Vec3 force;  ///< World-space force in newtons.
};

/// Independent update rates for deterministic multi-rate simulation.
struct SimulationTiming {
        double physicsDt{0.001};        ///< Physics step duration in seconds.
        double actuatorDt{0.001};       ///< Actuator update period in seconds.
        double controlDt{0.01};         ///< Control update period in seconds.
        double viewerRefreshRate{60.0}; ///< Viewer target rate in hertz.
};

} // namespace ai::simulation
