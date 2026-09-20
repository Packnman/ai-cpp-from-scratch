#pragma once

#include "simulation/PlantTypes.hpp"

namespace ai::simulation {

/// Engine-independent boundary for a robot rigid-body plant.
class IPlant {
    public:
        virtual ~IPlant() = default;

        /// Loads resources and creates the initial plant state.
        virtual void initialize() = 0;
        /// Restores model defaults and clears pending input.
        virtual void reset() = 0;
        /// Advances deterministic simulation time by the supplied duration.
        virtual void step(double dtSeconds) = 0;
        /// Queues generalized effort for one logical joint.
        virtual void applyJointTorque(const JointId &, double torque) = 0;
        /// Queues a world-frame force at a world-frame point on a body.
        virtual void applyLinearForce(const BodyId &, const Vec3 &point,
                                      const Vec3 &force) = 0;
        /// Updates an equality constraint or finite joint-lock request.
        virtual void setConstraintState(const ConstraintId &,
                                        const ConstraintState &) = 0;
        /// Returns a value-copy snapshot of the current plant state.
        [[nodiscard]] virtual RobotPlantState getState() const = 0;
};

} // namespace ai::simulation
