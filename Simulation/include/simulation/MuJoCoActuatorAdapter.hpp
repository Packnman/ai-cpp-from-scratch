#pragma once

#include "actuator/common/ActuatorTypes.hpp"
#include "simulation/IPlant.hpp"

#include <optional>
#include <unordered_map>

namespace ai::simulation {

/// Mapping from one Actuator ID to one or more logical Plant targets.
struct ActuatorPlantBinding {
        ai::actuator::ActuatorId actuatorId; ///< Source Actuator identifier.
        JointId jointId; ///< Optional joint receiving rotary/passive effort.
        std::optional<BodyId>
            linearForceBody;   ///< Optional body receiving linear force.
        Vec3 applicationPoint; ///< World-space force application point.
        Vec3 forceDirection; ///< Direction normalized before force application.
        std::optional<ConstraintId>
            constraintId; ///< Optional joint/equality lock target.
};

/// Translates engine-independent Actuator output into IPlant commands.
class MuJoCoActuatorAdapter {
    public:
        /// Registers one immutable Actuator-to-Plant binding.
        void bind(ActuatorPlantBinding);
        /// Applies the latest output from one Actuator to the supplied plant.
        void apply(const ai::actuator::ActuatorId &,
                   const ai::actuator::PlantOutput &, IPlant &) const;

    private:
        std::unordered_map<ai::actuator::ActuatorId, ActuatorPlantBinding>
            _bindings; ///< Bindings indexed by source Actuator ID.
};

} // namespace ai::simulation
