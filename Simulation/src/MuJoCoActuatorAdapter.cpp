#include "simulation/MuJoCoActuatorAdapter.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ai::simulation {

void MuJoCoActuatorAdapter::bind(ActuatorPlantBinding binding) {
    if (binding.actuatorId.empty() ||
        (binding.jointId.empty() && !binding.linearForceBody &&
         !binding.constraintId))
        throw std::invalid_argument("actuator binding has no plant target");
    const auto id = binding.actuatorId;
    if (!_bindings.emplace(id, std::move(binding)).second)
        throw std::invalid_argument("duplicate actuator plant binding");
}

void MuJoCoActuatorAdapter::apply(const ai::actuator::ActuatorId &actuatorId,
                                  const ai::actuator::PlantOutput &output,
                                  IPlant &plant) const {
    const auto found = _bindings.find(actuatorId);
    if (found == _bindings.end())
        throw std::out_of_range("unknown actuator plant binding");
    const auto &binding = found->second;
    // Active and passive rotary efforts share one generalized Plant input.
    if (!binding.jointId.empty())
        plant.applyJointTorque(
            binding.jointId, output.jointTorque + output.passiveReactionTorque);
    if (binding.linearForceBody && output.linearForce != 0.0) {
        // Bindings store direction only; normalize it before scaling by force.
        const double norm =
            std::sqrt(binding.forceDirection.x * binding.forceDirection.x +
                      binding.forceDirection.y * binding.forceDirection.y +
                      binding.forceDirection.z * binding.forceDirection.z);
        if (norm == 0.0)
            throw std::invalid_argument("linear force direction is zero");
        plant.applyLinearForce(
            *binding.linearForceBody, binding.applicationPoint,
            {binding.forceDirection.x / norm * output.linearForce,
             binding.forceDirection.y / norm * output.linearForce,
             binding.forceDirection.z / norm * output.linearForce});
    }
    // Locks remain finite constraints instead of ideal infinite-stiffness
    // joints.
    if (binding.constraintId)
        plant.setConstraintState(
            *binding.constraintId,
            {output.constraintActive, output.constraintPosition,
             output.constraintStiffness, 0.0, output.constraintTorqueLimit});
}

} // namespace ai::simulation
