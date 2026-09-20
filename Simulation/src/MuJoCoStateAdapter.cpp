#include "simulation/MuJoCoStateAdapter.hpp"

#include <array>

namespace ai::simulation {
namespace {

// Copies one body pose and world-frame spatial velocity.
RigidBodyState bodyState(const MuJoCoModel &owner,
                         const MuJoCoBodyBinding &binding) {
    const auto *model = owner.model();
    const auto *data = owner.data();
    const int position = 3 * binding.bodyId;
    const int orientation = 4 * binding.bodyId;
    std::array<mjtNum, 6> velocity{};
    // MuJoCo spatial velocity stores angular components before linear ones.
    mj_objectVelocity(model, data, mjOBJ_BODY, binding.bodyId, velocity.data(),
                      0);
    return {binding.logicalId,
            {data->xpos[position], data->xpos[position + 1],
             data->xpos[position + 2]},
            {data->xquat[orientation], data->xquat[orientation + 1],
             data->xquat[orientation + 2], data->xquat[orientation + 3]},
            {velocity[3], velocity[4], velocity[5]},
            {velocity[0], velocity[1], velocity[2]}};
}

} // namespace

RobotPlantState MuJoCoStateAdapter::convert(const MuJoCoModel &owner) const {
    const auto *data = owner.data();
    RobotPlantState state;
    state.simulationTime = data->time;
    // Resolve addresses through cached bindings rather than fixed qpos indices.
    state.joints.reserve(owner.jointBindings().size());
    for (const auto &[id, binding] : owner.jointBindings()) {
        state.joints.push_back({id, data->qpos[binding.qposAddress],
                                data->qvel[binding.dofAddress],
                                data->qacc[binding.dofAddress],
                                data->qfrc_applied[binding.dofAddress]});
    }
    state.bodies.reserve(owner.bodyBindings().size());
    for (const auto &[id, binding] : owner.bodyBindings())
        state.bodies.push_back(bodyState(owner, binding));

    // Prefer the humanoid pelvis, then the test-model base, as base state.
    const auto pelvis = owner.bodyBindings().find("pelvis");
    const auto base = pelvis != owner.bodyBindings().end()
                          ? pelvis
                          : owner.bodyBindings().find("base");
    if (base != owner.bodyBindings().end())
        state.base = bodyState(owner, base->second);
    else if (!state.bodies.empty())
        state.base = state.bodies.front();

    state.contacts = _contacts.contacts(owner);
    // Copy variable-width sensor slices into value-owned DTO storage.
    state.sensors.reserve(owner.sensorBindings().size());
    for (const auto &[id, binding] : owner.sensorBindings()) {
        SensorState sensor{id, binding.kind, {}};
        sensor.values.reserve(static_cast<std::size_t>(binding.dimension));
        for (int offset = 0; offset < binding.dimension; ++offset)
            sensor.values.push_back(
                data->sensordata[binding.dataAddress + offset]);
        state.sensors.push_back(std::move(sensor));
    }
    return state;
}

} // namespace ai::simulation
