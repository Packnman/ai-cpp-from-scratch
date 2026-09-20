#include "simulation/MuJoCoContactManager.hpp"

#include <array>
#include <string>

namespace ai::simulation {
namespace {

// Resolves a contact body name without exposing a numeric MuJoCo ID.
BodyId bodyName(const mjModel *model, int bodyId) {
    if (const char *name = mj_id2name(model, mjOBJ_BODY, bodyId); name && *name)
        return name;
    return "body_" + std::to_string(bodyId);
}

} // namespace

std::vector<ContactState>
MuJoCoContactManager::contacts(const MuJoCoModel &owner) const {
    const auto *model = owner.model();
    const auto *data = owner.data();
    std::vector<ContactState> result;
    result.reserve(static_cast<std::size_t>(data->ncon));
    for (int index = 0; index < data->ncon; ++index) {
        const auto &contact = data->contact[index];
        std::array<mjtNum, 6> contactForce{};
        // MuJoCo reports wrench components in the contact frame.
        mj_contactForce(model, data, index, contactForce.data());
        const int bodyA = model->geom_bodyid[contact.geom[0]];
        const int bodyB = model->geom_bodyid[contact.geom[1]];
        const Vec3 normal{contact.frame[0], contact.frame[1], contact.frame[2]};
        const Vec3 tangent1{contact.frame[3], contact.frame[4],
                            contact.frame[5]};
        const Vec3 tangent2{contact.frame[6], contact.frame[7],
                            contact.frame[8]};
        // Rotate normal and tangent components into a world-frame force.
        const Vec3 force{
            normal.x * contactForce[0] + tangent1.x * contactForce[1] +
                tangent2.x * contactForce[2],
            normal.y * contactForce[0] + tangent1.y * contactForce[1] +
                tangent2.y * contactForce[2],
            normal.z * contactForce[0] + tangent1.z * contactForce[1] +
                tangent2.z * contactForce[2]};
        result.push_back({bodyName(model, bodyA),
                          bodyName(model, bodyB),
                          {contact.pos[0], contact.pos[1], contact.pos[2]},
                          normal,
                          force,
                          contactForce[0],
                          contact.dist});
    }
    return result;
}

} // namespace ai::simulation
