#include "simulation/MuJoCoPlant.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ai::simulation {

MuJoCoPlant::MuJoCoPlant(std::filesystem::path modelPath)
    : _modelPath(std::move(modelPath)) {}

void MuJoCoPlant::initialize() {
    _model = std::make_unique<MuJoCoModel>(_modelPath);
    _jointEfforts.clear();
    _linearForces.clear();
    _constraints.clear();
}

void MuJoCoPlant::reset() {
    requireInitialized();
    _model->reset();
    _jointEfforts.clear();
    _linearForces.clear();
    _constraints.clear();
}

void MuJoCoPlant::step(double dtSeconds) {
    requireInitialized();
    if (!std::isfinite(dtSeconds) || dtSeconds <= 0.0)
        throw std::invalid_argument("MuJoCo step dt must be positive");
    auto *model = _model->model();
    auto *data = _model->data();
    model->opt.timestep = dtSeconds;
    // Clear MuJoCo input arrays so one-shot efforts never leak into later
    // steps.
    mju_zero(data->qfrc_applied, model->nv);
    mju_zero(data->xfrc_applied, 6 * model->nbody);
    applyPendingEfforts();
    // mj_step advances MuJoCo time; no wall-clock value affects physics.
    mj_step(model, data);
    _jointEfforts.clear();
    _linearForces.clear();
}

void MuJoCoPlant::applyJointTorque(const JointId &joint, double torque) {
    requireInitialized();
    if (!std::isfinite(torque))
        throw std::invalid_argument("joint torque must be finite");
    static_cast<void>(_model->jointBinding(joint));
    _jointEfforts.push_back({joint, torque});
}

void MuJoCoPlant::applyLinearForce(const BodyId &body, const Vec3 &point,
                                   const Vec3 &force) {
    requireInitialized();
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z) || !std::isfinite(force.x) ||
        !std::isfinite(force.y) || !std::isfinite(force.z))
        throw std::invalid_argument("linear force values must be finite");
    static_cast<void>(_model->bodyBinding(body));
    _linearForces.push_back({body, point, force});
}

void MuJoCoPlant::setConstraintState(const ConstraintId &id,
                                     const ConstraintState &state) {
    requireInitialized();
    if (!std::isfinite(state.targetPosition) ||
        !std::isfinite(state.stiffness) || !std::isfinite(state.damping) ||
        !std::isfinite(state.maxTorque) || state.stiffness < 0.0 ||
        state.damping < 0.0 || state.maxTorque < 0.0)
        throw std::invalid_argument(
            "constraint parameters must be finite and non-negative");
    if (_model->equalityId(id) < 0)
        static_cast<void>(_model->jointBinding(id));
    _constraints[id] = state;
}

RobotPlantState MuJoCoPlant::getState() const {
    requireInitialized();
    return _stateAdapter.convert(*_model);
}

bool MuJoCoPlant::initialized() const noexcept {
    return static_cast<bool>(_model);
}

MuJoCoModel &MuJoCoPlant::model() {
    requireInitialized();
    return *_model;
}

const MuJoCoModel &MuJoCoPlant::model() const {
    requireInitialized();
    return *_model;
}

void MuJoCoPlant::requireInitialized() const {
    if (!_model)
        throw std::logic_error("MuJoCo plant is not initialized");
}

void MuJoCoPlant::applyPendingEfforts() {
    auto *model = _model->model();
    auto *data = _model->data();
    for (const auto &effort : _jointEfforts) {
        const auto &binding = _model->jointBinding(effort.joint);
        data->qfrc_applied[binding.dofAddress] += effort.torque;
    }
    for (const auto &application : _linearForces) {
        const auto &binding = _model->bodyBinding(application.body);
        const std::array<mjtNum, 3> force{
            application.force.x, application.force.y, application.force.z};
        const std::array<mjtNum, 3> torque{};
        const std::array<mjtNum, 3> point{
            application.point.x, application.point.y, application.point.z};
        // Convert the requested world-point force into generalized coordinates.
        mj_applyFT(model, data, force.data(), torque.data(), point.data(),
                   binding.bodyId, data->qfrc_applied);
    }
    for (const auto &[id, constraint] : _constraints) {
        const int equality = _model->equalityId(id);
        // Prefer native equality constraints when the MJCF defines one.
        if (equality >= 0) {
            data->eq_active[equality] = constraint.enabled ? 1 : 0;
            continue;
        }
        if (!constraint.enabled)
            continue;
        const auto &binding = _model->jointBinding(id);
        // Otherwise model a finite lock as a bounded PD joint effort.
        double torque =
            constraint.stiffness *
                (constraint.targetPosition - data->qpos[binding.qposAddress]) -
            constraint.damping * data->qvel[binding.dofAddress];
        if (constraint.maxTorque > 0.0)
            torque =
                std::clamp(torque, -constraint.maxTorque, constraint.maxTorque);
        data->qfrc_applied[binding.dofAddress] += torque;
    }
}

} // namespace ai::simulation
