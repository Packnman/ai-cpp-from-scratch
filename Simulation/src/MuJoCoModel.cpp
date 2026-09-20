#include "simulation/MuJoCoModel.hpp"

#include <stdexcept>
#include <string>

namespace ai::simulation {
namespace {

// Returns a stable logical name, including for unnamed MuJoCo objects.
std::string objectName(const mjModel *model, mjtObj type, int id,
                       const char *prefix) {
    if (const char *name = mj_id2name(model, type, id); name && *name)
        return name;
    return std::string{prefix} + '_' + std::to_string(id);
}

// Returns qpos and generalized-velocity widths for each joint kind.
std::pair<int, int> jointWidths(int type) {
    switch (type) {
    case mjJNT_FREE:
        return {7, 6};
    case mjJNT_BALL:
        return {4, 3};
    case mjJNT_SLIDE:
    case mjJNT_HINGE:
        return {1, 1};
    default:
        throw std::runtime_error("unsupported MuJoCo joint type");
    }
}

// Maps MuJoCo sensor enums to the engine-independent category.
SensorKind sensorKind(int type) {
    switch (type) {
    case mjSENS_JOINTPOS:
        return SensorKind::JointPosition;
    case mjSENS_JOINTVEL:
        return SensorKind::JointVelocity;
    case mjSENS_FRAMEQUAT:
        return SensorKind::Orientation;
    case mjSENS_GYRO:
    case mjSENS_FRAMEANGVEL:
        return SensorKind::AngularVelocity;
    case mjSENS_TOUCH:
        return SensorKind::Contact;
    default:
        return SensorKind::Other;
    }
}

} // namespace

MuJoCoModel::MuJoCoModel(std::filesystem::path modelPath)
    : _path(std::move(modelPath)) {
    char error[1024]{};
    // Compile MJCF first, then allocate runtime data owned by the model.
    _model = mj_loadXML(_path.string().c_str(), nullptr, error, sizeof(error));
    if (!_model)
        throw std::runtime_error("MuJoCo model load failed: " +
                                 std::string{error});
    _data = mj_makeData(_model);
    if (!_data) {
        mj_deleteModel(_model);
        _model = nullptr;
        throw std::runtime_error("MuJoCo data allocation failed");
    }
    // Cache every ID/address once so no subsystem hard-codes MuJoCo indices.
    buildMappings();
    mj_forward(_model, _data);
}

MuJoCoModel::~MuJoCoModel() {
    if (_data)
        mj_deleteData(_data);
    if (_model)
        mj_deleteModel(_model);
}

void MuJoCoModel::reset() {
    mj_resetData(_model, _data);
    mj_forward(_model, _data);
}

void MuJoCoModel::buildMappings() {
    for (int id = 0; id < _model->njnt; ++id) {
        auto [qposWidth, dofWidth] = jointWidths(_model->jnt_type[id]);
        auto logicalId = objectName(_model, mjOBJ_JOINT, id, "joint");
        _jointBindings.emplace(
            logicalId,
            MuJoCoJointBinding{logicalId, id, _model->jnt_qposadr[id],
                               _model->jnt_dofadr[id], qposWidth, dofWidth});
    }
    for (int id = 0; id < _model->nbody; ++id) {
        auto logicalId = objectName(_model, mjOBJ_BODY, id, "body");
        _bodyBindings.emplace(logicalId, MuJoCoBodyBinding{logicalId, id});
    }
    for (int id = 0; id < _model->nsite; ++id) {
        auto logicalId = objectName(_model, mjOBJ_SITE, id, "site");
        _siteBindings.emplace(logicalId, MuJoCoSiteBinding{logicalId, id});
    }
    for (int id = 0; id < _model->nsensor; ++id) {
        auto logicalId = objectName(_model, mjOBJ_SENSOR, id, "sensor");
        _sensorBindings.emplace(
            logicalId,
            MuJoCoSensorBinding{logicalId, id, _model->sensor_adr[id],
                                _model->sensor_dim[id],
                                sensorKind(_model->sensor_type[id])});
    }
    for (int id = 0; id < _model->neq; ++id) {
        auto logicalId = objectName(_model, mjOBJ_EQUALITY, id, "constraint");
        _equalityBindings.emplace(logicalId, id);
    }
}

const MuJoCoJointBinding &MuJoCoModel::jointBinding(const JointId &id) const {
    const auto found = _jointBindings.find(id);
    if (found == _jointBindings.end())
        throw std::out_of_range("unknown MuJoCo joint: " + id);
    return found->second;
}

const MuJoCoBodyBinding &MuJoCoModel::bodyBinding(const BodyId &id) const {
    const auto found = _bodyBindings.find(id);
    if (found == _bodyBindings.end())
        throw std::out_of_range("unknown MuJoCo body: " + id);
    return found->second;
}

const MuJoCoSiteBinding &MuJoCoModel::siteBinding(const SiteId &id) const {
    const auto found = _siteBindings.find(id);
    if (found == _siteBindings.end())
        throw std::out_of_range("unknown MuJoCo site: " + id);
    return found->second;
}

int MuJoCoModel::equalityId(const ConstraintId &id) const {
    const auto found = _equalityBindings.find(id);
    return found == _equalityBindings.end() ? -1 : found->second;
}

} // namespace ai::simulation
