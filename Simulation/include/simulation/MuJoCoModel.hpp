#pragma once

#include "simulation/PlantTypes.hpp"

#include <mujoco/mujoco.h>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai::simulation {

/// Logical-to-MuJoCo mapping and array widths for one joint.
struct MuJoCoJointBinding {
        JointId logicalId;   ///< Stable logical joint name.
        int jointId{-1};     ///< MuJoCo joint object ID.
        int qposAddress{-1}; ///< First index in mjData::qpos.
        int dofAddress{-1};  ///< First index in velocity/force arrays.
        int qposWidth{};     ///< Number of qpos elements used by the joint.
        int dofWidth{};      ///< Number of generalized velocity elements.
};

/// Logical-to-MuJoCo mapping for one rigid body.
struct MuJoCoBodyBinding {
        BodyId logicalId; ///< Stable logical body name.
        int bodyId{-1};   ///< MuJoCo body object ID.
};

/// Logical-to-MuJoCo mapping for one site.
struct MuJoCoSiteBinding {
        SiteId logicalId; ///< Stable logical site name.
        int siteId{-1};   ///< MuJoCo site object ID.
};

/// Logical-to-MuJoCo mapping and data slice for one sensor.
struct MuJoCoSensorBinding {
        SensorId logicalId;  ///< Stable logical sensor name.
        int sensorId{-1};    ///< MuJoCo sensor object ID.
        int dataAddress{-1}; ///< First index in mjData::sensordata.
        int dimension{};     ///< Number of values produced by the sensor.
        SensorKind kind{SensorKind::Other}; ///< Semantic sensor category.
};

/// RAII owner of a compiled MuJoCo model, runtime data and ID mappings.
class MuJoCoModel {
    public:
        /// Loads and compiles an MJCF model from disk.
        explicit MuJoCoModel(std::filesystem::path modelPath);
        /// Releases mjData before its associated mjModel.
        ~MuJoCoModel();
        MuJoCoModel(const MuJoCoModel &) = delete;
        MuJoCoModel &operator=(const MuJoCoModel &) = delete;
        MuJoCoModel(MuJoCoModel &&) = delete;
        MuJoCoModel &operator=(MuJoCoModel &&) = delete;

        /// Restores MuJoCo runtime data to model defaults.
        void reset();
        /// Returns the mutable model for Simulation-internal operations.
        [[nodiscard]] mjModel *model() noexcept { return _model; }
        /// Returns the read-only compiled model.
        [[nodiscard]] const mjModel *model() const noexcept { return _model; }
        /// Returns mutable runtime data.
        [[nodiscard]] mjData *data() noexcept { return _data; }
        /// Returns read-only runtime data.
        [[nodiscard]] const mjData *data() const noexcept { return _data; }
        /// Returns the MJCF source path.
        [[nodiscard]] const std::filesystem::path &path() const noexcept {
            return _path;
        }
        /// Resolves a logical joint or throws when it is unknown.
        [[nodiscard]] const MuJoCoJointBinding &
        jointBinding(const JointId &) const;
        /// Resolves a logical body or throws when it is unknown.
        [[nodiscard]] const MuJoCoBodyBinding &
        bodyBinding(const BodyId &) const;
        /// Resolves a logical site or throws when it is unknown.
        [[nodiscard]] const MuJoCoSiteBinding &
        siteBinding(const SiteId &) const;
        /// Returns an equality ID, or -1 when the logical ID is absent.
        [[nodiscard]] int equalityId(const ConstraintId &) const;
        /// Returns every mapped joint indexed by logical ID.
        [[nodiscard]] const auto &jointBindings() const noexcept {
            return _jointBindings;
        }
        /// Returns every mapped body indexed by logical ID.
        [[nodiscard]] const auto &bodyBindings() const noexcept {
            return _bodyBindings;
        }
        /// Returns every mapped site indexed by logical ID.
        [[nodiscard]] const auto &siteBindings() const noexcept {
            return _siteBindings;
        }
        /// Returns every mapped sensor indexed by logical ID.
        [[nodiscard]] const auto &sensorBindings() const noexcept {
            return _sensorBindings;
        }

    private:
        /// Builds all logical ID tables once after model compilation.
        void buildMappings();
        std::filesystem::path _path; ///< MJCF file used to construct the model.
        mjModel *_model{};           ///< Compiled model owned by this instance.
        mjData *_data{};             ///< Runtime data owned by this instance.
        std::unordered_map<JointId, MuJoCoJointBinding>
            _jointBindings; ///< Joint lookup table.
        std::unordered_map<BodyId, MuJoCoBodyBinding>
            _bodyBindings; ///< Body lookup table.
        std::unordered_map<SiteId, MuJoCoSiteBinding>
            _siteBindings; ///< Site lookup table.
        std::unordered_map<SensorId, MuJoCoSensorBinding>
            _sensorBindings; ///< Sensor lookup table.
        std::unordered_map<ConstraintId, int>
            _equalityBindings; ///< Equality lookup table.
};

} // namespace ai::simulation
