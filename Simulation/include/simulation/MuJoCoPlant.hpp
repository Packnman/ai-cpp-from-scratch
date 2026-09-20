#pragma once

#include "simulation/IPlant.hpp"
#include "simulation/MuJoCoModel.hpp"
#include "simulation/MuJoCoStateAdapter.hpp"

#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ai::simulation {

/// MuJoCo-backed implementation of the engine-independent robot plant.
class MuJoCoPlant final : public IPlant {
    public:
        /// Stores the MJCF path; resources are loaded by initialize().
        explicit MuJoCoPlant(std::filesystem::path modelPath);
        ~MuJoCoPlant() override = default;
        /// Loads the MJCF and creates a fresh runtime state.
        void initialize() override;
        /// Restores model defaults and clears all queued commands.
        void reset() override;
        /// Applies queued commands and executes one MuJoCo step.
        void step(double dtSeconds) override;
        /// Queues joint effort for the next physics step.
        void applyJointTorque(const JointId &, double torque) override;
        /// Queues a force application for the next physics step.
        void applyLinearForce(const BodyId &, const Vec3 &point,
                              const Vec3 &force) override;
        /// Stores an equality or finite-lock constraint command.
        void setConstraintState(const ConstraintId &,
                                const ConstraintState &) override;
        /// Converts MuJoCo runtime arrays into engine-independent state.
        [[nodiscard]] RobotPlantState getState() const override;
        /// Reports whether the MuJoCo owner has been created.
        [[nodiscard]] bool initialized() const noexcept;
        /// Returns the Simulation-internal model owner.
        [[nodiscard]] MuJoCoModel &model();
        /// Returns the read-only Simulation-internal model owner.
        [[nodiscard]] const MuJoCoModel &model() const;

    private:
        /// One joint effort queued until the next call to step().
        struct JointEffort {
                JointId joint;   ///< Logical target joint.
                double torque{}; ///< Generalized effort in Nm or N.
        };

        /// Throws when an operation is requested before initialize().
        void requireInitialized() const;
        /// Writes queued efforts and constraints into MuJoCo runtime arrays.
        void applyPendingEfforts();
        std::filesystem::path
            _modelPath; ///< MJCF source selected at construction.
        std::unique_ptr<MuJoCoModel> _model; ///< Owned MuJoCo model and data.
        MuJoCoStateAdapter _stateAdapter;    ///< Converts engine state to DTOs.
        std::vector<JointEffort>
            _jointEfforts; ///< One-step joint effort queue.
        std::vector<LinearForceApplication>
            _linearForces; ///< One-step body-force queue.
        std::unordered_map<ConstraintId, ConstraintState>
            _constraints; ///< Persistent constraint commands.
};

} // namespace ai::simulation
