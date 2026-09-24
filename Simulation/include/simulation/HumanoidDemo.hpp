#pragma once

#include "actuator/driver/sim/SimActuatorDriver.hpp"
#include "actuator/manager/ActuatorManager.hpp"
#include "brain/constraint/ConstraintManager.hpp"
#include "brain/execution/ExecutionManager.hpp"
#include "brain/planning/Planner.hpp"
#include "brain/policy/PolicyManager.hpp"
#include "brain/preprocess/ContextRecognizer.hpp"
#include "brain/preprocess/Preprocessor.hpp"
#include "simulation/MuJoCoActuatorAdapter.hpp"
#include "simulation/MuJoCoPlant.hpp"
#include "simulation/SimulationManager.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace ai::simulation::demo {

/// Joint-space target accepted from a Brain action.
struct JointTarget {
        brain::ActionId actionId{}; ///< Brain action that produced the target.
        JointId jointId;            ///< Logical MuJoCo joint name.
        double position{};          ///< Desired joint position in radians.
};

/// Summary of one text-to-Control Brain cycle.
struct BrainCommandResult {
        bool recognized{}; ///< Whether text produced a supported Goal.
        bool planned{};    ///< Whether the Goal produced a valid ActionPlan.
        bool dispatched{}; ///< Whether Control accepted the resulting Action.
        std::string goalType; ///< Recognized stable Goal type.
};

/// Deterministic recognizer for the first visible humanoid demonstration.
class HumanoidCommandRecognizer final : public brain::IContextRecognizer {
    public:
        /// Recognizes right-arm raise/lower commands in Japanese or English.
        brain::ContextRecognitionResult
        recognize(const std::string &text, brain::TimePoint timestamp) override;
};

/// Brain Control port that publishes validated joint-position targets.
class JointTargetDispatcher final : public brain::IControlDispatcher {
    public:
        /// Accepts a set_joint_target Action and acknowledges it immediately.
        std::optional<brain::ActionResult>
        dispatch(const brain::ActionCommand &command) override;
        /// Removes a target when its originating action is cancelled.
        void cancel(brain::ActionId actionId) override;
        /// Returns the latest target as a thread-safe value copy.
        [[nodiscard]] std::optional<JointTarget> target() const;

    private:
        mutable std::mutex _mutex; ///< Protects viewer/control thread access.
        std::optional<JointTarget> _target; ///< Most recently accepted target.
};

/// Minimal Brain pipeline used to turn visible-demo text into Control action.
class HumanoidBrainBridge {
    public:
        /// Builds fixed arm policies around the injected Control dispatcher.
        explicit HumanoidBrainBridge(
            std::shared_ptr<JointTargetDispatcher> dispatcher);
        /// Runs recognition, policy selection, planning and dispatch once.
        BrainCommandResult issue(const std::string &text,
                                 brain::TimePoint now = brain::steady_now());

    private:
        brain::Preprocessor _preprocessor; ///< Text-to-semantic Brain stage.
        brain::WorldState _world;          ///< Snapshot supplied to planning.
        brain::ConstraintManager _constraints; ///< Active safety constraints.
        brain::PolicyManager _policies;        ///< Fixed Phase-1 skills.
        brain::RuleBasedPlanner _planner;      ///< Deterministic plan builder.
        std::shared_ptr<JointTargetDispatcher>
            _dispatcher;                    ///< Injected Control boundary.
        brain::ExecutionManager _execution; ///< Action dispatch lifecycle.
        brain::GoalId _nextGoalId{1};       ///< Stable local Goal sequence.
};

/// End-to-end Phase-1 rig: Brain -> Control -> Actuator -> MuJoCo.
class HumanoidDemo {
    public:
        /// Creates the supported humanoid plant and right-elbow actuator.
        explicit HumanoidDemo(std::filesystem::path modelPath);
        /// Loads the MuJoCo model and prepares deterministic callbacks.
        void initialize();
        /// Sends one natural-language instruction through the Brain bridge.
        BrainCommandResult command(const std::string &text);
        /// Advances a fixed number of headless physics steps.
        void runSteps(std::size_t count);
        /// Returns the current right-elbow joint position.
        [[nodiscard]] double rightElbowPosition() const;
        /// Returns the current Control target, when one has been issued.
        [[nodiscard]] std::optional<JointTarget> target() const;
        /// Exposes the scheduler to the optional viewer.
        [[nodiscard]] SimulationManager &manager() noexcept { return _manager; }
        /// Exposes the concrete Plant to the optional viewer.
        [[nodiscard]] MuJoCoPlant &plant() noexcept { return *_plant; }

    private:
        std::shared_ptr<JointTargetDispatcher>
            _dispatcher;            ///< Brain-to-Control target mailbox.
        HumanoidBrainBridge _brain; ///< Deterministic Phase-1 Brain pipeline.
        MuJoCoPlant *_plant{};      ///< Non-owning view of manager-owned Plant.
        SimulationManager _manager; ///< Multi-rate physics scheduler.
        actuator::ActuatorManager _actuators; ///< Right-arm drive registry.
        actuator::sim::SimActuatorDriver
            *_rightElbowDriver{};       ///< Non-owning registered driver view.
        MuJoCoActuatorAdapter _adapter; ///< Mechanical output to Plant mapping.
        actuator::CommandId _nextCommandId{1}; ///< Control command sequence.
};

} // namespace ai::simulation::demo
