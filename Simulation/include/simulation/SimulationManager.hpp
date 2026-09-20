#pragma once

#include "simulation/IPlant.hpp"

#include <cstddef>
#include <functional>
#include <memory>

namespace ai::simulation {

/// Owns a plant and schedules control, actuator, physics and logging cycles.
class SimulationManager {
    public:
        /// Callback invoked with its cycle duration and pre-step plant state.
        using CycleCallback =
            std::function<void(double, const RobotPlantState &, IPlant &)>;
        /// Callback invoked with the completed post-step state.
        using LogCallback = std::function<void(const RobotPlantState &)>;

        /// Creates a manager for one plant and a fixed timing configuration.
        SimulationManager(std::unique_ptr<IPlant>, SimulationTiming);
        /// Initializes the plant and schedules callbacks at simulation time
        /// zero.
        void initialize();
        /// Resets the plant and all cycle accumulators.
        void reset();
        /// Executes callbacks that are due and advances one physics step.
        void step();
        /// Executes a fixed number of deterministic physics steps.
        void runSteps(std::size_t count);
        /// Installs the control-cycle callback.
        void setControlCallback(CycleCallback callback);
        /// Installs the actuator-cycle callback.
        void setActuatorCallback(CycleCallback callback);
        /// Installs the post-step logging callback.
        void setLogCallback(LogCallback callback);
        /// Returns the owned plant for adapter access.
        [[nodiscard]] IPlant &plant() noexcept { return *_plant; }
        /// Returns the owned plant for read-only access.
        [[nodiscard]] const IPlant &plant() const noexcept { return *_plant; }
        /// Returns the active multi-rate timing configuration.
        [[nodiscard]] const SimulationTiming &timing() const noexcept {
            return _timing;
        }
        /// Reports whether physics advancement is paused.
        [[nodiscard]] bool paused() const noexcept { return _paused; }
        /// Pauses or resumes physics advancement without changing state.
        void setPaused(bool paused) noexcept { _paused = paused; }

    private:
        std::unique_ptr<IPlant> _plant;  ///< Exclusively owned plant backend.
        SimulationTiming _timing;        ///< Fixed update periods.
        CycleCallback _controlCallback;  ///< Optional control-cycle handler.
        CycleCallback _actuatorCallback; ///< Optional actuator-cycle handler.
        LogCallback _logCallback;        ///< Optional post-step logger.
        double _controlAccumulator{}; ///< Simulated time toward control cycle.
        double
            _actuatorAccumulator{}; ///< Simulated time toward actuator cycle.
        bool _initialized{};        ///< True after successful initialization.
        bool _paused{};             ///< True while step requests are ignored.
};

} // namespace ai::simulation
