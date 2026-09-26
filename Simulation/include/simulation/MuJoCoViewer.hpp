#pragma once

#include "simulation/MuJoCoPlant.hpp"
#include "simulation/SimulationManager.hpp"

#include <filesystem>
#include <functional>
#include <memory>

namespace ai::simulation {

/// Optional GLFW viewer that renders a MuJoCoPlant without owning physics.
class MuJoCoViewer {
    public:
        /// Optional action invoked by Space instead of pause/resume.
        using ReplayCallback = std::function<void()>;
        /// Creates an empty viewer implementation.
        MuJoCoViewer();
        /// Releases GLFW and MuJoCo visualization resources.
        ~MuJoCoViewer();
        MuJoCoViewer(const MuJoCoViewer &) = delete;
        MuJoCoViewer &operator=(const MuJoCoViewer &) = delete;
        /// Runs the interactive render loop until the window closes.
        void run(MuJoCoPlant &, SimulationManager &,
                 ReplayCallback replay = {});
        /// Renders the current Plant pose to a binary PPM image.
        void saveFrame(MuJoCoPlant &, const std::filesystem::path &,
                       int width = 1280, int height = 720);

    private:
        class Impl; ///< Hides GLFW types from the public header.
        std::unique_ptr<Impl> _impl; ///< Owned viewer implementation.
};

} // namespace ai::simulation
