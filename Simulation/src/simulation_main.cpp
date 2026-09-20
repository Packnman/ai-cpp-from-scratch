#include "simulation/MuJoCoPlant.hpp"
#include "simulation/SimulationManager.hpp"

#ifdef AI_CPP_MUJOCO_VIEWER
#include "simulation/MuJoCoViewer.hpp"
#endif

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

/// Command-line options for one simulation run.
struct Options {
        std::filesystem::path model{
            AI_CPP_SIMULATION_DEFAULT_MODEL}; ///< MJCF path.
        std::size_t steps{1'000};             ///< Headless physics-step count.
        bool viewer{}; ///< Enables the interactive viewer.
};

/// Parses supported CLI flags and rejects incomplete options.
Options parse(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--viewer")
            options.viewer = true;
        else if (argument == "--headless")
            options.viewer = false;
        else if (argument == "--model" && i + 1 < argc)
            options.model = argv[++i];
        else if (argument == "--steps" && i + 1 < argc)
            options.steps = std::stoull(argv[++i]);
        else
            throw std::invalid_argument(
                "usage: simulation_cli [--headless|--viewer] "
                "[--model path] [--steps count]");
    }
    return options;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = parse(argc, argv);
        auto plant =
            std::make_unique<ai::simulation::MuJoCoPlant>(options.model);
        auto *mujocoPlant = plant.get();
        ai::simulation::SimulationManager manager{std::move(plant),
                                                  {0.001, 0.001, 0.01, 60.0}};
        manager.initialize();
        if (options.viewer) {
#ifdef AI_CPP_MUJOCO_VIEWER
            ai::simulation::MuJoCoViewer viewer;
            viewer.run(*mujocoPlant, manager);
#else
            throw std::runtime_error(
                "viewer support was not enabled at build time");
#endif
        } else {
            manager.runSteps(options.steps);
            const auto state = manager.plant().getState();
            std::cout << "simulation_time=" << state.simulationTime
                      << " joints=" << state.joints.size()
                      << " contacts=" << state.contacts.size() << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "simulation error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
