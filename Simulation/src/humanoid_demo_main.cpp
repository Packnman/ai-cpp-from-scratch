#include "simulation/HumanoidDemo.hpp"

#ifdef AI_CPP_MUJOCO_VIEWER
#include "simulation/MuJoCoViewer.hpp"
#endif

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

/// Command-line configuration for the visible Phase-1 demonstration.
struct Options {
        std::filesystem::path model{AI_CPP_HUMANOID_DEMO_MODEL}; ///< MJCF.
        std::string command{"右手を上げて"}; ///< Initial Brain instruction.
        std::size_t steps{2'000};            ///< Headless physics steps.
        std::optional<std::filesystem::path>
            snapshot;  ///< Optional offscreen PPM output.
        bool viewer{}; ///< Opens the GLFW viewer.
};

/// Parses only explicit options so accidental arguments fail visibly.
Options parse(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--viewer")
            options.viewer = true;
        else if (argument == "--headless")
            options.viewer = false;
        else if (argument == "--model" && index + 1 < argc)
            options.model = argv[++index];
        else if (argument == "--command" && index + 1 < argc)
            options.command = argv[++index];
        else if (argument == "--steps" && index + 1 < argc)
            options.steps = std::stoull(argv[++index]);
        else if (argument == "--snapshot" && index + 1 < argc)
            options.snapshot = argv[++index];
        else
            throw std::invalid_argument(
                "usage: humanoid_demo [--headless|--viewer] [--model path] "
                "[--command text] [--steps count] [--snapshot output.ppm]");
    }
    return options;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = parse(argc, argv);
        ai::simulation::demo::HumanoidDemo demo{options.model};
        demo.initialize();
        const auto result = demo.command(options.command);
        if (!result.recognized || !result.planned || !result.dispatched)
            throw std::runtime_error(
                "unsupported command; try: 右手を上げて / 右手を下げて");

        std::cout << "goal=" << result.goalType << " command=accepted\n";
        if (options.viewer) {
#ifdef AI_CPP_MUJOCO_VIEWER
            ai::simulation::MuJoCoViewer viewer;
            viewer.run(demo.plant(), demo.manager());
#else
            throw std::runtime_error(
                "viewer support was not enabled at build time");
#endif
        } else {
            demo.runSteps(options.steps);
            if (options.snapshot) {
#ifdef AI_CPP_MUJOCO_VIEWER
                ai::simulation::MuJoCoViewer viewer;
                viewer.saveFrame(demo.plant(), *options.snapshot);
                std::cout << "snapshot=" << options.snapshot->string() << '\n';
#else
                throw std::runtime_error(
                    "snapshot support was not enabled at build time");
#endif
            }
            const auto target = demo.target();
            std::cout << "simulation_time="
                      << demo.manager().plant().getState().simulationTime
                      << " joint=right_elbow target="
                      << (target ? target->position : 0.0)
                      << " position=" << demo.rightElbowPosition() << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "humanoid demo error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
