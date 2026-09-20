#include "simulation/MuJoCoViewer.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace ai::simulation {

/// GLFW and MuJoCo rendering state hidden from the public interface.
class MuJoCoViewer::Impl {
    public:
        GLFWwindow *window{};   ///< Owned GLFW window.
        const mjModel *model{}; ///< Non-owning model used for camera motion.
        mjvCamera camera{};     ///< Interactive camera state.
        mjvOption option{};     ///< MuJoCo visualization options.
        mjvScene scene{};       ///< Allocated render scene.
        mjrContext context{};   ///< GPU rendering context.
        bool leftButton{};      ///< Left-button drag state.
        bool middleButton{};    ///< Middle-button drag state.
        bool rightButton{};     ///< Right-button drag state.
        double lastX{};         ///< Previous cursor X coordinate.
        double lastY{};         ///< Previous cursor Y coordinate.

        /// Resolves the implementation stored as GLFW user data.
        static Impl &from(GLFWwindow *window) {
            return *static_cast<Impl *>(glfwGetWindowUserPointer(window));
        }
};

MuJoCoViewer::MuJoCoViewer() : _impl(std::make_unique<Impl>()) {}

MuJoCoViewer::~MuJoCoViewer() = default;

void MuJoCoViewer::run(MuJoCoPlant &plant, SimulationManager &manager) {
    if (!glfwInit())
        throw std::runtime_error("GLFW initialization failed");
    // Ensures GLFW global state is released on every exit path.
    struct GlfwGuard {
            ~GlfwGuard() { glfwTerminate(); }
    } guard;

    _impl->window =
        glfwCreateWindow(1280, 720, "ai_cpp MuJoCo", nullptr, nullptr);
    if (!_impl->window)
        throw std::runtime_error("GLFW window creation failed");
    glfwMakeContextCurrent(_impl->window);
    glfwSwapInterval(1);
    glfwSetWindowUserPointer(_impl->window, _impl.get());

    auto *model = plant.model().model();
    auto *data = plant.model().data();
    _impl->model = model;
    mjv_defaultCamera(&_impl->camera);
    mjv_defaultOption(&_impl->option);
    mjv_defaultScene(&_impl->scene);
    mjr_defaultContext(&_impl->context);
    mjv_makeScene(model, &_impl->scene, 2'000);
    mjr_makeContext(model, &_impl->context, mjFONTSCALE_150);
    // Frees MuJoCo render resources before the GLFW window disappears.
    struct RenderGuard {
            Impl &impl;
            ~RenderGuard() {
                mjr_freeContext(&impl.context);
                mjv_freeScene(&impl.scene);
                glfwDestroyWindow(impl.window);
            }
    } renderGuard{*_impl};

    glfwSetKeyCallback(_impl->window,
                       [](GLFWwindow *window, int key, int, int action, int) {
                           if (action != GLFW_PRESS)
                               return;
                           if (key == GLFW_KEY_ESCAPE)
                               glfwSetWindowShouldClose(window, GLFW_TRUE);
                       });
    glfwSetMouseButtonCallback(
        _impl->window, [](GLFWwindow *window, int button, int action, int) {
            auto &impl = Impl::from(window);
            const bool pressed = action == GLFW_PRESS;
            if (button == GLFW_MOUSE_BUTTON_LEFT)
                impl.leftButton = pressed;
            if (button == GLFW_MOUSE_BUTTON_MIDDLE)
                impl.middleButton = pressed;
            if (button == GLFW_MOUSE_BUTTON_RIGHT)
                impl.rightButton = pressed;
            glfwGetCursorPos(window, &impl.lastX, &impl.lastY);
        });
    glfwSetCursorPosCallback(_impl->window, [](GLFWwindow *window, double x,
                                               double y) {
        auto &impl = Impl::from(window);
        if (!impl.leftButton && !impl.middleButton && !impl.rightButton)
            return;
        int width{};
        int height{};
        glfwGetWindowSize(window, &width, &height);
        const double dx = (x - impl.lastX) / std::max(1, height);
        const double dy = (y - impl.lastY) / std::max(1, height);
        impl.lastX = x;
        impl.lastY = y;
        const bool shift =
            glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        const mjtMouse action =
            impl.rightButton  ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
            : impl.leftButton ? (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V)
                              : mjMOUSE_ZOOM;
        mjv_moveCamera(impl.model, action, dx, dy, &impl.camera);
    });
    glfwSetScrollCallback(_impl->window, [](GLFWwindow *window, double,
                                            double y) {
        auto &impl = Impl::from(window);
        mjv_moveCamera(impl.model, mjMOUSE_ZOOM, 0.0, -0.05 * y, &impl.camera);
    });

    auto previous = std::chrono::steady_clock::now();
    double accumulator{};
    while (!glfwWindowShouldClose(_impl->window)) {
        // Wall time only paces rendering; SimulationManager owns physics time.
        const auto now = std::chrono::steady_clock::now();
        accumulator += std::chrono::duration<double>(now - previous).count();
        previous = now;
        accumulator = std::min(accumulator, 0.1);
        while (!manager.paused() && accumulator >= manager.timing().physicsDt) {
            manager.step();
            accumulator -= manager.timing().physicsDt;
        }

        if (glfwGetKey(_impl->window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            manager.setPaused(!manager.paused());
            while (glfwGetKey(_impl->window, GLFW_KEY_SPACE) == GLFW_PRESS)
                glfwPollEvents();
        }
        if (glfwGetKey(_impl->window, GLFW_KEY_RIGHT) == GLFW_PRESS &&
            manager.paused()) {
            manager.setPaused(false);
            manager.step();
            manager.setPaused(true);
        }
        if (glfwGetKey(_impl->window, GLFW_KEY_BACKSPACE) == GLFW_PRESS)
            manager.reset();

        int width{};
        int height{};
        glfwGetFramebufferSize(_impl->window, &width, &height);
        const mjrRect viewport{0, 0, width, height};
        // Rendering reads the latest state and never advances the Plant itself.
        mjv_updateScene(model, data, &_impl->option, nullptr, &_impl->camera,
                        mjCAT_ALL, &_impl->scene);
        mjr_render(viewport, &_impl->scene, &_impl->context);
        glfwSwapBuffers(_impl->window);
        glfwPollEvents();
    }
}

} // namespace ai::simulation
