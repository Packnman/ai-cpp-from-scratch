#include "simulation/SimulationManager.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ai::simulation {

SimulationManager::SimulationManager(std::unique_ptr<IPlant> plant,
                                     SimulationTiming timing)
    : _plant(std::move(plant)), _timing(timing) {
    if (!_plant)
        throw std::invalid_argument("simulation plant is null");
    if (!std::isfinite(_timing.physicsDt) || _timing.physicsDt <= 0.0 ||
        !std::isfinite(_timing.actuatorDt) || _timing.actuatorDt <= 0.0 ||
        !std::isfinite(_timing.controlDt) || _timing.controlDt <= 0.0 ||
        _timing.actuatorDt < _timing.physicsDt ||
        _timing.controlDt < _timing.physicsDt)
        throw std::invalid_argument("invalid simulation timing");
}

void SimulationManager::initialize() {
    _plant->initialize();
    _initialized = true;
    _controlAccumulator = _timing.controlDt;
    _actuatorAccumulator = _timing.actuatorDt;
}

void SimulationManager::reset() {
    if (!_initialized)
        throw std::logic_error("simulation manager is not initialized");
    _plant->reset();
    _controlAccumulator = _timing.controlDt;
    _actuatorAccumulator = _timing.actuatorDt;
}

void SimulationManager::step() {
    if (!_initialized)
        throw std::logic_error("simulation manager is not initialized");
    if (_paused)
        return;
    // Callbacks observe one consistent pre-physics snapshot for this cycle.
    const auto state = _plant->getState();
    constexpr double tolerance = 1e-12;
    if (_controlAccumulator + tolerance >= _timing.controlDt) {
        if (_controlCallback)
            _controlCallback(_timing.controlDt, state, *_plant);
        _controlAccumulator -= _timing.controlDt;
    }
    if (_actuatorAccumulator + tolerance >= _timing.actuatorDt) {
        if (_actuatorCallback)
            _actuatorCallback(_timing.actuatorDt, state, *_plant);
        _actuatorAccumulator -= _timing.actuatorDt;
    }
    // Advance physics exactly once; logging always receives post-step state.
    _plant->step(_timing.physicsDt);
    _controlAccumulator += _timing.physicsDt;
    _actuatorAccumulator += _timing.physicsDt;
    if (_logCallback)
        _logCallback(_plant->getState());
}

void SimulationManager::runSteps(std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
        step();
}

void SimulationManager::setControlCallback(CycleCallback callback) {
    _controlCallback = std::move(callback);
}

void SimulationManager::setActuatorCallback(CycleCallback callback) {
    _actuatorCallback = std::move(callback);
}

void SimulationManager::setLogCallback(LogCallback callback) {
    _logCallback = std::move(callback);
}

} // namespace ai::simulation
