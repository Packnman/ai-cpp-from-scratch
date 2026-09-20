#include "actuator/manager/ActuatorManager.hpp"

#include <stdexcept>
#include <utility>

namespace ai::actuator {

void ActuatorManager::registerActuator(
    ActuatorDescriptor descriptor, std::unique_ptr<IActuatorDriver> driver) {
    if (descriptor.actuatorId.empty())
        throw std::invalid_argument("actuator id is empty");
    if (!driver)
        throw std::invalid_argument("actuator driver is null");
    // The registry takes ownership only after ID and pointer validation.
    const auto id = descriptor.actuatorId;
    if (_entries.contains(id))
        throw std::invalid_argument("duplicate actuator id");
    _entries.emplace(id, Entry{std::move(descriptor), std::move(driver)});
}

bool ActuatorManager::contains(std::string_view actuatorId) const {
    return _entries.contains(std::string{actuatorId});
}

void ActuatorManager::enable(std::string_view actuatorId) {
    entry(actuatorId).driver->enable();
}

void ActuatorManager::disable(std::string_view actuatorId) {
    entry(actuatorId).driver->disable();
}

void ActuatorManager::command(const DriveCommand &command) {
    entry(command.actuatorId).driver->command(command);
}

void ActuatorManager::stop(std::string_view actuatorId, StopMode mode) {
    entry(actuatorId).driver->stop(mode);
}

ActuatorState ActuatorManager::readState(std::string_view actuatorId) {
    return entry(actuatorId).driver->readState();
}

const ActuatorDescriptor &
ActuatorManager::descriptor(std::string_view actuatorId) const {
    return entry(actuatorId).descriptor;
}

IActuatorDriver &ActuatorManager::driver(std::string_view actuatorId) {
    return *entry(actuatorId).driver;
}

ActuatorManager::Entry &ActuatorManager::entry(std::string_view actuatorId) {
    const auto found = _entries.find(std::string{actuatorId});
    if (found == _entries.end())
        throw std::out_of_range("unknown actuator id");
    return found->second;
}

const ActuatorManager::Entry &
ActuatorManager::entry(std::string_view actuatorId) const {
    const auto found = _entries.find(std::string{actuatorId});
    if (found == _entries.end())
        throw std::out_of_range("unknown actuator id");
    return found->second;
}

} // namespace ai::actuator
