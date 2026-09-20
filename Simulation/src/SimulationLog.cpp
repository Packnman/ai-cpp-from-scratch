#include "simulation/SimulationLog.hpp"

#include <stdexcept>

namespace ai::simulation {

CsvSimulationLogger::CsvSimulationLogger(const std::filesystem::path &path)
    : _stream(path) {
    if (!_stream)
        throw std::runtime_error("failed to open simulation log");
    _stream << "simulation_time,joint,position,velocity,acceleration,"
               "applied_torque,contacts\n";
}

void CsvSimulationLogger::write(const RobotPlantState &state) {
    // Emit one row per joint to keep high-rate logs simple to stream.
    for (const auto &joint : state.joints)
        _stream << state.simulationTime << ',' << joint.id << ','
                << joint.position << ',' << joint.velocity << ','
                << joint.acceleration << ',' << joint.appliedTorque << ','
                << state.contacts.size() << '\n';
}

} // namespace ai::simulation
