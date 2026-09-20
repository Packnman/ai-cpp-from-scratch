#pragma once

#include "simulation/PlantTypes.hpp"

#include <filesystem>
#include <fstream>

namespace ai::simulation {

/// Writes high-rate Plant snapshots to a compact CSV stream.
class CsvSimulationLogger {
    public:
        /// Opens the destination and writes the CSV header.
        explicit CsvSimulationLogger(const std::filesystem::path &);
        /// Appends all joint values from one simulation snapshot.
        void write(const RobotPlantState &);

    private:
        std::ofstream _stream; ///< Owned output stream for the CSV file.
};

} // namespace ai::simulation
