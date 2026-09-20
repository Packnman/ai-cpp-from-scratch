#pragma once

#include "simulation/PlantTypes.hpp"

namespace ai::simulation {

/// Exposes ideal, noise-free sensor readings from a Plant state snapshot.
class GroundTruthSensorSimulator {
    public:
        /// Copies the sensor readings associated with the supplied state.
        [[nodiscard]] std::vector<SensorState>
        sample(const RobotPlantState &) const;
};

} // namespace ai::simulation
