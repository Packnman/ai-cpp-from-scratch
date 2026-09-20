#pragma once

#include "simulation/MuJoCoContactManager.hpp"

namespace ai::simulation {

/// Copies MuJoCo arrays into an engine-independent RobotPlantState snapshot.
class MuJoCoStateAdapter {
    public:
        /// Converts the owner's current mjData without retaining references.
        [[nodiscard]] RobotPlantState convert(const MuJoCoModel &) const;

    private:
        MuJoCoContactManager _contacts; ///< Contact conversion helper.
};

} // namespace ai::simulation
