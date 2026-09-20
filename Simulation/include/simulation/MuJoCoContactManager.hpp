#pragma once

#include "simulation/MuJoCoModel.hpp"

namespace ai::simulation {

/// Converts MuJoCo contact records and local wrenches into Plant DTOs.
class MuJoCoContactManager {
    public:
        /// Returns every contact active in the current MuJoCo state.
        [[nodiscard]] std::vector<ContactState>
        contacts(const MuJoCoModel &) const;
};

} // namespace ai::simulation
