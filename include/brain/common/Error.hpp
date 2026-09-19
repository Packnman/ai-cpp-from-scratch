#pragma once

#include "brain/common/Time.hpp"
#include "brain/common/Types.hpp"

#include <string>

namespace ai::brain {

enum class ErrorLevel { Trace, Debug, Info, Warning, Error, Critical };

enum class RecoveryAction {
    None,
    Retry,
    Ignore,
    Replan,
    Fallback,
    Cancel,
    SafeStop,
    EmergencyStop
};

struct BrainError {
        ErrorId id{};
        ErrorLevel level{ErrorLevel::Error};
        ModuleId module{};
        TimePoint timestamp{};
        std::string description;
        std::string cause;
        RecoveryAction recovery{RecoveryAction::None};

        bool operator==(const BrainError &) const = default;
};

inline bool valid_error(const BrainError &error) noexcept {
    return valid_id(error.id) && !error.description.empty();
}

} // namespace ai::brain
