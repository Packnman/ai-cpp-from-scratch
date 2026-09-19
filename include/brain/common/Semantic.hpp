#pragma once

#include "brain/common/Attribute.hpp"
#include "brain/common/Time.hpp"
#include "brain/common/Types.hpp"

#include <optional>

namespace ai::brain {

enum class SemanticType {
    Unknown,
    Perception,
    RobotState,
    Goal,
    Condition,
    Constraint,
    Conversation,
    ActionResult,
    Error
};

struct SemanticItem {
        SemanticId id{};
        SemanticType type{SemanticType::Unknown};
        TimePoint timestamp{};
        float confidence{};
        bool valid{true};
        std::optional<SemanticId> target;
        AttributeMap attributes;

        bool operator==(const SemanticItem &) const = default;
};

bool valid_confidence(float confidence) noexcept;
bool valid_semantic_item(const SemanticItem &item) noexcept;

} // namespace ai::brain
