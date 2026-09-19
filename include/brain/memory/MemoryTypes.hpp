#pragma once
#include "brain/common/Dtos.hpp"
#include <optional>
#include <string>
#include <vector>
namespace ai::brain {
enum class MemoryType {
    Perception,
    Conversation,
    GoalOutcome,
    ActionOutcome,
    Error,
    Knowledge
};
struct MemoryItem {
        MemoryId id{};
        MemoryType type{MemoryType::Knowledge};
        SemanticItem content;
        float importance{};
        float confidence{};
        TimePoint createdAt{};
        TimePoint updatedAt{};
        TimePoint lastAccessed{};
        std::uint64_t accessCount{};
        std::uint32_t version{1};
        std::vector<std::string> tags;
};
struct MemoryQuery {
        std::optional<MemoryType> type;
        std::vector<std::string> tags;
        std::optional<SemanticId> target;
        std::size_t maxResults{20};
};
} // namespace ai::brain
