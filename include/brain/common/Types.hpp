#pragma once

#include <cstdint>

namespace ai::brain {

using SemanticId = std::uint64_t;
using GoalId = std::uint64_t;
using PlanId = std::uint64_t;
using ActionId = std::uint64_t;
using ConstraintId = std::uint64_t;
using ConditionId = std::uint64_t;
using PolicyId = std::uint64_t;
using MemoryId = std::uint64_t;
using RequestId = std::uint64_t;
using TraceId = std::uint64_t;
using ErrorId = std::uint64_t;
using ResourceId = std::uint64_t;
using TrackingId = std::uint64_t;
using SpeakerId = std::uint64_t;
using ModuleId = std::uint32_t;

template <class Id> constexpr bool valid_id(Id id) noexcept { return id != 0; }

} // namespace ai::brain
