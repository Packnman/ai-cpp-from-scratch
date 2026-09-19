#pragma once

#include <chrono>

namespace ai::brain {

using TimePoint = std::chrono::steady_clock::time_point;
using WallTimePoint = std::chrono::system_clock::time_point;
using Duration = std::chrono::milliseconds;

inline TimePoint steady_now() noexcept { return TimePoint::clock::now(); }

inline Duration elapsed(TimePoint begin, TimePoint end) noexcept {
    return std::chrono::duration_cast<Duration>(end - begin);
}

inline bool expired(TimePoint now, TimePoint deadline) noexcept {
    return now >= deadline;
}

} // namespace ai::brain
