#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace ai::brain {

using AttributeValue = std::variant<std::monostate, bool, std::int64_t,
                                    std::uint64_t, double, std::string>;
using AttributeMap = std::map<std::string, AttributeValue, std::less<>>;
using ByteBuffer = std::vector<std::uint8_t>;
using Payload =
    std::variant<std::monostate, std::string, ByteBuffer, AttributeMap>;

} // namespace ai::brain
