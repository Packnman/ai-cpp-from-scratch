#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ai::ner {

// The first eight values are the upstream Stockmark annotation classes.
enum class EntityType {
    Person,
    Corporation,
    PoliticalOrganization,
    OtherOrganization,
    Location,
    Facility,
    Product,
    Event,
    Money,
    Date,
    Time,
    Duration,
    Quantity,
    Condition
};

enum class EntitySource { Rule, Model, Hybrid };

struct EntityMention {
        EntityType type{EntityType::Person};
        // UTF-8 byte offsets, always a half-open interval [start, end).
        std::size_t start{};
        std::size_t end{};
        std::string surface;
        std::optional<std::string> normalized;
        EntitySource source{EntitySource::Rule};
        std::string utterance_id;
        // Model scores are uncalibrated ranking scores, not probabilities.
        std::optional<float> score;
};

std::string to_string(EntityType);
std::optional<EntityType> entity_type_from_string(std::string_view);
std::string to_string(EntitySource);
bool valid_mention(std::string_view text, const EntityMention &mention);

struct Utf8CodePoint {
        char32_t value{};
        std::size_t byte_start{};
        std::size_t byte_end{};
};

// Throws invalid_argument for malformed UTF-8. This mapping is the canonical
// bridge between model code-point indices and public byte offsets.
std::vector<Utf8CodePoint> decode_utf8(std::string_view);

} // namespace ai::ner
