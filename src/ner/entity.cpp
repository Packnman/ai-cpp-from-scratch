#include "ai/ner/entity.h"

#include <array>
#include <stdexcept>

namespace ai::ner {
namespace {
using Pair = std::pair<EntityType, std::string_view>;
constexpr std::array<Pair, 14> names{{
    {EntityType::Person, "人名"},
    {EntityType::Corporation, "法人名"},
    {EntityType::PoliticalOrganization, "政治的組織名"},
    {EntityType::OtherOrganization, "その他の組織名"},
    {EntityType::Location, "地名"},
    {EntityType::Facility, "施設名"},
    {EntityType::Product, "製品名"},
    {EntityType::Event, "イベント名"},
    {EntityType::Money, "金額"},
    {EntityType::Date, "日付"},
    {EntityType::Time, "時刻"},
    {EntityType::Duration, "期間"},
    {EntityType::Quantity, "数量"},
    {EntityType::Condition, "条件"},
}};
} // namespace

std::string to_string(EntityType value) {
    for (const auto &[type, name] : names)
        if (type == value)
            return std::string(name);
    throw std::invalid_argument("unknown entity type");
}
std::optional<EntityType> entity_type_from_string(std::string_view value) {
    for (const auto &[type, name] : names)
        if (name == value)
            return type;
    return std::nullopt;
}
std::string to_string(EntitySource source) {
    switch (source) {
    case EntitySource::Rule:
        return "rule";
    case EntitySource::Model:
        return "model";
    case EntitySource::Hybrid:
        return "hybrid";
    }
    throw std::invalid_argument("unknown entity source");
}
bool valid_mention(std::string_view text, const EntityMention &mention) {
    return mention.start < mention.end && mention.end <= text.size() &&
           text.substr(mention.start, mention.end - mention.start) ==
               mention.surface;
}

std::vector<Utf8CodePoint> decode_utf8(std::string_view text) {
    std::vector<Utf8CodePoint> out;
    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        std::size_t width = 0;
        char32_t value = 0;
        if (lead < 0x80) {
            width = 1;
            value = lead;
        } else if ((lead & 0xe0) == 0xc0) {
            width = 2;
            value = lead & 0x1f;
        } else if ((lead & 0xf0) == 0xe0) {
            width = 3;
            value = lead & 0x0f;
        } else if ((lead & 0xf8) == 0xf0) {
            width = 4;
            value = lead & 0x07;
        } else {
            throw std::invalid_argument("invalid UTF-8 lead byte");
        }
        if (i + width > text.size())
            throw std::invalid_argument("truncated UTF-8");
        for (std::size_t j = 1; j < width; ++j) {
            const auto c = static_cast<unsigned char>(text[i + j]);
            if ((c & 0xc0) != 0x80)
                throw std::invalid_argument("invalid UTF-8 continuation byte");
            value = (value << 6) | (c & 0x3f);
        }
        if ((width == 2 && value < 0x80) || (width == 3 && value < 0x800) ||
            (width == 4 && value < 0x10000) || value > 0x10ffff ||
            (value >= 0xd800 && value <= 0xdfff))
            throw std::invalid_argument("non-canonical UTF-8");
        out.push_back({value, i, i + width});
        i += width;
    }
    return out;
}
} // namespace ai::ner
