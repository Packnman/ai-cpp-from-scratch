#include "ai/ner/extractor.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace ai::ner {
namespace {
struct Pattern {
        EntityType type;
        std::regex expression;
};

std::string ascii_number(std::string_view input) {
    const auto cps = decode_utf8(input);
    std::string out;
    for (const auto &cp : cps) {
        if (cp.value >= U'０' && cp.value <= U'９')
            out += static_cast<char>('0' + cp.value - U'０');
        else if (cp.value == U'，' || cp.value == U'．')
            out += cp.value == U'，' ? ',' : '.';
        else
            out.append(
                input.substr(cp.byte_start, cp.byte_end - cp.byte_start));
    }
    return out;
}

std::optional<std::string> normalize(EntityType type, std::string surface) {
    surface = ascii_number(surface);
    std::smatch match;
    const std::regex number(R"(([0-9]+(?:[.,][0-9]+)?))");
    if (type == EntityType::Money &&
        std::regex_search(surface, match, number)) {
        double value =
            std::stod(std::regex_replace(match.str(), std::regex(","), ""));
        if (surface.find("兆") != std::string::npos)
            value *= 1e12;
        else if (surface.find("億") != std::string::npos)
            value *= 1e8;
        else if (surface.find("万") != std::string::npos)
            value *= 1e4;
        std::ostringstream stream;
        stream << (surface.find("円") != std::string::npos ? "JPY:" : "VALUE:")
               << std::fixed
               << std::setprecision(value == std::floor(value) ? 0 : 6)
               << value;
        auto result = stream.str();
        while (result.find('.') != std::string::npos && result.back() == '0')
            result.pop_back();
        if (!result.empty() && result.back() == '.')
            result.pop_back();
        return result;
    }
    if (type == EntityType::Date) {
        const std::regex full(R"(([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日)");
        if (std::regex_match(surface, match, full)) {
            std::ostringstream stream;
            stream << match[1].str() << '-' << std::setfill('0') << std::setw(2)
                   << std::stoi(match[2].str()) << '-' << std::setw(2)
                   << std::stoi(match[3].str());
            return stream.str();
        }
        return std::nullopt; // Partial and relative dates need external
                             // context.
    }
    if (type == EntityType::Time) {
        const std::regex time(R"(([0-9]{1,2})時(?:([0-9]{1,2})分)?)");
        if (std::regex_match(surface, match, time)) {
            std::ostringstream stream;
            stream << std::setfill('0') << std::setw(2)
                   << std::stoi(match[1].str()) << ':' << std::setw(2)
                   << (match[2].matched ? std::stoi(match[2].str()) : 0);
            return stream.str();
        }
    }
    if (type == EntityType::Duration || type == EntityType::Quantity) {
        if (std::regex_search(surface, match, number))
            return match.str() + "|" +
                   surface.substr(match.position() + match.length());
    }
    return std::nullopt;
}

bool overlaps(const EntityMention &a, const EntityMention &b) {
    return a.start < b.end && b.start < a.end;
}

void add_matches(std::string_view text, std::string_view utterance,
                 const Pattern &pattern, std::vector<EntityMention> &out) {
    const std::string owned(text);
    for (std::sregex_iterator
             it(owned.begin(), owned.end(), pattern.expression),
         end;
         it != end; ++it) {
        const auto start = static_cast<std::size_t>(it->position());
        const auto finish = start + static_cast<std::size_t>(it->length());
        EntityMention mention{pattern.type,
                              start,
                              finish,
                              it->str(),
                              normalize(pattern.type, it->str()),
                              EntitySource::Rule,
                              std::string(utterance),
                              std::nullopt};
        if (std::none_of(out.begin(), out.end(), [&](const auto &old) {
                return overlaps(old, mention);
            }))
            out.push_back(std::move(mention));
    }
}
} // namespace

std::vector<EntityMention>
RuleEntityExtractor::extract(std::string_view text,
                             std::string_view utterance_id) const {
    decode_utf8(text); // Reject invalid input before exposing offsets.
    const std::string digits = "(?:[0-9]|０|１|２|３|４|５|６|７|８|９)+";
    const std::string decimal = digits + "(?:(?:\\.|,|，|．)" + digits + ")?";
    // Priority is semantic specificity. Later matches never replace an overlap.
    const std::vector<Pattern> patterns{
        {EntityType::Money,
         std::regex(decimal + "(?:万|億|兆)?(?:円|ドル|ユーロ)")},
        {EntityType::Date,
         std::regex("(?:" + digits + "年)?" + digits + "月" + digits + "日")},
        {EntityType::Time, std::regex(digits + "時(?:" + digits + "分)?")},
        {EntityType::Duration,
         std::regex(
             decimal +
             "(?:年間|年|か月|ヶ月|カ月|月間|週間|週|日間|時間|分間|秒間)")},
        {EntityType::Quantity,
         std::regex(decimal +
                    "(?:個|名|人|件|本|枚|台|冊|回|kg|g|km|m|リットル)")},
    };
    std::vector<EntityMention> result;
    for (const auto &pattern : patterns)
        add_matches(text, utterance_id, pattern, result);

    for (const std::string marker :
         {"もし", "ただし", "条件は", "場合は", "場合、"}) {
        std::size_t start = 0;
        while ((start = text.find(marker, start)) != std::string_view::npos) {
            auto finish = text.size();
            for (const std::string boundary : {"。", "！", "？"}) {
                const auto found = text.find(boundary, start);
                if (found != std::string_view::npos)
                    finish = std::min(finish, found);
            }
            EntityMention mention{
                EntityType::Condition,
                start,
                finish,
                std::string(text.substr(start, finish - start)),
                std::nullopt,
                EntitySource::Rule,
                std::string(utterance_id),
                std::nullopt};
            if (mention.start < mention.end &&
                std::none_of(result.begin(), result.end(),
                             [&](const auto &old) {
                                 return old.type == EntityType::Condition &&
                                        overlaps(old, mention);
                             }))
                result.push_back(std::move(mention));
            start += marker.size();
        }
    }
    std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
        return a.start != b.start ? a.start < b.start : a.end < b.end;
    });
    for (const auto &mention : result)
        if (!valid_mention(text, mention))
            throw std::logic_error("extractor produced an invalid byte span");
    return result;
}

HybridEntityExtractor::HybridEntityExtractor(
    std::shared_ptr<IEntityExtractor> rules,
    std::shared_ptr<IEntityExtractor> model)
    : _rules(std::move(rules)), _model(std::move(model)) {
    if (!_rules || !_model)
        throw std::invalid_argument(
            "hybrid extractor requires both extractors");
}
std::vector<EntityMention>
HybridEntityExtractor::extract(std::string_view text,
                               std::string_view utterance_id) const {
    auto result = _rules->extract(text, utterance_id);
    auto model = _model->extract(text, utterance_id);
    // Deterministic policy: high-precision numeric rules win overlaps; disjoint
    // model NER spans retain all eight upstream classes.
    for (auto &mention : model) {
        if (std::none_of(result.begin(), result.end(), [&](const auto &rule) {
                return rule.type != EntityType::Condition &&
                       overlaps(rule, mention);
            })) {
            mention.source = EntitySource::Hybrid;
            result.push_back(std::move(mention));
        }
    }
    std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
        return a.start != b.start ? a.start < b.start : a.end < b.end;
    });
    return result;
}
} // namespace ai::ner
