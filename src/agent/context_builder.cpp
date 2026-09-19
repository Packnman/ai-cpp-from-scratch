#include "ai/agent/context_builder.h"

#include <algorithm>
#include <stdexcept>

namespace ai::agent {
ContextBuilder::ContextBuilder(std::size_t budget, Counter counter)
    : _budget(budget), _counter(std::move(counter)) {
    if (!_counter)
        _counter = utf8_codepoints;
}
std::size_t ContextBuilder::utf8_codepoints(std::string_view s) {
    std::size_t n = 0;
    for (unsigned char c : s)
        if ((c & 0xc0) != 0x80)
            ++n;
    return n;
}
std::string ContextBuilder::build(const ContextInput &in) const {
    auto section = [](std::string_view name, std::string_view body) {
        return "[" + std::string(name) + "]\n" + std::string(body) + "\n";
    };
    std::string fixed = section("Current Input", in.current_input) +
                        section("Goal", in.goal) +
                        section("Current Task", in.current_task);
    std::string critical;
    for (const auto &c : in.constraints)
        if (c.critical)
            critical += (critical.empty() ? "" : "\n") + c.text;
    fixed += section("Critical Constraints", critical);
    if (in.require_summary)
        fixed += section("Conversation Summary",
                         in.summary.empty() ? "- なし" : in.summary);
    if (_counter(fixed) > _budget)
        throw std::length_error("required context exceeds budget");
    std::vector<std::string> optional;
    if (!in.entity_candidates.empty()) {
        std::string evidence =
            "抽出候補であり確定事実ではない。否定・提案・訂正は別途判断する。\n";
        for (const auto &mention : in.entity_candidates) {
            evidence += "- " + ai::ner::to_string(mention.type) + " [" +
                        std::to_string(mention.start) + "," +
                        std::to_string(mention.end) + "): " + mention.surface;
            if (mention.normalized) evidence += " => " + *mention.normalized;
            evidence += " (" + ai::ner::to_string(mention.source) + ")\n";
        }
        optional.push_back(section("Grounded Entity Candidates", evidence));
    }
    const auto recent_begin =
        in.recent.end() - std::min(in.recent.size(), in.recent_limit);
    for (auto it = recent_begin; it != in.recent.end(); ++it)
        optional.push_back(
            section("Recent Conversation",
                    "User: " + it->user + "\nAgent: " + it->assistant));
    for (const auto &m : in.memories)
        optional.push_back(section("Retrieved Memory",
                                   "[" + to_string(m.type) + "] " + m.content));
    if (!in.require_summary && !in.summary.empty())
        optional.push_back(section("Conversation Summary", in.summary));
    for (const auto &r : in.previous_results)
        optional.push_back(section("Previous Result", r.value.dump()));
    std::string out = fixed;
    for (const auto &item : optional)
        if (_counter(out) + _counter(item) <= _budget)
            out += item;
    return out;
}
ContextInput structured_prompt_input(ModelMode, std::string_view payload,
                                     std::string_view schema, bool repair) {
    ContextInput context;
    context.current_input = std::string(payload);
    context.goal = repair ? "Repair invalid JSON once and return JSON only"
                          : "Return an answer matching the requested mode";
    context.current_task = "Schema: " + std::string(schema);
    return context;
}
std::string build_model_prompt(const ContextInput &input, std::size_t budget,
                               ContextBuilder::Counter counter) {
    return ContextBuilder(budget, std::move(counter)).build(input);
}
} // namespace ai::agent
