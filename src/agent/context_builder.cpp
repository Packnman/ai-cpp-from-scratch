#include "ai/agent/context_builder.h"

#include <stdexcept>

namespace ai::agent {
ContextBuilder::ContextBuilder(std::size_t budget, Counter counter)
    : budget_(budget), counter_(std::move(counter)) {
    if (!counter_)
        counter_ = utf8_codepoints;
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
    if (counter_(fixed) > budget_)
        throw std::length_error("required context exceeds budget");
    std::vector<std::string> optional;
    for (auto it = in.recent.rbegin(); it != in.recent.rend(); ++it)
        optional.push_back(
            section("Recent Conversation",
                    "User: " + it->user + "\nAgent: " + it->assistant));
    for (const auto &m : in.memories)
        optional.push_back(section("Retrieved Memory",
                                   "[" + to_string(m.type) + "] " + m.content));
    if (!in.summary.empty())
        optional.push_back(section("Conversation Summary", in.summary));
    for (const auto &r : in.previous_results)
        optional.push_back(section("Previous Result", r.value.dump()));
    std::string out = fixed;
    for (const auto &item : optional)
        if (counter_(out) + counter_(item) <= budget_)
            out += item;
    return out;
}
} // namespace ai::agent
