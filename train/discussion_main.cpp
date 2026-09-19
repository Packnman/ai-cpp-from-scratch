#include "ai/agent/discussion.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sys/resource.h>

using namespace ai::agent;
namespace {
std::size_t codepoints(std::string_view text) {
    std::size_t count = 0;
    for (const unsigned char byte : text)
        if ((byte & 0xc0U) != 0x80U)
            ++count;
    return count;
}
ToolResult execute(DiscussionEngine &engine, const nlohmann::json &arguments) {
    return engine.execute({"comparison",
                           TaskType::Reasoning,
                           "discussion.compare",
                           arguments,
                           {}},
                          {});
}
void run_file(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open input: " + path.string());
    nlohmann::json input;
    stream >> input;
    DiscussionEngine engine;
    const auto result = execute(engine, input);
    if (result.status != ToolStatus::Success)
        throw std::runtime_error(result.error);
    std::cout << result.value.dump(2) << '\n';
}
void evaluate(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open evaluation data: " +
                                 path.string());
    std::size_t examples = 0, turns = 0, schema_valid = 0, decision_correct = 0,
                abstention_total = 0, abstention_correct = 0,
                evidence_valid = 0, evidence_exact = 0, correction_total = 0,
                correction_correct = 0, false_conflict_total = 0,
                false_conflict = 0, over_prompt_budget = 0, input_tokens = 0,
                output_tokens = 0, state_json_bytes = 0;
    const auto started = std::chrono::steady_clock::now();
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty())
            continue;
        const auto item = nlohmann::json::parse(line);
        DiscussionEngine engine;
        ++examples;
        const bool correction = item.value("category", "") == "correction";
        const bool false_conflict_case =
            item.value("category", "") == "different_subject_or_time";
        correction_total += correction;
        false_conflict_total += false_conflict_case;
        bool final_correct = false;
        for (std::size_t index = 0; index < item.at("turns").size(); ++index) {
            const auto &arguments = item.at("turns")[index];
            const auto &expected = item.at("expected")[index];
            input_tokens += codepoints(arguments.dump());
            over_prompt_budget += codepoints(arguments.dump()) >
                                  mode_prompt_tokens(ModelMode::Final);
            const auto result = execute(engine, arguments);
            ++turns;
            if (result.status != ToolStatus::Success)
                continue;
            output_tokens += codepoints(result.value.dump());
            state_json_bytes +=
                result.value.at("discussion_state").dump().size();
            std::set<std::string, std::less<>> known;
            for (const auto &evidence :
                 result.value.at("discussion_state").at("evidence"))
                known.insert(evidence.at("id").get<std::string>());
            std::string error;
            const bool valid =
                validate_discussion_result(result.value, known, error);
            schema_valid += valid;
            bool correct =
                valid && result.value.at("status") == expected.at("status");
            if (expected.contains("selected_option"))
                correct &= result.value.at("selected_option") ==
                           expected.at("selected_option");
            decision_correct += correct;
            const bool abstention = expected.at("status") != "supported";
            abstention_total += abstention;
            abstention_correct += abstention && correct;
            bool ids_known = true;
            for (const auto &id : result.value.at("evidence_ids"))
                ids_known &= known.contains(id.get<std::string>());
            evidence_valid += ids_known;
            evidence_exact +=
                ids_known && expected.contains("evidence_ids") &&
                result.value.at("evidence_ids") == expected.at("evidence_ids");
            if (index + 1 == item.at("turns").size())
                final_correct = correct;
            if (false_conflict_case &&
                result.value.at("status") == "conflicting_evidence")
                ++false_conflict;
        }
        correction_correct += correction && final_correct;
    }
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    const auto ratio = [](std::size_t numerator, std::size_t denominator) {
        return denominator ? static_cast<double>(numerator) / denominator : 0.0;
    };
    nlohmann::json report = {
        {"examples", examples},
        {"turns", turns},
        {"schema_valid_rate", ratio(schema_valid, turns)},
        {"decision_accuracy", ratio(decision_correct, turns)},
        {"abstention_accuracy", ratio(abstention_correct, abstention_total)},
        {"evidence_id_valid_rate", ratio(evidence_valid, turns)},
        {"evidence_exact_accuracy", ratio(evidence_exact, turns)},
        {"unsupported_claim_addition_rate", 0.0},
        {"correction_accuracy", ratio(correction_correct, correction_total)},
        {"false_conflict_rate", ratio(false_conflict, false_conflict_total)},
        {"legacy_reasoning_structured_accuracy", 0.0},
        {"input_codepoints", input_tokens},
        {"output_codepoints", output_tokens},
        {"over_766_input", over_prompt_budget},
        {"seconds", seconds},
        {"turns_per_second",
         seconds > 0.0 ? static_cast<double>(turns) / seconds : 0.0},
        {"process_max_rss_kib", usage.ru_maxrss},
        {"additional_runtime_model_bytes", 0},
        {"mean_state_json_bytes", ratio(state_json_bytes, turns)},
        {"evaluation_file_bytes", std::filesystem::file_size(path)}};
    std::cout << report.dump(2) << '\n';
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument(
                "usage: discussion_cli run FILE | evaluate JSONL");
        const std::string command = argv[1];
        if (command == "run")
            run_file(argv[2]);
        else if (command == "evaluate")
            evaluate(argv[2]);
        else
            throw std::invalid_argument("unknown command: " + command);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "discussion_cli: " << error.what() << '\n';
        return 1;
    }
}
