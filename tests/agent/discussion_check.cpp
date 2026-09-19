#include "ai/agent/discussion.h"

#include <iostream>
#include <stdexcept>

using namespace ai::agent;
namespace {
int checks = 0;
#define CHECK(x)                                                               \
    do {                                                                       \
        ++checks;                                                              \
        if (!(x))                                                              \
            throw std::runtime_error(std::string("check failed: ") + #x);      \
    } while (false)

nlohmann::json evidence(std::string id, std::string origin = "document",
                        bool verified = false) {
    return {{"id", id},
            {"text", "source text for " + id},
            {"source_id", "document-1"},
            {"start", 0},
            {"end", 10},
            {"origin", origin},
            {"content_verified", verified}};
}
nlohmann::json claim(std::string id, std::string subject, std::string attribute,
                     double value, std::string unit, std::string evidence_id,
                     std::string time = "current",
                     std::string stance = "asserted") {
    return {{"id", id},           {"speaker", "source"},
            {"subject", subject}, {"attribute", attribute},
            {"time", time},       {"condition", "default"},
            {"value", value},     {"unit", unit},
            {"stance", stance},   {"evidence_ids", {evidence_id}}};
}
nlohmann::json base(std::string id) {
    return {
        {"scenario_id", id},
        {"topic", "A案とB案"},
        {"objective", "必須条件内で選ぶ"},
        {"options",
         {{{"id", "A"}, {"label", "A案"}}, {{"id", "B"}, {"label", "B案"}}}},
        {"evidence",
         {evidence("e-a-price"), evidence("e-a-memory"), evidence("e-a-time"),
          evidence("e-b-price"), evidence("e-b-memory"), evidence("e-b-time"),
          evidence("e-budget", "user"), evidence("e-memory", "user")}},
        {"claims",
         {claim("a-price", "A", "price", 30000, "JPY", "e-a-price"),
          claim("a-memory", "A", "memory", 4, "GB", "e-a-memory"),
          claim("a-time", "A", "time", 2, "s", "e-a-time"),
          claim("b-price", "B", "price", 50000, "JPY", "e-b-price"),
          claim("b-memory", "B", "memory", 2, "GB", "e-b-memory"),
          claim("b-time", "B", "time", 1, "s", "e-b-time")}},
        {"constraints",
         {{{"id", "budget"},
           {"attribute", "price"},
           {"op", "<="},
           {"value", 40000},
           {"unit", "JPY"},
           {"required", true},
           {"evidence_ids", {"e-budget"}}},
          {{"id", "memory-limit"},
           {"attribute", "memory"},
           {"op", "<="},
           {"value", 3},
           {"unit", "GB"},
           {"required", true},
           {"evidence_ids", {"e-memory"}}}}},
        {"requested_attributes", {"price", "memory"}},
        {"priorities", nlohmann::json::array()}};
}
ToolResult run(DiscussionEngine &engine, const nlohmann::json &arguments) {
    return engine.execute(
        {"compare", TaskType::Reasoning, "discussion.compare", arguments, {}},
        {});
}
void core_cases() {
    DiscussionEngine engine;
    auto none = run(engine, base("none"));
    CHECK(none.status == ToolStatus::Success);
    CHECK(none.value["status"] == "no_feasible_option");
    CHECK(none.value["selected_option"].is_null());
    CHECK(none.value["open_questions"].size() == 1);

    auto one_input = base("one");
    one_input["constraints"][1]["value"] = 4;
    auto one = run(engine, one_input);
    CHECK(one.value["status"] == "supported");
    CHECK(one.value["selected_option"] == "A");

    auto priority = base("priority");
    priority["constraints"][0]["value"] = 60000;
    priority["constraints"][1]["value"] = 5;
    priority["requested_attributes"].push_back("time");
    priority["priorities"] = {{{"attribute", "time"}, {"direction", "min"}}};
    CHECK(run(engine, priority).value["selected_option"] == "B");
    priority["scenario_id"] = "priority-price";
    priority["priorities"] = {{{"attribute", "price"}, {"direction", "min"}}};
    CHECK(run(engine, priority).value["selected_option"] == "A");

    auto no_priority = priority;
    no_priority["scenario_id"] = "no-priority";
    no_priority["priorities"] = nlohmann::json::array();
    CHECK(run(engine, no_priority).value["status"] == "insufficient_evidence");

    auto missing = base("missing");
    missing["claims"].erase(missing["claims"].begin() + 4);
    CHECK(run(engine, missing).value["status"] == "insufficient_evidence");
    missing["scenario_id"] = "unknown-attribute";
    missing = base("unknown-attribute");
    missing["requested_attributes"].push_back("battery");
    CHECK(run(engine, missing).value["status"] == "insufficient_evidence");
}
void evidence_semantics() {
    DiscussionEngine engine;
    auto conflict = base("conflict");
    conflict["evidence"].push_back(evidence("e-conflict"));
    conflict["claims"].push_back(
        claim("a-price-2", "A", "price", 35000, "JPY", "e-conflict"));
    CHECK(run(engine, conflict).value["status"] == "conflicting_evidence");

    auto different_time = base("different-time");
    different_time["evidence"].push_back(evidence("e-old"));
    different_time["claims"].push_back(
        claim("a-old-price", "A", "price", 70000, "JPY", "e-old", "2025"));
    CHECK(run(engine, different_time).value["status"] == "no_feasible_option");

    auto proposal = base("proposal");
    proposal["claims"][0]["stance"] = "proposed";
    CHECK(run(engine, proposal).value["status"] == "insufficient_evidence");

    auto unsupported = base("unsupported-user");
    unsupported["evidence"][0]["origin"] = "user";
    CHECK(run(engine, unsupported).value["status"] == "insufficient_evidence");

    auto different_subject = base("different-subject");
    different_subject["claims"][3]["value"] = 30000;
    CHECK(run(engine, different_subject).value["status"] == "supported");
}
void corrections_and_validation() {
    DiscussionEngine engine;
    CHECK(run(engine, base("correction")).value["status"] ==
          "no_feasible_option");
    auto update_evidence = evidence("e-budget-correction", "user");
    nlohmann::json update = {{"scenario_id", "correction"},
                             {"updates",
                              {{{"id", "u-1"},
                                {"kind", "correct_constraint"},
                                {"target_id", "budget"},
                                {"value", 60000},
                                {"evidence", update_evidence}}}}};
    auto corrected = run(engine, update);
    CHECK(corrected.value["status"] == "supported");
    CHECK(corrected.value["selected_option"] == "B");
    CHECK(corrected.value["discussion_state"]["revisions"].size() == 1);
    CHECK(corrected.value["discussion_state"]["revisions"][0]
                         ["old_evidence_ids"][0] == "e-budget");
    auto repeated = run(engine, update);
    CHECK(repeated.value["discussion_state"]["revisions"].size() == 1);

    auto priority_case = base("priority-correction");
    priority_case["constraints"][0]["value"] = 60000;
    priority_case["constraints"][1]["value"] = 5;
    priority_case["requested_attributes"].push_back("time");
    priority_case["priorities"] = {
        {{"attribute", "time"}, {"direction", "min"}}};
    CHECK(run(engine, priority_case).value["selected_option"] == "B");
    const nlohmann::json priority_update = {
        {"scenario_id", "priority-correction"},
        {"updates",
         {{{"id", "priority-u-1"},
           {"kind", "set_priorities"},
           {"target_id", "priorities"},
           {"value", {{{"attribute", "price"}, {"direction", "min"}}}}}}}};
    const auto reprioritized = run(engine, priority_update);
    CHECK(reprioritized.value["selected_option"] == "A");
    CHECK(reprioritized.value["discussion_state"]["revisions"].size() == 1);

    auto invalid = update;
    invalid["updates"][0]["id"] = "bad";
    invalid["updates"][0]["target_id"] = "missing";
    CHECK(run(engine, invalid).status == ToolStatus::PermanentError);
    CHECK(engine.state("correction")["revisions"].size() == 1);

    std::string error;
    CHECK(!validate_discussion_result(
        {{"status", "supported"},
         {"conclusion", "x"},
         {"evidence_ids", {"unknown"}},
         {"open_questions", nlohmann::json::array()}},
        {"known"}, error));
    CHECK(!validate_discussion_result(
        {{"status", "supported"},
         {"conclusion", "x"},
         {"selected_option", 7},
         {"evidence_ids", nlohmann::json::array()},
         {"open_questions", nlohmann::json::array()}},
        {}, error));
}
void aggregator_semantics() {
    DefaultAggregator aggregator;
    auto unrelated = aggregator.aggregate(
        {{"a", ToolStatus::Success, {{"state", true}}, {}},
         {"b", ToolStatus::Success, {{"state", false}}, {}}});
    CHECK(unrelated["contradictions"].empty());
    auto explicit_conflict = aggregator.aggregate(
        {{"a",
          ToolStatus::Success,
          {{"semantic_conflicts", {"same-subject-time-condition"}}},
          {}}});
    CHECK(explicit_conflict["contradictions"].size() == 1);
}
} // namespace
int main() {
    try {
        core_cases();
        evidence_semantics();
        corrections_and_validation();
        aggregator_semantics();
        std::cout << "discussion_check: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
