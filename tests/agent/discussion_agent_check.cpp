#include "ai/agent/agent.h"
#include "ai/agent/discussion.h"
#include "ai/agent/memory.h"
#include "ai/agent/reasoner.h"
#include "ai/agent/tools.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace ai::agent;

namespace {
int checks = 0;
#define CHECK(value)                                                           \
    do {                                                                       \
        ++checks;                                                              \
        if (!(value))                                                          \
            throw std::runtime_error("check failed: " #value);                 \
    } while (false)

struct RejectingModel final : ILanguageModel {
        int calls{};
        std::string complete(ModelMode, std::string_view) override {
            ++calls;
            throw std::runtime_error("model must not run for /compare");
        }
};

nlohmann::json initial() {
    const auto evidence = [](std::string id, std::string origin) {
        return nlohmann::json{{"id", id},
                              {"text", id},
                              {"source_id", "fixture"},
                              {"start", 0},
                              {"end", id.size()},
                              {"origin", origin},
                              {"content_verified", false}};
    };
    const auto claim = [](std::string id, std::string subject,
                          std::string attribute, double value, std::string unit,
                          std::string evidence_id) {
        return nlohmann::json{{"id", id},
                              {"speaker", "source"},
                              {"subject", subject},
                              {"attribute", attribute},
                              {"time", "current"},
                              {"condition", "default"},
                              {"value", value},
                              {"unit", unit},
                              {"stance", "asserted"},
                              {"evidence_ids", {evidence_id}}};
    };
    return {{"scenario_id", "agent-e2e"},
            {"topic", "A案とB案"},
            {"objective", "必須条件で比較"},
            {"options", {{{"id", "A"}}, {{"id", "B"}}}},
            {"evidence",
             {evidence("a-price", "document"), evidence("a-memory", "document"),
              evidence("b-price", "document"), evidence("b-memory", "document"),
              evidence("budget", "user"), evidence("memory-limit", "user")}},
            {"claims",
             {claim("ca-price", "A", "price", 30000, "JPY", "a-price"),
              claim("ca-memory", "A", "memory", 4, "GB", "a-memory"),
              claim("cb-price", "B", "price", 50000, "JPY", "b-price"),
              claim("cb-memory", "B", "memory", 2, "GB", "b-memory")}},
            {"constraints",
             {{{"id", "budget"},
               {"attribute", "price"},
               {"op", "<="},
               {"value", 40000},
               {"unit", "JPY"},
               {"required", true},
               {"evidence_ids", {"budget"}}},
              {{"id", "memory-limit"},
               {"attribute", "memory"},
               {"op", "<="},
               {"value", 3},
               {"unit", "GB"},
               {"required", true},
               {"evidence_ids", {"memory-limit"}}}}},
            {"requested_attributes", {"price", "memory"}},
            {"priorities", nlohmann::json::array()}};
}

void run() {
    const auto directory =
        std::filesystem::temp_directory_path() / "discussion_agent_check";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    auto reasoner = std::make_shared<RuleReasoner>();
    auto memory = std::make_shared<SqliteMemory>(directory / "memory.sqlite");
    auto tools = std::make_shared<ToolRegistry>();
    tools->add("calculator.calculate", std::make_shared<CalculatorTool>());
    Agent agent(std::make_shared<DefaultInputParser>(reasoner),
                std::make_shared<DefaultRouter>(),
                std::make_shared<DefaultPlanner>(reasoner),
                std::make_shared<DefaultExecutor>(
                    tools, std::make_shared<DiscussionEngine>()),
                std::make_shared<DefaultEvaluator>(reasoner),
                std::make_shared<DefaultAggregator>(), memory, reasoner);

    const auto first = agent.process("/compare " + initial().dump());
    CHECK(first.success);
    CHECK(first.state.discussion_state["decision"]["status"] ==
          "no_feasible_option");
    CHECK(first.text.find("満たす選択肢はありません") != std::string::npos);

    const nlohmann::json update = {{"scenario_id", "agent-e2e"},
                                   {"updates",
                                    {{{"id", "raise-budget"},
                                      {"kind", "correct_constraint"},
                                      {"target_id", "budget"},
                                      {"value", 60000},
                                      {"evidence",
                                       {{"id", "budget-correction"},
                                        {"text", "予算を6万円へ訂正"},
                                        {"source_id", "user:2"},
                                        {"start", 0},
                                        {"end", 27},
                                        {"origin", "user"},
                                        {"content_verified", false}}}}}}};
    const auto corrected = agent.process("/compare " + update.dump());
    CHECK(corrected.success);
    CHECK(corrected.state.discussion_state["decision"]["status"] ==
          "supported");
    CHECK(corrected.state.discussion_state["decision"]["selected_option"] ==
          "B");
    CHECK(corrected.state.discussion_state["revisions"].size() == 1);

    const auto repeated = agent.process("/compare " + update.dump());
    CHECK(repeated.success);
    CHECK(repeated.state.discussion_state["revisions"].size() == 1);
    const auto chat = agent.process("こんにちは");
    CHECK(chat.success);
    CHECK(chat.state.discussion_state["decision"]["selected_option"] == "B");
    AgentResponse after_summaries;
    for (int phase = 0; phase < 2; ++phase)
        for (int index = 0; index < 5; ++index)
            after_summaries =
                agent.process(std::string(180, char('a' + phase)));
    CHECK(after_summaries.success);
    CHECK(!after_summaries.state.conversation_summary.empty());
    CHECK(
        after_summaries.state.discussion_state["decision"]["selected_option"] ==
        "B");
    CHECK(agent.process("/calc 6*7").text == "42.0");

    auto rejecting_model = std::make_shared<RejectingModel>();
    auto model_reasoner = std::make_shared<ModelReasoner>(rejecting_model);
    auto model_memory =
        std::make_shared<SqliteMemory>(directory / "model-memory.sqlite");
    Agent model_agent(std::make_shared<DefaultInputParser>(model_reasoner),
                      std::make_shared<DefaultRouter>(),
                      std::make_shared<DefaultPlanner>(model_reasoner),
                      std::make_shared<DefaultExecutor>(
                          tools, std::make_shared<DiscussionEngine>()),
                      std::make_shared<DefaultEvaluator>(model_reasoner),
                      std::make_shared<DefaultAggregator>(), model_memory,
                      model_reasoner);
    CHECK(model_agent.process("/compare " + initial().dump()).success);
    CHECK(rejecting_model->calls == 0);
    std::filesystem::remove_all(directory);
}
} // namespace

int main() {
    try {
        run();
        std::cout << "discussion_agent_check: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
