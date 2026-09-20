#include "TestSupport.hpp"
#include "brain/BrainSystem.hpp"
#include "brain/preprocess/ModelContextRecognizer.hpp"

#include <memory>
#include <string>

using namespace ai::brain;

namespace {
constexpr const char *valid_response = R"json({
  "intent":"command",
  "goal":{"type":"AcquireObject","target":"blue_ball","priority":7},
  "conditions":[{"type":"ObjectVisible","active":true,"confidence":0.8,
                 "attributes":{"target":"blue_ball"}}],
  "constraints":[{"type":"safety","critical":true,
                  "scope":{"type":"global","target_id":null},
                  "source":"human_explicit",
                  "expression":{"type":"deny_action_type",
                                "arguments":{"type":"throw"}}}],
  "confidence":0.95
})json";

ContextModelResponse succeeded(std::string content) {
    return {ContextModelStatus::Succeeded, std::move(content), {}};
}

void expect_rule_fallback(brain_test::Suite &t, ContextModelResponse response,
                          std::string_view id) {
    auto backend =
        std::make_unique<FakeContextModelBackend>(std::move(response));
    auto *backendPointer = backend.get();
    auto logger = std::make_shared<InMemoryLogManager>();
    ModelContextRecognizer recognizer(std::move(backend),
                                      std::make_unique<RuleContextRecognizer>(),
                                      {}, logger);
    const auto result = recognizer.recognize("goal:fallback", steady_now());
    t.expect(result.goals.size() == 1 &&
                 result.goals.front().type == "fallback" &&
                 logger->errors().size() == 1 && backendPointer->calls() == 1 &&
                 backendPointer->lastRequest().text == "goal:fallback",
             id, "invalid model result falls back to rule recognition");
}
} // namespace

int main() {
    brain_test::Suite t{"Context model recognition"};
    const auto now = steady_now();

    auto validBackend =
        std::make_unique<FakeContextModelBackend>(succeeded(valid_response));
    ModelContextRecognizer validRecognizer(std::move(validBackend));
    const auto result = validRecognizer.recognize("青いボールを取って", now);
    t.expect(result.intent == "command" && result.goals.size() == 1 &&
                 result.goals.front().type == "AcquireObject" &&
                 result.goals.front().target &&
                 result.goals.front().priority == 7 &&
                 std::get<std::string>(
                     result.goals.front().completionCondition.arguments.at(
                         "target_name")) == "blue_ball",
             "CTX-MODEL-001", "valid goal is mapped to the existing DTO");
    t.expect(result.conditions.size() == 1 &&
                 result.conditions.front().type == "ObjectVisible" &&
                 result.conditions.front().active &&
                 std::get<std::string>(result.conditions.front().attributes.at(
                     "target")) == "blue_ball",
             "CTX-MODEL-002", "condition fields are mapped");
    t.expect(result.constraints.size() == 1 &&
                 result.constraints.front().type == "safety" &&
                 result.constraints.front().critical &&
                 result.constraints.front().expression.expression ==
                     "deny_action_type" &&
                 std::get<std::string>(
                     result.constraints.front().expression.arguments.at(
                         "type")) == "throw" &&
                 result.confidence == 0.95F,
             "CTX-MODEL-003", "constraint and confidence fields are mapped");

    expect_rule_fallback(t, succeeded("not-json"), "CTX-MODEL-010");
    expect_rule_fallback(
        t,
        succeeded(
            R"({"intent":"command","conditions":[],"constraints":[],"confidence":0.9})"),
        "CTX-MODEL-011");
    expect_rule_fallback(
        t,
        succeeded(
            R"({"intent":"invented","goal":null,"conditions":[],"constraints":[],"confidence":0.9})"),
        "CTX-MODEL-012");
    expect_rule_fallback(
        t,
        succeeded(
            R"({"intent":"command","goal":{"type":"Invented","target":"x","priority":1},"conditions":[],"constraints":[],"confidence":0.9})"),
        "CTX-MODEL-013");
    expect_rule_fallback(
        t,
        succeeded(
            R"({"intent":"command","goal":{"type":"AcquireObject","target":"bad target","priority":1},"conditions":[],"constraints":[],"confidence":0.9})"),
        "CTX-MODEL-014");
    expect_rule_fallback(
        t,
        succeeded(
            R"({"intent":"command","goal":{"type":"AcquireObject","target":"x","priority":1},"conditions":[],"constraints":[],"confidence":1.1})"),
        "CTX-MODEL-015");
    expect_rule_fallback(
        t,
        succeeded(
            R"({"intent":"command","goal":{"type":"AcquireObject","target":"x","priority":1},"conditions":[],"constraints":[],"confidence":-0.1})"),
        "CTX-MODEL-015B");
    expect_rule_fallback(
        t,
        succeeded(
            R"({"intent":"command","goal":null,"conditions":[],"constraints":[],"confidence":0.9,"extra":1})"),
        "CTX-MODEL-016");
    expect_rule_fallback(t, {ContextModelStatus::Timeout, {}, "fake timeout"},
                         "CTX-MODEL-020");
    expect_rule_fallback(
        t, {ContextModelStatus::ConnectionFailure, {}, "fake unavailable"},
        "CTX-MODEL-021");
    expect_rule_fallback(t,
                         {ContextModelStatus::HttpError, {}, "fake HTTP 500"},
                         "CTX-MODEL-022");
    expect_rule_fallback(
        t, {ContextModelStatus::InvalidResponse, {}, "fake invalid envelope"},
        "CTX-MODEL-023");

    auto throwingBackend =
        std::make_unique<FakeContextModelBackend>(succeeded(valid_response));
    throwingBackend->throwOnInfer(true);
    auto logger = std::make_shared<InMemoryLogManager>();
    ModelContextRecognizer throwingRecognizer(
        std::move(throwingBackend), std::make_unique<RuleContextRecognizer>(),
        {}, logger);
    const auto fallback = throwingRecognizer.recognize("goal:fallback", now);
    t.expect(fallback.goals.size() == 1 &&
                 fallback.goals.front().type == "fallback" &&
                 logger->errors().size() == 1,
             "CTX-MODEL-024", "backend exception falls back and is logged");

    auto deterministicBackend =
        std::make_unique<FakeContextModelBackend>(succeeded(valid_response));
    ModelContextRecognizer deterministic(std::move(deterministicBackend));
    const auto first = deterministic.recognize("same", now);
    const auto second = deterministic.recognize("same", now);
    t.expect(first.intent == second.intent && first.goals == second.goals &&
                 first.conditions == second.conditions &&
                 first.constraints == second.constraints &&
                 first.confidence == second.confidence,
             "CTX-MODEL-030", "same fake response maps deterministically");

    auto integrationBackend =
        std::make_unique<FakeContextModelBackend>(succeeded(valid_response));
    auto recognizer =
        std::make_unique<ModelContextRecognizer>(std::move(integrationBackend));
    BrainSystem brain(std::move(recognizer));
    brain.push({InputSource::HumanInterface, BrainInputType::Text, 1, now,
                std::string("青いボールを取って"), 1});
    const auto cycle = brain.runOnce(now);
    const auto goal = brain.goals().get(1);
    t.expect(cycle.planId && cycle.actionResults == 1 && goal &&
                 goal->type == "AcquireObject" && goal->target &&
                 std::get<std::string>(goal->completionCondition.arguments.at(
                     "target_name")) == "blue_ball" &&
                 goal->status == GoalStatus::Achieved,
             "CTX-MODEL-040",
             "model result reaches GoalManager and RuleBasedPlanner");
    t.expect(brain.world().snapshot().conditions.size() == 1 &&
                 brain.constraints().active().size() == 1,
             "CTX-MODEL-041",
             "validated conditions and constraints reach their managers");
    return t.finish();
}
