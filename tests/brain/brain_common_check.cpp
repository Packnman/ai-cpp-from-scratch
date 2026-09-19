#include "brain/common/Dtos.hpp"
#include "brain/common/Error.hpp"
#include "brain/common/Result.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace ai::brain;
namespace {
int checks = 0;
#define CHECK(x)                                                               \
    do {                                                                       \
        ++checks;                                                              \
        if (!(x))                                                              \
            throw std::runtime_error(std::string("check failed: ") + #x);      \
    } while (false)

void identifiers_and_time() {
    static_assert(std::is_same_v<SemanticId, std::uint64_t>);
    static_assert(
        std::is_same_v<TimePoint, std::chrono::steady_clock::time_point>);
    CHECK(!valid_id(SemanticId{}));
    CHECK(valid_id(SemanticId{1}));
    const TimePoint start{};
    const auto finish = start + Duration{25};
    CHECK(elapsed(start, finish) == Duration{25});
    CHECK(expired(finish, finish));
    CHECK(!expired(start, finish));
}

void attributes_and_semantics() {
    AttributeMap attributes{{"name", std::string("blue ball")},
                            {"visible", true},
                            {"count", std::int64_t{2}}};
    CHECK(std::get<std::string>(attributes.at("name")) == "blue ball");
    CHECK(std::get<bool>(attributes.at("visible")));
    SemanticItem item{1,           SemanticType::Perception,
                      TimePoint{}, 0.75F,
                      true,        std::nullopt,
                      attributes};
    CHECK(valid_semantic_item(item));
    item.confidence = std::nanf("");
    CHECK(!valid_semantic_item(item));
    item.confidence = 1.01F;
    CHECK(!valid_semantic_item(item));
}

void errors_and_results() {
    BrainError error{7,
                     ErrorLevel::Error,
                     2,
                     TimePoint{},
                     "recognition failed",
                     "invalid payload",
                     RecoveryAction::Fallback};
    CHECK(valid_error(error));
    auto success = Result<std::string>::success("ok");
    CHECK(success);
    CHECK(success.value() == "ok");
    auto failure = Result<std::string>::failure(error);
    CHECK(!failure);
    CHECK(failure.error() == error);
    bool threw = false;
    try {
        (void)failure.value();
    } catch (const std::logic_error &) {
        threw = true;
    }
    CHECK(threw);
    CHECK(Result<void>::success());
    CHECK(!Result<void>::failure(error));
}

void inputs_goals_and_constraints() {
    BrainInput input{1,
                     InputSource::HumanInterface,
                     TimePoint{},
                     InputStatus::Valid,
                     BrainInputType::Text,
                     std::string("pick blue ball")};
    CHECK(valid_brain_input(input));
    input.status = InputStatus::Invalid;
    CHECK(!valid_brain_input(input));

    Goal goal{1,
              "AcquireObject",
              SemanticId{10},
              5,
              {"holding(target)", {}},
              GoalStatus::Pending,
              GoalSource::Human,
              std::nullopt,
              TimePoint{},
              TimePoint{}};
    CHECK(valid_goal(goal));
    goal.parentGoal = goal.id;
    CHECK(!valid_goal(goal));

    Constraint global{1,
                      "Safety",
                      true,
                      true,
                      {ConstraintScopeType::Global, std::nullopt},
                      {"safety_level < EmergencyStop", {}},
                      ConstraintSource::Safety,
                      std::nullopt,
                      TimePoint{}};
    CHECK(valid_constraint(global));
    global.scope.targetId = 3;
    CHECK(!valid_constraint(global));
}

void plans_and_results() {
    Action locate{1,
                  "Locate",
                  SemanticId{10},
                  {},
                  {},
                  {{"visible(target)", {}}},
                  {},
                  Duration{1000},
                  1,
                  {{1, ResourceAccess::Shared}}};
    Action grasp{2,
                 "Grasp",
                 SemanticId{10},
                 {},
                 {{"visible(target)", {}}},
                 {{"holding(target)", {}}},
                 {},
                 Duration{2000},
                 1,
                 {{2, ResourceAccess::Exclusive}}};
    CHECK(valid_action(locate));
    ActionPlan plan{1,
                    1,
                    {{locate, {}}, {grasp, {locate.id}}},
                    {1},
                    PlanStatus::Pending,
                    TimePoint{},
                    4,
                    1};
    CHECK(valid_action_plan(plan));
    plan.actions[1].dependencies = {99};
    CHECK(!valid_action_plan(plan));

    ActionResult result{1,           ActionResultCode::Succeeded, {},
                        TimePoint{}, TimePoint{} + Duration{1},   {}};
    CHECK(valid_action_result(result));
    result.endTime = TimePoint{} - Duration{1};
    CHECK(!valid_action_result(result));
}
} // namespace

int main() {
    try {
        identifiers_and_time();
        attributes_and_semantics();
        errors_and_results();
        inputs_goals_and_constraints();
        plans_and_results();
        std::cout << "brain_common_check: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
