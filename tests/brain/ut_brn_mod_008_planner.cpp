#include "TestSupport.hpp"
#include "brain/planning/Planner.hpp"

using namespace ai::brain;

namespace {
Policy policy(std::vector<Action> actions) {
    return {1, "test", "goal", {}, std::move(actions), 5};
}
PlanningContext context(std::vector<Action> actions) {
    Goal goal{1, "goal"};
    return {goal, {}, {}, {}, {policy(std::move(actions))}, {}};
}
Action action(std::string type) {
    Action value;
    value.type = std::move(type);
    value.timeout = Duration{10};
    return value;
}
} // namespace

int main() {
    brain_test::Suite t{"UT-BRN-MOD-008"};
    ConstraintManager constraints;
    RuleBasedPlanner planner(constraints);
    auto simple = planner.plan(context({action("locate")}));
    t.expect(simple && simple.value().plan.actions.size() == 1, "UT-008-001",
             "simple plan generated");
    auto multi = planner.plan(context({action("reach"), action("grasp")}));
    t.expect(multi && multi.value().plan.actions.size() == 2, "UT-008-002",
             "multi-action plan generated");
    t.expect(multi.value().plan.actions[1].dependencies ==
                 std::vector<ActionId>{multi.value().plan.actions[0].action.id},
             "UT-008-003", "dependency DAG generated");
    auto guarded = action("guarded");
    guarded.preconditions.push_back(
        {"condition_active", {{"id", std::uint64_t{99}}}});
    t.expect(!planner.plan(context({guarded})), "UT-008-004",
             "missing precondition rejected");
    Constraint hard{1,
                    "hard",
                    true,
                    true,
                    {},
                    {"deny_action_type", {{"type", std::string("move")}}},
                    ConstraintSource::Safety,
                    {},
                    steady_now()};
    constraints.add(hard);
    t.expect(!planner.plan(context({action("move")})), "UT-008-005",
             "hard constraint rejects plan");
    ConstraintManager softConstraints;
    auto soft = hard;
    soft.id = 2;
    soft.critical = false;
    softConstraints.add(soft);
    RuleBasedPlanner softPlanner(softConstraints);
    auto penalized = softPlanner.plan(context({action("move")}));
    t.expect(penalized && penalized.value().score < 5.5, "UT-008-006",
             "soft constraint penalizes score");
    auto resources = context({action("a"), action("b")});
    resources.policies[0].actionTemplate[0].resources = {
        {1, ResourceAccess::Exclusive}};
    resources.policies[0].actionTemplate[1].resources = {
        {1, ResourceAccess::Exclusive}};
    auto serialized = softPlanner.plan(resources);
    t.expect(serialized &&
                 !serialized.value().plan.actions[1].dependencies.empty(),
             "UT-008-007", "template actions serialized");
    Action first{10, "a"};
    first.timeout = Duration{1};
    Action second{11, "b"};
    second.timeout = Duration{1};
    ActionPlan cyclic{9, 1, {{first, {11}}, {second, {10}}}};
    t.expect(!PlanValidator::validate(cyclic, softConstraints), "UT-008-008",
             "cycle rejected");
    auto replanned = softPlanner.plan(context({action("wait")}));
    t.expect(replanned &&
                 replanned.value().plan.id != penalized.value().plan.id,
             "UT-008-009", "replanning creates new plan version identity");
    TransformerPlanner transformer;
    t.expect(!transformer.plan(context({action("local")})) &&
                 softPlanner.plan(context({action("local")})),
             "UT-008-010", "unavailable transformer permits local fallback");
    auto a = softPlanner.plan(context({action("same")}));
    auto b = softPlanner.plan(context({action("same")}));
    t.expect(a.value().plan.actions[0].action.type ==
                 b.value().plan.actions[0].action.type,
             "UT-008-011", "deterministic plan contents");
    return t.finish();
}
