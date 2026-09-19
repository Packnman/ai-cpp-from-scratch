#include "brain/memory/MemoryManager.hpp"
#include "brain/planning/Planner.hpp"
#include <cstdlib>
#include <iostream>
using namespace ai::brain;
namespace {
int n;
void check(bool v, const char *m) {
    ++n;
    if (!v) {
        std::cerr << "FAILED: " << m << '\n';
        std::exit(1);
    }
}
} // namespace
int main() {
    auto backend = std::make_unique<SQLiteMemoryBackend>(":memory:");
    check(backend->available(), "sqlite available");
    MemoryManager memory(std::move(backend));
    MemoryItem item;
    item.type = MemoryType::Knowledge;
    item.content = {
        1,  SemanticType::Conversation,          steady_now(), 1, true,
        {}, {{"text", std::string("blue ball")}}};
    item.importance = .8F;
    item.confidence = .9F;
    item.tags = {"object"};
    auto id = memory.remember(item);
    auto found = memory.recall({MemoryType::Knowledge, {"object"}, {}, 10});
    check(id && found.size() == 1, "sqlite store/search");
    PolicyManager policies;
    Action a;
    a.type = "locate";
    a.timeout = Duration{100};
    Policy p{1, "locate", "inspect", {}, {a}, 10};
    check(policies.add(p), "policy add");
    Goal g{1, "inspect"};
    auto applicable = policies.findApplicable(g, {});
    check(applicable.size() == 1, "policy lookup");
    ConstraintManager constraints;
    RuleBasedPlanner planner(constraints);
    PlanningContext c{g, {}, {}, {}, applicable, {}};
    auto plan = planner.plan(c);
    check(plan && plan.value().plan.actions.size() == 1, "plan");
    Constraint deny{1,
                    "safety",
                    true,
                    true,
                    {},
                    {"deny_action_type", {{"type", std::string("locate")}}},
                    ConstraintSource::Safety,
                    {},
                    steady_now()};
    check(constraints.add(deny), "deny add");
    check(!planner.plan(c), "constraint rejection");
    TransformerPlanner transformer;
    check(!transformer.modelLoaded() && !transformer.plan(c),
          "unloaded transformer fails");
    std::cout << "brain_phase3_4_check: " << n << " checks passed\n";
}
