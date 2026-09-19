#include "brain/constraint/ConstraintManager.hpp"
#include "brain/goal/GoalManager.hpp"
#include "brain/input/InputAdapter.hpp"
#include "brain/world/WorldStateManager.hpp"
#include <cstdlib>
#include <iostream>
using namespace ai::brain;
namespace {
int checks;
void check(bool v, const char *m) {
    ++checks;
    if (!v) {
        std::cerr << "FAILED: " << m << '\n';
        std::exit(1);
    }
}
} // namespace
int main() {
    auto now = steady_now();
    InputAdapter a;
    check(bool(a.push({InputSource::Sensor, BrainInputType::Text, 1, now,
                       std::string("sensor"), 1})),
          "sensor push");
    check(bool(a.push({InputSource::Safety, BrainInputType::StopRequest, 1, now,
                       std::string("stop"), 2})),
          "safety push");
    auto in = a.poll();
    check(in.size() == 2 && in[0].source == InputSource::Safety, "priority");
    check(!a.push({InputSource::Sensor, BrainInputType::Text, 2, now,
                   std::string("bad"), 3}),
          "schema");
    WorldStateManager w;
    SemanticItem s{1,  SemanticType::Perception,       now, .9F, true,
                   {}, {{"name", std::string("ball")}}};
    check(w.update(s) && w.version() == 1, "world");
    s.timestamp = now - Duration{100};
    check(!w.update(s), "old");
    check(w.expire(now + Duration{1000}, Duration{500}) == 1, "stale");
    check(w.updateSafety({now, SafetyLevel::Normal, true, {}}) &&
              w.snapshot().safetyState.level == SafetyLevel::Normal,
          "safety state");
    GoalManager g;
    Goal low{0, "inspect", {}, 1};
    Goal safe{0, "stop", {}, 0};
    safe.source = GoalSource::Safety;
    auto lid = g.add(low), sid = g.add(safe);
    check(lid && sid && g.activeGoal()->id == sid, "preempt");
    check(g.complete(sid, true) && g.activeGoal()->id == lid, "resume");
    Action act{1, "move"};
    act.timeout = Duration{100};
    ActionPlan p{1, lid, {{act, {}}}};
    Constraint c{1,
                 "safety",
                 true,
                 true,
                 {},
                 {"deny_action_type", {{"type", std::string("move")}}},
                 ConstraintSource::Safety,
                 {},
                 now};
    ConstraintManager cm;
    check(cm.add(c), "constraint");
    auto e = cm.evaluate(p);
    check(!e.allowed && e.violated == std::vector<ConstraintId>{1}, "hard");
    std::cout << "brain_phase2_check: " << checks << " checks passed\n";
}
