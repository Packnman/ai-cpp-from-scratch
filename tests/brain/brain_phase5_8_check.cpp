#include "brain/BrainSystem.hpp"
#include <cstdlib>
#include <iostream>
using namespace ai::brain;
namespace {
int n;
template <class T> void check(T &&v, const char *m) {
    ++n;
    if (!static_cast<bool>(v)) {
        std::cerr << "FAILED: " << m << '\n';
        std::exit(1);
    }
}
} // namespace
int main() {
    auto now = steady_now();
    auto async = std::make_shared<MockControlDispatcher>(false);
    ExecutionManager exec(async);
    Action a{1, "move"};
    a.timeout = Duration{10};
    a.resources = {{1, ResourceAccess::Exclusive}};
    ActionPlan p{1, 1, {{a, {}}}};
    check(exec.submit(p), "submit");
    check(exec.tick(now).empty() && exec.status() == ExecutionStatus::Running,
          "dispatch running");
    auto timed = exec.tick(now + Duration{11});
    check(timed.size() == 1 && timed[0].result == ActionResultCode::Timeout,
          "timeout");
    FakeExternalAI external(Duration{20});
    ExternalAIRequest req;
    req.createdAt = now;
    req.timeout = Duration{10};
    req.payload = std::string("x");
    req.goalId = 1;
    auto rid = external.submit(req);
    check(rid && external.poll(rid, now + Duration{11})->status ==
                     ResponseStatus::Timeout,
          "external timeout");
    Preprocessor pre(std::make_unique<FakeObjectDetector>(),
                     std::make_unique<FakeSpeechRecognizer>("goal:listen"),
                     std::make_unique<RuleContextRecognizer>());
    BrainInput voice{1,
                     InputSource::HumanInterface,
                     now,
                     InputStatus::Valid,
                     BrainInputType::Voice,
                     ByteBuffer{1}};
    auto first = pre.process(voice), second = pre.process(voice);
    check(first.size() == 2 && second.size() == 2 &&
              first[0].attributes == second[0].attributes,
          "recognition deterministic");
    BrainSystem brain;
    ExternalMessage message{InputSource::HumanInterface,
                            BrainInputType::Text,
                            1,
                            now,
                            std::string("goal:inspect"),
                            1};
    check(brain.push(message), "brain input");
    auto cycle = brain.runOnce(now);
    check(cycle.inputs == 1 && cycle.semantics == 2 && cycle.planId &&
              cycle.actionResults == 1,
          "end to end");
    check(!brain.goals().activeGoal(), "goal completed");
    check(!brain.memory().recall({MemoryType::ActionOutcome}).empty(),
          "postprocess memory");
    std::cout << "brain_phase5_8_check: " << n << " checks passed\n";
}
