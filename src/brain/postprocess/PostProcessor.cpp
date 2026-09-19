#include "brain/postprocess/PostProcessor.hpp"
namespace ai::brain {
PostProcessor::PostProcessor(IGoalManager &g, IMemoryManager &m,
                             IPolicyManager &p)
    : _goals(g), _memory(m), _policies(p) {}
PostProcessResult PostProcessor::process(const PostProcessInput &i) {
    PostProcessResult o;
    bool ok = i.actionResult.result == ActionResultCode::Succeeded;
    o.replanRequired = !ok;
    if (i.planComplete || !ok)
        o.goalUpdated = _goals.complete(i.goal.id, ok);
    MemoryItem mem;
    mem.type = MemoryType::ActionOutcome;
    mem.content = {i.actionResult.actionId,
                   SemanticType::ActionResult,
                   i.actionResult.endTime,
                   1.F,
                   true,
                   i.goal.target,
                   {{"result", std::string(ok ? "succeeded" : "failed")},
                    {"reason", i.actionResult.reason}}};
    mem.importance = ok ? .6F : .9F;
    mem.confidence = 1;
    mem.tags = {"action_result", ok ? "success" : "failure"};
    o.memoryId = _memory.remember(mem);
    if (i.policyId)
        o.policyUpdated = _policies.recordResult(*i.policyId, ok);
    const Action *action = nullptr;
    for (const auto &n : i.plan.actions)
        if (n.action.id == i.actionResult.actionId)
            action = &n.action;
    if (action)
        o.sample = {PlanningContext{i.goal, i.worldState, {}, {}, {}, {}},
                    {i.goal.id},
                    i.plan.id,
                    *action,
                    i.actionResult,
                    ok ? 1.F : -1.F,
                    steady_now()};
    return o;
}
} // namespace ai::brain
