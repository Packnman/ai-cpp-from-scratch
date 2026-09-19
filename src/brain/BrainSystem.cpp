#include "brain/BrainSystem.hpp"
namespace ai::brain {
namespace {
std::optional<std::string> string_attr(const AttributeMap &a,
                                       const std::string &k) {
    auto i = a.find(k);
    if (i != a.end() && std::holds_alternative<std::string>(i->second))
        return std::get<std::string>(i->second);
    return {};
}
int int_attr(const AttributeMap &a, const std::string &k) {
    auto i = a.find(k);
    return i != a.end() && std::holds_alternative<std::int64_t>(i->second)
               ? int(std::get<std::int64_t>(i->second))
               : 0;
}
bool bool_attr(const AttributeMap &a, const std::string &key, bool fallback) {
    const auto found = a.find(key);
    return found != a.end() && std::holds_alternative<bool>(found->second)
               ? std::get<bool>(found->second)
               : fallback;
}
SafetyLevel safety_level(const AttributeMap &attributes) {
    const auto level = string_attr(attributes, "level");
    if (!level)
        return SafetyLevel::SafeStop;
    if (*level == "normal")
        return SafetyLevel::Normal;
    if (*level == "warning")
        return SafetyLevel::Warning;
    if (*level == "limited")
        return SafetyLevel::Limited;
    if (*level == "emergency_stop")
        return SafetyLevel::EmergencyStop;
    return SafetyLevel::SafeStop;
}
} // namespace
BrainSystem::BrainSystem(std::string path)
    : _log(std::make_shared<InMemoryLogManager>()), _input({}, _log),
      _preprocessor(std::make_unique<FakeObjectDetector>(),
                    std::make_unique<FakeSpeechRecognizer>(),
                    std::make_unique<RuleContextRecognizer>()),
      _memory(std::make_unique<SQLiteMemoryBackend>(path)),
      _planner(_constraints),
      _control(std::make_shared<MockControlDispatcher>()), _execution(_control),
      _post(_goals, _memory, _policies) {
    Action action;
    action.type = "acknowledge";
    action.timeout = Duration{1000};
    Policy policy;
    policy.id = 1;
    policy.name = "default local policy";
    policy.goalType = "*";
    policy.actionTemplate = {action};
    _policies.add(policy);
}
Result<void> BrainSystem::push(const ExternalMessage &m) {
    return _input.push(m);
}
BrainCycleResult BrainSystem::runOnce(TimePoint now) {
    BrainCycleResult out;
    auto inputs = _input.poll();
    out.inputs = inputs.size();
    for (const auto &i : inputs) {
        if (i.type == BrainInputType::StopRequest) {
            _execution.emergencyStop();
            continue;
        }
        if (i.type == BrainInputType::SafetyState) {
            const auto attributes =
                std::holds_alternative<AttributeMap>(i.payload)
                    ? std::get<AttributeMap>(i.payload)
                    : AttributeMap{};
            _world.updateSafety(
                {i.timestamp, safety_level(attributes), true, attributes});
            continue;
        }
        if (i.type == BrainInputType::NetworkState) {
            const auto attributes =
                std::holds_alternative<AttributeMap>(i.payload)
                    ? std::get<AttributeMap>(i.payload)
                    : AttributeMap{};
            _world.updateCommunication(
                {i.timestamp, bool_attr(attributes, "online", false),
                 bool_attr(attributes, "degraded", false), attributes});
            continue;
        }
        auto semantic = _preprocessor.process(i);
        out.semantics += semantic.size();
        for (const auto &s : semantic) {
            if (s.type == SemanticType::Perception ||
                s.type == SemanticType::Condition ||
                s.type == SemanticType::RobotState)
                _world.update(s);
            else if (s.type == SemanticType::Goal) {
                auto type = string_attr(s.attributes, "goal_type");
                if (type) {
                    Goal g;
                    g.type = *type;
                    g.target = s.target;
                    g.priority = int_attr(s.attributes, "priority");
                    g.source = GoalSource::Human;
                    g.createdAt = s.timestamp;
                    g.updatedAt = s.timestamp;
                    _goals.add(g);
                }
            } else if (s.type == SemanticType::Conversation) {
                MemoryItem m;
                m.type = MemoryType::Conversation;
                m.content = s;
                m.importance = .4F;
                m.confidence = s.confidence;
                m.tags = {"conversation"};
                _memory.remember(m);
            }
        }
    }
    if (_execution.status() == ExecutionStatus::Idle ||
        _execution.status() == ExecutionStatus::Succeeded ||
        _execution.status() == ExecutionStatus::Failed ||
        _execution.status() == ExecutionStatus::Cancelled) {
        if (auto goal = _goals.activeGoal()) {
            auto policies = _policies.findApplicable(*goal, _world.snapshot());
            PlanningContext context{*goal,
                                    _world.snapshot(),
                                    _constraints.active(),
                                    _memory.recall({}),
                                    policies,
                                    {}};
            auto planned = _planner.plan(context);
            if (planned) {
                _currentPlan = planned.value().plan;
                _currentPolicy = policies.empty() ? std::optional<PolicyId>{}
                                                  : policies.front().id;
                out.planId = _currentPlan->id;
                auto submitted = _execution.submit(*_currentPlan);
                if (!submitted)
                    out.errors.push_back(submitted.error());
            } else
                out.errors.push_back(planned.error());
        }
    }
    if (_currentPlan && _execution.status() == ExecutionStatus::Running) {
        auto results = _execution.tick(now);
        out.actionResults = results.size();
        auto goal = _goals.get(_currentPlan->goalId);
        for (const auto &r : results)
            if (goal)
                _post.process(
                    {*goal, *_currentPlan, r, _world.snapshot(), _currentPolicy,
                     _execution.status() == ExecutionStatus::Succeeded});
    }
    return out;
}
} // namespace ai::brain
