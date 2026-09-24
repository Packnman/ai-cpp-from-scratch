#include "simulation/HumanoidDemo.hpp"

#include "brain/preprocess/Preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ai::simulation::demo {
namespace {

using namespace ai::brain;
using namespace ai::actuator;

/// Reads a string action parameter without implicit type conversion.
std::optional<std::string> stringParameter(const AttributeMap &parameters,
                                           const std::string &name) {
    const auto found = parameters.find(name);
    if (found == parameters.end() ||
        !std::holds_alternative<std::string>(found->second))
        return {};
    return std::get<std::string>(found->second);
}

/// Reads a numeric action parameter from the supported scalar DTO types.
std::optional<double> numberParameter(const AttributeMap &parameters,
                                      const std::string &name) {
    const auto found = parameters.find(name);
    if (found == parameters.end())
        return {};
    if (std::holds_alternative<double>(found->second))
        return std::get<double>(found->second);
    if (std::holds_alternative<std::int64_t>(found->second))
        return static_cast<double>(std::get<std::int64_t>(found->second));
    if (std::holds_alternative<std::uint64_t>(found->second))
        return static_cast<double>(std::get<std::uint64_t>(found->second));
    return {};
}

/// Reports whether a UTF-8 command contains one of the accepted phrases.
bool containsAny(const std::string &text,
                 std::initializer_list<std::string_view> phrases) {
    return std::ranges::any_of(phrases, [&](std::string_view phrase) {
        return text.find(phrase) != std::string::npos;
    });
}

/// Creates one joint-position action for a predefined humanoid skill.
Action jointTargetAction(double position) {
    Action action;
    action.type = "set_joint_target";
    action.parameters = {{"joint", std::string{"right_elbow"}},
                         {"position", position}};
    action.timeout = Duration{5'000};
    action.priority = 10;
    action.resources = {{1, ResourceAccess::Exclusive}};
    return action;
}

/// Finds one named joint in an engine-independent Plant snapshot.
const JointState &joint(const RobotPlantState &state, std::string_view id) {
    const auto found =
        std::ranges::find_if(state.joints, [&](const JointState &candidate) {
            return candidate.id == id;
        });
    if (found == state.joints.end())
        throw std::runtime_error("humanoid demo joint is missing: " +
                                 std::string{id});
    return *found;
}

/// Returns the prototype descriptor paired with default elbow dynamics.
ActuatorDescriptor rightElbowDescriptor() {
    ActuatorDescriptor descriptor;
    descriptor.actuatorId = "right_elbow_biceps";
    descriptor.componentId = std::string{actuator::sim::muscleComponentId};
    descriptor.type = ActuatorType::Muscle;
    descriptor.positionLimit = {-2.4, 0.0};
    descriptor.velocityLimit = {-10.0, 10.0};
    descriptor.torqueLimit = {-1.0, 1.0};
    descriptor.currentLimit = {-0.7, 0.7};
    descriptor.temperatureLimit = {-30.0, 125.0};
    descriptor.supportedModes = {ControlMode::Position, ControlMode::Torque,
                                 ControlMode::Stop, ControlMode::Disable};
    return descriptor;
}

} // namespace

ContextRecognitionResult
HumanoidCommandRecognizer::recognize(const std::string &text,
                                     TimePoint timestamp) {
    ContextRecognitionResult result;
    result.intent = "statement";
    result.confidence = 1.0F;
    const bool rightArm =
        containsAny(text, {"右手", "右腕", "right hand", "right arm"});
    const bool raise = containsAny(text, {"上げ", "あげ", "raise", "lift"});
    const bool lower =
        containsAny(text, {"下げ", "おろ", "戻", "lower", "down"});
    if (!rightArm || (!raise && !lower))
        return result;

    Goal goal;
    goal.type = raise ? "RaiseRightArm" : "LowerRightArm";
    goal.priority = 10;
    goal.source = GoalSource::Human;
    goal.createdAt = timestamp;
    goal.updatedAt = timestamp;
    result.intent = "command";
    result.goals.push_back(std::move(goal));
    return result;
}

std::optional<ActionResult>
JointTargetDispatcher::dispatch(const ActionCommand &command) {
    const auto jointId = stringParameter(command.parameters, "joint");
    const auto position = numberParameter(command.parameters, "position");
    if (command.type != "set_joint_target" || !jointId || jointId->empty() ||
        !position || !std::isfinite(*position)) {
        const auto now = steady_now();
        return ActionResult{command.id,
                            ActionResultCode::Failed,
                            "invalid joint target",
                            now,
                            now,
                            {}};
    }
    {
        std::scoped_lock lock(_mutex);
        _target = JointTarget{command.id, *jointId, *position};
    }
    // Phase 1 acknowledges target acceptance. Physical completion feedback is
    // intentionally a later Control-layer milestone.
    const auto now = steady_now();
    return ActionResult{command.id,
                        ActionResultCode::Succeeded,
                        "joint target accepted",
                        now,
                        now,
                        {}};
}

void JointTargetDispatcher::cancel(ActionId actionId) {
    std::scoped_lock lock(_mutex);
    if (_target && _target->actionId == actionId)
        _target.reset();
}

std::optional<JointTarget> JointTargetDispatcher::target() const {
    std::scoped_lock lock(_mutex);
    return _target;
}

HumanoidBrainBridge::HumanoidBrainBridge(
    std::shared_ptr<JointTargetDispatcher> dispatcher)
    : _preprocessor(std::make_unique<FakeObjectDetector>(),
                    std::make_unique<FakeSpeechRecognizer>(),
                    std::make_unique<HumanoidCommandRecognizer>()),
      _planner(_constraints), _dispatcher(std::move(dispatcher)),
      _execution(_dispatcher) {
    if (!_dispatcher)
        throw std::invalid_argument("humanoid dispatcher is null");

    Policy raise;
    raise.id = 1001;
    raise.name = "raise right arm";
    raise.goalType = "RaiseRightArm";
    raise.actionTemplate = {jointTargetAction(-1.35)};
    raise.priority = 100;
    raise.successRate = 1.0F;
    if (!_policies.add(raise))
        throw std::logic_error("failed to register raise-arm policy");

    Policy lower = raise;
    lower.id = 1002;
    lower.name = "lower right arm";
    lower.goalType = "LowerRightArm";
    lower.actionTemplate = {jointTargetAction(0.0)};
    if (!_policies.add(lower))
        throw std::logic_error("failed to register lower-arm policy");
}

BrainCommandResult HumanoidBrainBridge::issue(const std::string &text,
                                              TimePoint now) {
    BrainCommandResult result;
    BrainInput input{_nextGoalId,        InputSource::HumanInterface, now,
                     InputStatus::Valid, BrainInputType::Text,        text};
    const auto semantics = _preprocessor.process(input);
    for (const auto &semantic : semantics) {
        if (semantic.type != SemanticType::Goal)
            continue;
        const auto type = semantic.attributes.find("goal_type");
        if (type == semantic.attributes.end() ||
            !std::holds_alternative<std::string>(type->second))
            continue;
        Goal goal;
        goal.id = _nextGoalId++;
        goal.type = std::get<std::string>(type->second);
        goal.priority = 10;
        goal.source = GoalSource::Human;
        goal.createdAt = now;
        goal.updatedAt = now;
        result.recognized = true;
        result.goalType = goal.type;

        auto policies = _policies.findApplicable(goal, _world);
        auto planned = _planner.plan(
            {goal, _world, _constraints.active(), {}, policies, {}});
        if (!planned)
            return result;
        result.planned = true;
        if (!_execution.submit(planned.value().plan))
            return result;
        const auto actions = _execution.tick(now);
        result.dispatched = !actions.empty() && actions.front().result ==
                                                    ActionResultCode::Succeeded;
        return result;
    }
    return result;
}

HumanoidDemo::HumanoidDemo(std::filesystem::path modelPath)
    : _dispatcher(std::make_shared<JointTargetDispatcher>()),
      _brain(_dispatcher),
      _manager(
          [&] {
              auto plant = std::make_unique<MuJoCoPlant>(std::move(modelPath));
              _plant = plant.get();
              return plant;
          }(),
          {0.001, 0.001, 0.01, 60.0}) {
    auto descriptor = rightElbowDescriptor();
    auto driver = std::make_unique<actuator::sim::SimActuatorDriver>(
        actuator::sim::defaultElbowMuscleConfig());
    _rightElbowDriver = driver.get();
    driver->configure(descriptor);
    _actuators.registerActuator(descriptor, std::move(driver));
    _actuators.enable(descriptor.actuatorId);
    _adapter.bind({descriptor.actuatorId, "right_elbow", {}, {}, {}, {}});

    // Control converts the Brain target into a bounded driver command at
    // 100 Hz. Physics and component dynamics continue at 1 kHz.
    _manager.setControlCallback(
        [this](double, const RobotPlantState &, IPlant &) {
            const auto requested = _dispatcher->target();
            if (!requested)
                return;
            _actuators.command({_nextCommandId++,
                                "right_elbow_biceps",
                                actuator::ControlMode::Position,
                                requested->position,
                                {},
                                actuator::TimePoint{},
                                actuator::Duration{100}});
        });
    _manager.setActuatorCallback([this](double dt, const RobotPlantState &state,
                                        IPlant &plant) {
        const auto &elbow = joint(state, "right_elbow");
        _rightElbowDriver->advance(dt,
                                   {elbow.position, elbow.velocity, 0.0, true});
        _adapter.apply("right_elbow_biceps", _rightElbowDriver->plantOutput(),
                       plant);
    });
}

void HumanoidDemo::initialize() { _manager.initialize(); }

BrainCommandResult HumanoidDemo::command(const std::string &text) {
    return _brain.issue(text);
}

void HumanoidDemo::runSteps(std::size_t count) { _manager.runSteps(count); }

double HumanoidDemo::rightElbowPosition() const {
    return joint(_manager.plant().getState(), "right_elbow").position;
}

std::optional<JointTarget> HumanoidDemo::target() const {
    return _dispatcher->target();
}

} // namespace ai::simulation::demo
