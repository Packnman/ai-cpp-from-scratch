#include "simulation/HumanoidDemo.hpp"

#include "brain/preprocess/Preprocessor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ai::simulation::demo {
namespace {

using namespace ai::brain;
using namespace ai::actuator;
using namespace ai::actuator::sim;

constexpr std::array<std::string_view, 17> nativeMuscleNames{
    "right_pectoralis_upper_actuator",
    "right_pectoralis_middle_actuator",
    "right_pectoralis_lower_actuator",
    "right_deltoid_anterior_actuator",
    "right_deltoid_lateral_actuator",
    "right_deltoid_posterior_actuator",
    "right_serratus_upper_actuator",
    "right_serratus_middle_actuator",
    "right_serratus_lower_actuator",
    "right_trapezius_upper_actuator",
    "right_trapezius_middle_actuator",
    "right_trapezius_lower_actuator",
    "right_latissimus_upper_actuator",
    "right_latissimus_middle_actuator",
    "right_latissimus_lower_actuator",
    "right_biceps_actuator",
    "right_triceps_actuator",
};

/// Reads one entry from MuJoCo's sparse actuator-moment matrix.
double actuatorMoment(const mjData *data, int actuatorId, int dofAddress) {
    const int start = data->moment_rowadr[actuatorId];
    const int count = data->moment_rownnz[actuatorId];
    for (int offset = 0; offset < count; ++offset) {
        const int index = start + offset;
        if (data->moment_colind[index] == dofAddress)
            return data->actuator_moment[index];
    }
    return 0.0;
}

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
Action jointTargetAction(std::string joint, double position,
                         ResourceId resource) {
    Action action;
    action.type = "set_joint_target";
    action.parameters = {{"joint", std::move(joint)}, {"position", position}};
    action.timeout = Duration{5'000};
    action.priority = 10;
    action.resources = {{resource, ResourceAccess::Exclusive}};
    return action;
}

/// Creates the five-joint pose used by the Prototype-1 right-arm demo.
std::vector<Action> rightArmPose(bool raised) {
    return {
        jointTargetAction("right_scapula_rotation", raised ? 0.20 : 0.0, 1),
        jointTargetAction("right_shoulder_abduction", raised ? -2.95 : 0.0, 2),
        jointTargetAction("right_shoulder_flexion", 0.0, 3),
        jointTargetAction("right_shoulder_rotation", 0.0, 4),
        jointTargetAction("right_elbow", raised ? -0.10 : 0.0, 5),
    };
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
ActuatorDescriptor muscleDescriptor(std::string id, Range positionLimit,
                                    double torqueLimit) {
    ActuatorDescriptor descriptor;
    descriptor.actuatorId = std::move(id);
    descriptor.componentId = std::string{actuator::sim::muscleComponentId};
    descriptor.type = ActuatorType::Muscle;
    descriptor.positionLimit = positionLimit;
    descriptor.velocityLimit = {-6.0, 6.0};
    descriptor.torqueLimit = {-torqueLimit, torqueLimit};
    descriptor.currentLimit = {-5.0, 5.0};
    descriptor.temperatureLimit = {-30.0, 125.0};
    descriptor.supportedModes = {ControlMode::Position, ControlMode::Torque,
                                 ControlMode::Stop, ControlMode::Disable};
    return descriptor;
}

/// Builds a configurable provisional transmission around a documented motor
/// group. Geometry-dependent values remain explicit Prototype-1 assumptions.
SimActuatorDriverConfig muscleConfig(MotorPreset preset, std::size_t count,
                                     double momentArm, double positionGain) {
    auto config = defaultElbowMuscleConfig();
    config.muscle.motors.assign(count, motorParameters(preset));
    config.momentArm = momentArm;
    config.positionGain = positionGain;
    config.velocityGain = 2.0;
    return config;
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
        _targets[*jointId] = JointTarget{command.id, *jointId, *position};
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
    std::erase_if(_targets, [&](const auto &entry) {
        return entry.second.actionId == actionId;
    });
}

std::optional<JointTarget>
JointTargetDispatcher::target(std::string_view jointId) const {
    std::scoped_lock lock(_mutex);
    const auto found = _targets.find(std::string{jointId});
    return found == _targets.end() ? std::nullopt
                                   : std::optional<JointTarget>{found->second};
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
    raise.actionTemplate = rightArmPose(true);
    raise.priority = 100;
    raise.successRate = 1.0F;
    if (!_policies.add(raise))
        throw std::logic_error("failed to register raise-arm policy");

    Policy lower = raise;
    lower.id = 1002;
    lower.name = "lower right arm";
    lower.goalType = "LowerRightArm";
    lower.actionTemplate = rightArmPose(false);
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
        bool dispatchedAny{};
        while (_execution.status() == ExecutionStatus::Running) {
            const auto actions = _execution.tick(now);
            if (actions.empty())
                break;
            dispatchedAny = true;
            if (std::ranges::any_of(actions, [](const ActionResult &action) {
                    return action.result != ActionResultCode::Succeeded;
                }))
                break;
        }
        result.dispatched =
            dispatchedAny && _execution.status() == ExecutionStatus::Succeeded;
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
    const auto addMuscle = [this](std::string actuatorId, std::string jointId,
                                  Range range, MotorPreset preset,
                                  std::size_t motorCount, double momentArm,
                                  double positionGain, double torqueLimit) {
        auto descriptor = muscleDescriptor(actuatorId, range, torqueLimit);
        auto driver = std::make_unique<SimActuatorDriver>(
            muscleConfig(preset, motorCount, momentArm, positionGain));
        auto *driverView = driver.get();
        driver->configure(descriptor);
        _actuators.registerActuator(descriptor, std::move(driver));
        _actuators.enable(actuatorId);
        _adapter.bind({actuatorId, jointId, {}, {}, {}, {}});
        _controlledActuators.push_back(
            {std::move(actuatorId), std::move(jointId), driverView});
    };

    // Motor articles/counts follow the Prototype-1 upper-limb design. Moment
    // arms and gains are replaceable assumptions pending bracket/CAD data.
    addMuscle("right_scapula_serratus", "right_scapula_rotation", {-0.35, 0.35},
              MotorPreset::Ect35_80, 2, 0.025, 7.0, 10.0);
    addMuscle("right_scapula_trapezius", "right_scapula_rotation",
              {-0.35, 0.35}, MotorPreset::Ect35_80, 2, 0.025, 7.0, 10.0);
    addMuscle("right_shoulder_deltoid", "right_shoulder_abduction",
              {-3.10, 0.35}, MotorPreset::Ect35_80, 3, 0.035, 12.0, 18.0);
    addMuscle("right_shoulder_pectoralis", "right_shoulder_flexion",
              {-1.80, 1.80}, MotorPreset::Ect48_35, 2, 0.040, 8.0, 18.0);
    addMuscle("right_shoulder_latissimus", "right_shoulder_rotation",
              {-1.20, 1.20}, MotorPreset::Ect48_35, 2, 0.040, 8.0, 14.0);
    addMuscle("right_elbow_biceps", "right_elbow", {-2.40, 0.0},
              MotorPreset::Ect35_80, 2, 0.030, 8.0, 15.0);

    // Control converts the Brain target into a bounded driver command at
    // 100 Hz. Physics and component dynamics continue at 1 kHz.
    _manager.setControlCallback(
        [this](double, const RobotPlantState &, IPlant &) {
            if (_nativeTendonControl)
                return;
            for (const auto &actuator : _controlledActuators) {
                const auto requested = _dispatcher->target(actuator.jointId);
                if (!requested)
                    continue;
                _actuators.command({_nextCommandId++,
                                    actuator.actuatorId,
                                    actuator::ControlMode::Position,
                                    requested->position,
                                    {},
                                    actuator::TimePoint{},
                                    actuator::Duration{100}});
            }
        });
    _manager.setActuatorCallback(
        [this](double dt, const RobotPlantState &state, IPlant &plant) {
            if (_nativeTendonControl) {
                applyNativeTendonControl(state);
                return;
            }
            for (const auto &actuator : _controlledActuators) {
                const auto &feedback = joint(state, actuator.jointId);
                actuator.driver->advance(
                    dt, {feedback.position, feedback.velocity, 0.0, true});
                _adapter.apply(actuator.actuatorId,
                               actuator.driver->plantOutput(), plant);
            }
        });
}

void HumanoidDemo::initialize() {
    _manager.initialize();
    initializeNativeTendonControl();
}

void HumanoidDemo::initializeNativeTendonControl() {
    auto *model = _plant->model().model();
    const int first = mj_name2id(model, mjOBJ_ACTUATOR,
                                nativeMuscleNames.front().data());
    if (first < 0)
        return;

    _nativeMuscleIds.clear();
    _nativeMuscleIds.reserve(nativeMuscleNames.size());
    for (const auto name : nativeMuscleNames) {
        const int id = mj_name2id(model, mjOBJ_ACTUATOR, name.data());
        if (id < 0)
            throw std::runtime_error("native tendon actuator is missing: " +
                                     std::string{name});
        _nativeMuscleIds.push_back(id);
    }
    _nativeTendonControl = true;
}

void HumanoidDemo::applyNativeTendonControl(const RobotPlantState &state) {
    struct JointControl {
            std::string_view name;
            double kp;
            double kd;
            double maxTorque;
    };
    constexpr std::array<JointControl, 5> joints{{
        {"right_scapula_rotation", 8.0, 1.0, 10.0},
        {"right_shoulder_abduction", 12.0, 2.0, 18.0},
        {"right_shoulder_flexion", 8.0, 1.5, 14.0},
        {"right_shoulder_rotation", 8.0, 1.5, 12.0},
        {"right_elbow", 8.0, 1.2, 15.0},
    }};

    auto *model = _plant->model().model();
    auto *data = _plant->model().data();
    const std::size_t muscleCount = _nativeMuscleIds.size();
    std::array<double, joints.size()> desiredTorque{};
    std::vector<double> moment(joints.size() * muscleCount);

    for (std::size_t row = 0; row < joints.size(); ++row) {
        const auto requested = _dispatcher->target(joints[row].name);
        const auto &feedback = joint(state, joints[row].name);
        const double target = requested ? requested->position : 0.0;
        desiredTorque[row] = std::clamp(
            joints[row].kp * (target - feedback.position) -
                joints[row].kd * feedback.velocity,
            -joints[row].maxTorque, joints[row].maxTorque);
        const auto &binding = _plant->model().jointBinding(
            std::string{joints[row].name});
        for (std::size_t column = 0; column < muscleCount; ++column) {
            const int actuatorId = _nativeMuscleIds[column];
            moment[row * muscleCount + column] = actuatorMoment(
                data, actuatorId, binding.dofAddress);
        }
    }

    // Non-negative least squares by projected gradient descent:
    // minimize ||moment * force - desiredTorque|| while enforcing each
    // actuator's [0, maximum] pull-only control range.
    std::vector<double> force(muscleCount);
    constexpr double regularization = 1e-7;
    std::array<double, joints.size()> residual{};
    for (std::size_t row = 0; row < joints.size(); ++row)
        residual[row] = -desiredTorque[row];
    // Coordinate descent avoids a single large multi-joint moment arm making
    // the global step size too small for every other muscle.
    for (int iteration = 0; iteration < 80; ++iteration) {
        for (std::size_t column = 0; column < muscleCount; ++column) {
            double gradient = regularization * force[column];
            double curvature = regularization;
            for (std::size_t row = 0; row < joints.size(); ++row)
                gradient += moment[row * muscleCount + column] * residual[row];
            for (std::size_t row = 0; row < joints.size(); ++row) {
                const double arm = moment[row * muscleCount + column];
                curvature += arm * arm;
            }
            const int actuatorId = _nativeMuscleIds[column];
            const double maximum = model->actuator_ctrlrange[2 * actuatorId + 1];
            const double previous = force[column];
            force[column] = std::clamp(previous - gradient / curvature, 0.0,
                                       maximum);
            const double change = force[column] - previous;
            for (std::size_t row = 0; row < joints.size(); ++row)
                residual[row] +=
                    moment[row * muscleCount + column] * change;
        }
    }

    mju_zero(data->ctrl, model->nu);
    for (std::size_t column = 0; column < muscleCount; ++column)
        data->ctrl[_nativeMuscleIds[column]] = force[column];
}

BrainCommandResult HumanoidDemo::command(const std::string &text) {
    return _brain.issue(text);
}

void HumanoidDemo::runSteps(std::size_t count) { _manager.runSteps(count); }

void HumanoidDemo::replay() {
    _manager.reset();
    for (const auto &actuator : _controlledActuators)
        actuator.driver->resetFault();
}

double HumanoidDemo::rightElbowPosition() const {
    return jointPosition("right_elbow");
}

double HumanoidDemo::jointPosition(std::string_view jointId) const {
    return joint(_manager.plant().getState(), jointId).position;
}

std::optional<JointTarget> HumanoidDemo::target() const {
    return _dispatcher->target("right_elbow");
}

} // namespace ai::simulation::demo
