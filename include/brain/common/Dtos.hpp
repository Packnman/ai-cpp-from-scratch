#pragma once

#include "brain/common/Attribute.hpp"
#include "brain/common/Semantic.hpp"
#include "brain/common/Time.hpp"
#include "brain/common/Types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai::brain {

enum class InputSource {
    Sensor,
    HumanInterface,
    Control,
    Safety,
    Communication,
    ExternalAI
};
enum class InputStatus { Valid, Warning, Invalid, Timeout, Stale };
enum class BrainInputType {
    CameraFrame,
    Imu,
    Force,
    Touch,
    Distance,
    Voice,
    Text,
    RobotState,
    ActionResult,
    SafetyState,
    Constraint,
    StopRequest,
    NetworkState,
    ExternalAIResult
};

struct BrainInput {
        std::uint64_t id{};
        InputSource source{InputSource::Sensor};
        TimePoint timestamp{};
        InputStatus status{InputStatus::Invalid};
        BrainInputType type{BrainInputType::Text};
        Payload payload;

        bool operator==(const BrainInput &) const = default;
};

struct ConditionExpression {
        std::string expression;
        AttributeMap arguments;

        bool operator==(const ConditionExpression &) const = default;
};
using ConstraintExpression = ConditionExpression;

struct Condition {
        ConditionId id{};
        std::string type;
        bool active{true};
        float confidence{};
        TimePoint timestamp{};
        AttributeMap attributes;

        bool operator==(const Condition &) const = default;
};

using GoalType = std::string;
enum class GoalStatus {
    Pending,
    Active,
    Suspended,
    Achieved,
    Failed,
    Cancelled
};
enum class GoalSource { Human, Internal, Safety, Scheduled, SubGoal };

struct Goal {
        GoalId id{};
        GoalType type;
        std::optional<SemanticId> target;
        int priority{};
        ConditionExpression completionCondition;
        GoalStatus status{GoalStatus::Pending};
        GoalSource source{GoalSource::Internal};
        std::optional<GoalId> parentGoal;
        TimePoint createdAt{};
        TimePoint updatedAt{};

        bool operator==(const Goal &) const = default;
};

using ConstraintType = std::string;
enum class ConstraintScopeType { Global, Goal, Plan, Action, Resource, Entity };
enum class ConstraintSource {
    Safety,
    SystemRule,
    HumanExplicit,
    Environment,
    Policy
};

struct ConstraintScope {
        ConstraintScopeType type{ConstraintScopeType::Global};
        std::optional<std::uint64_t> targetId;

        bool operator==(const ConstraintScope &) const = default;
};

struct Constraint {
        ConstraintId id{};
        ConstraintType type;
        bool critical{};
        bool active{true};
        ConstraintScope scope;
        ConstraintExpression expression;
        ConstraintSource source{ConstraintSource::Environment};
        std::optional<TimePoint> expiresAt;
        TimePoint timestamp{};

        bool operator==(const Constraint &) const = default;
};

struct Perception {
        SemanticId id{};
        std::string type;
        std::optional<TrackingId> trackingId;
        float confidence{};
        TimePoint timestamp{};
        bool valid{true};
        AttributeMap attributes;

        bool operator==(const Perception &) const = default;
};

struct RobotState {
        TimePoint timestamp{};
        bool valid{};
        AttributeMap attributes;

        bool operator==(const RobotState &) const = default;
};
struct CommunicationState {
        TimePoint timestamp{};
        bool online{};
        bool degraded{};
        AttributeMap attributes;

        bool operator==(const CommunicationState &) const = default;
};
enum class SafetyLevel { Normal, Warning, Limited, SafeStop, EmergencyStop };
struct SafetyState {
        TimePoint timestamp{};
        SafetyLevel level{SafetyLevel::SafeStop};
        bool valid{};
        AttributeMap attributes;

        bool operator==(const SafetyState &) const = default;
};

struct WorldState {
        std::uint64_t version{};
        TimePoint timestamp{};
        std::unordered_map<SemanticId, Perception> perceptions;
        RobotState robotState;
        std::unordered_map<ConditionId, Condition> conditions;
        CommunicationState communicationState;
        SafetyState safetyState;

        bool operator==(const WorldState &) const = default;
};

using ActionType = std::string;
enum class ResourceAccess { Shared, Exclusive };
struct ResourceRequest {
        ResourceId id{};
        ResourceAccess access{ResourceAccess::Exclusive};

        bool operator==(const ResourceRequest &) const = default;
};

struct Action {
        ActionId id{};
        ActionType type;
        std::optional<SemanticId> target;
        AttributeMap parameters;
        std::vector<ConditionExpression> preconditions;
        std::vector<ConditionExpression> completionConditions;
        std::vector<ConditionExpression> failureConditions;
        Duration timeout{};
        int priority{};
        std::vector<ResourceRequest> resources;

        bool operator==(const Action &) const = default;
};

enum class PlanStatus { Pending, Active, Succeeded, Failed, Cancelled };
struct ActionNode {
        Action action;
        std::vector<ActionId> dependencies;

        bool operator==(const ActionNode &) const = default;
};
struct ActionPlan {
        PlanId id{};
        GoalId goalId{};
        std::vector<ActionNode> actions;
        std::vector<ConstraintId> constraints;
        PlanStatus status{PlanStatus::Pending};
        TimePoint createdAt{};
        std::uint64_t worldStateVersion{};
        std::uint32_t version{};

        bool operator==(const ActionPlan &) const = default;
};

enum class ActionResultCode { Succeeded, Failed, Cancelled, Timeout };
struct ActionResult {
        ActionId actionId{};
        ActionResultCode result{ActionResultCode::Failed};
        std::string reason;
        TimePoint startTime{};
        TimePoint endTime{};
        AttributeMap observedState;

        bool operator==(const ActionResult &) const = default;
};

bool valid_brain_input(const BrainInput &input) noexcept;
bool valid_goal(const Goal &goal) noexcept;
bool valid_constraint(const Constraint &constraint) noexcept;
bool valid_action(const Action &action) noexcept;
bool valid_action_plan(const ActionPlan &plan) noexcept;
bool valid_action_result(const ActionResult &result) noexcept;

} // namespace ai::brain
