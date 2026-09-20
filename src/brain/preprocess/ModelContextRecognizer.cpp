#include "brain/preprocess/ModelContextRecognizer.hpp"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ai::brain {
namespace {

using Json = nlohmann::json;

constexpr ModuleId context_module = 2;

bool known(const std::vector<std::string> &values, std::string_view value) {
    return std::ranges::find(values, value) != values.end();
}

void require_keys(const Json &value,
                  std::initializer_list<std::string_view> required,
                  std::initializer_list<std::string_view> optional = {}) {
    if (!value.is_object())
        throw std::invalid_argument("expected JSON object");
    std::set<std::string_view> allowed(required.begin(), required.end());
    allowed.insert(optional.begin(), optional.end());
    for (const auto &[key, ignored] : value.items()) {
        (void)ignored;
        if (!allowed.contains(key))
            throw std::invalid_argument("unknown field: " + key);
    }
    for (const auto key : required)
        if (!value.contains(std::string(key)))
            throw std::invalid_argument("missing field: " + std::string(key));
}

std::string bounded_string(const Json &value, std::string_view name,
                           std::size_t maximum, bool identifier = false) {
    if (!value.is_string())
        throw std::invalid_argument(std::string(name) + " must be a string");
    auto result = value.get<std::string>();
    if (result.empty() || result.size() > maximum)
        throw std::invalid_argument(std::string(name) + " has invalid length");
    if (identifier && !std::ranges::all_of(result, [](unsigned char ch) {
            return std::isalnum(ch) || ch == '_' || ch == '-' || ch == ':' ||
                   ch == '.';
        }))
        throw std::invalid_argument(std::string(name) +
                                    " is not an identifier");
    return result;
}

float confidence(const Json &value, std::string_view name) {
    if (!value.is_number())
        throw std::invalid_argument(std::string(name) + " must be numeric");
    const auto parsed = value.get<double>();
    if (!std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0)
        throw std::invalid_argument(std::string(name) + " is out of range");
    return static_cast<float>(parsed);
}

AttributeValue attribute_value(const Json &value, std::size_t maximum) {
    if (value.is_boolean())
        return value.get<bool>();
    if (value.is_number_unsigned())
        return value.get<std::uint64_t>();
    if (value.is_number_integer())
        return value.get<std::int64_t>();
    if (value.is_number_float()) {
        const auto number = value.get<double>();
        if (!std::isfinite(number))
            throw std::invalid_argument("attribute number is not finite");
        return number;
    }
    if (value.is_string())
        return bounded_string(value, "attribute", maximum);
    throw std::invalid_argument("attribute must be a scalar");
}

AttributeMap attributes(const Json &value,
                        const ModelContextRecognizerConfig &config) {
    if (!value.is_object() || value.size() > config.maxAttributes)
        throw std::invalid_argument("invalid attributes object");
    AttributeMap result;
    for (const auto &[key, item] : value.items()) {
        if (key.empty() || key.size() > config.maxStringBytes)
            throw std::invalid_argument("invalid attribute key");
        result.emplace(key, attribute_value(item, config.maxStringBytes));
    }
    return result;
}

SemanticId target_id(std::string_view target) {
    std::uint64_t value = 1469598103934665603ULL;
    for (const unsigned char byte : target) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    return value ? value : 1;
}

ConstraintScopeType scope_type(std::string_view value) {
    if (value == "global")
        return ConstraintScopeType::Global;
    if (value == "goal")
        return ConstraintScopeType::Goal;
    if (value == "plan")
        return ConstraintScopeType::Plan;
    if (value == "action")
        return ConstraintScopeType::Action;
    if (value == "resource")
        return ConstraintScopeType::Resource;
    if (value == "entity")
        return ConstraintScopeType::Entity;
    throw std::invalid_argument("unknown constraint scope");
}

ConstraintSource constraint_source(std::string_view value) {
    if (value == "safety")
        return ConstraintSource::Safety;
    if (value == "system_rule")
        return ConstraintSource::SystemRule;
    if (value == "human_explicit")
        return ConstraintSource::HumanExplicit;
    if (value == "environment")
        return ConstraintSource::Environment;
    if (value == "policy")
        return ConstraintSource::Policy;
    throw std::invalid_argument("unknown constraint source");
}

ContextRecognitionResult
parse_result(std::string_view raw, TimePoint now,
             const ModelContextRecognizerConfig &config) {
    const auto root = Json::parse(raw);
    require_keys(root,
                 {"intent", "goal", "conditions", "constraints", "confidence"});
    ContextRecognitionResult result;
    result.intent = bounded_string(root.at("intent"), "intent",
                                   config.maxStringBytes, true);
    if (!known(config.knownIntents, result.intent))
        throw std::invalid_argument("unknown intent");
    result.confidence = confidence(root.at("confidence"), "confidence");

    const auto &goalJson = root.at("goal");
    if (!goalJson.is_null()) {
        require_keys(goalJson, {"type", "target"}, {"priority"});
        Goal goal;
        goal.type = bounded_string(goalJson.at("type"), "goal.type",
                                   config.maxStringBytes, true);
        if (!known(config.knownGoalTypes, goal.type))
            throw std::invalid_argument("unknown goal type");
        if (!goalJson.at("target").is_null()) {
            const auto target =
                bounded_string(goalJson.at("target"), "goal.target",
                               config.maxStringBytes, true);
            goal.target = target_id(target);
            goal.completionCondition.arguments.emplace("target_name", target);
        }
        if (goalJson.contains("priority")) {
            if (!goalJson.at("priority").is_number_integer())
                throw std::invalid_argument("goal.priority must be integer");
            const auto priority = goalJson.at("priority").get<std::int64_t>();
            if (priority < -100 || priority > 100)
                throw std::invalid_argument("goal.priority is out of range");
            goal.priority = static_cast<int>(priority);
        }
        goal.source = GoalSource::Human;
        goal.createdAt = now;
        goal.updatedAt = now;
        result.goals.push_back(std::move(goal));
    } else if (result.intent == "command") {
        throw std::invalid_argument("command requires goal");
    }

    const auto &conditionArray = root.at("conditions");
    if (!conditionArray.is_array() ||
        conditionArray.size() > config.maxConditions)
        throw std::invalid_argument("invalid conditions array");
    ConditionId nextCondition = 1;
    for (const auto &item : conditionArray) {
        require_keys(item, {"type", "active", "confidence", "attributes"});
        if (!item.at("active").is_boolean())
            throw std::invalid_argument("condition.active must be boolean");
        Condition condition;
        condition.id = nextCondition++;
        condition.type = bounded_string(item.at("type"), "condition.type",
                                        config.maxStringBytes, true);
        condition.active = item.at("active").get<bool>();
        condition.confidence =
            confidence(item.at("confidence"), "condition.confidence");
        condition.timestamp = now;
        condition.attributes = attributes(item.at("attributes"), config);
        result.conditions.push_back(std::move(condition));
    }

    const auto &constraintArray = root.at("constraints");
    if (!constraintArray.is_array() ||
        constraintArray.size() > config.maxConstraints)
        throw std::invalid_argument("invalid constraints array");
    ConstraintId nextConstraint = 1;
    for (const auto &item : constraintArray) {
        require_keys(item,
                     {"type", "critical", "scope", "source", "expression"});
        if (!item.at("critical").is_boolean())
            throw std::invalid_argument("constraint.critical must be boolean");
        Constraint constraint;
        constraint.id = nextConstraint++;
        constraint.type = bounded_string(item.at("type"), "constraint.type",
                                         config.maxStringBytes, true);
        constraint.critical = item.at("critical").get<bool>();
        constraint.source = constraint_source(
            bounded_string(item.at("source"), "constraint.source",
                           config.maxStringBytes, true));
        const auto &scope = item.at("scope");
        require_keys(scope, {"type", "target_id"});
        constraint.scope.type =
            scope_type(bounded_string(scope.at("type"), "constraint.scope.type",
                                      config.maxStringBytes, true));
        if (!scope.at("target_id").is_null()) {
            if (!scope.at("target_id").is_number_unsigned())
                throw std::invalid_argument("scope.target_id must be unsigned");
            constraint.scope.targetId =
                scope.at("target_id").get<std::uint64_t>();
        }
        if ((constraint.scope.type == ConstraintScopeType::Global) !=
            !constraint.scope.targetId)
            throw std::invalid_argument("constraint scope target mismatch");
        const auto &expression = item.at("expression");
        require_keys(expression, {"type", "arguments"});
        constraint.expression.expression =
            bounded_string(expression.at("type"), "constraint.expression.type",
                           config.maxStringBytes, true);
        static const std::vector<std::string> knownExpressions{
            "deny_all", "deny_action_type", "max_action_priority"};
        if (!known(knownExpressions, constraint.expression.expression))
            throw std::invalid_argument("unknown constraint expression");
        constraint.expression.arguments =
            attributes(expression.at("arguments"), config);
        constraint.timestamp = now;
        if (!valid_constraint(constraint))
            throw std::invalid_argument("invalid constraint");
        result.constraints.push_back(std::move(constraint));
    }
    return result;
}

std::string status_reason(ContextModelStatus status) {
    switch (status) {
    case ContextModelStatus::Timeout:
        return "context model timeout";
    case ContextModelStatus::ConnectionFailure:
        return "context model connection failure";
    case ContextModelStatus::HttpError:
        return "context model HTTP error";
    case ContextModelStatus::InvalidResponse:
        return "context model invalid response";
    case ContextModelStatus::Failed:
        return "context model failure";
    case ContextModelStatus::Succeeded:
        return {};
    }
    return "context model unknown failure";
}

} // namespace

ModelContextRecognizer::ModelContextRecognizer(
    std::unique_ptr<IContextModelBackend> backend,
    std::unique_ptr<IContextRecognizer> fallbackRecognizer,
    ModelContextRecognizerConfig config, std::shared_ptr<ILogManager> logger)
    : _backend(std::move(backend)), _fallback(std::move(fallbackRecognizer)),
      _config(std::move(config)), _logger(std::move(logger)) {
    if (!_backend || !_fallback || !_config.maxResponseBytes ||
        !_config.maxStringBytes || !_config.maxConditions ||
        !_config.maxConstraints || !_config.maxAttributes ||
        _config.knownIntents.empty() || _config.knownGoalTypes.empty())
        throw std::invalid_argument("invalid ModelContextRecognizer config");
    if (!_logger)
        _logger = std::make_shared<InMemoryLogManager>();
}

ContextRecognitionResult
ModelContextRecognizer::fallback(const std::string &text, TimePoint now,
                                 std::string reason) {
    if (_logger) {
        try {
            _logger->report({_nextErrorId++, ErrorLevel::Warning,
                             context_module, now, "context model fallback",
                             std::move(reason), RecoveryAction::Fallback});
        } catch (...) {
        }
    }
    return _fallback->recognize(text, now);
}

ContextRecognitionResult
ModelContextRecognizer::recognize(const std::string &text, TimePoint now) {
    try {
        const auto response = _backend->infer({text});
        if (response.status != ContextModelStatus::Succeeded)
            return fallback(text, now,
                            response.error.empty()
                                ? status_reason(response.status)
                                : response.error);
        if (response.content.empty())
            return fallback(text, now, "empty context model response");
        if (response.content.size() > _config.maxResponseBytes)
            return fallback(text, now, "context model response too large");
        return parse_result(response.content, now, _config);
    } catch (const std::exception &error) {
        return fallback(text, now, error.what());
    } catch (...) {
        return fallback(text, now, "unknown context backend exception");
    }
}

} // namespace ai::brain
