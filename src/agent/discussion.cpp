#include "ai/agent/discussion.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ai::agent {
namespace {
EvidenceOrigin parse_origin(std::string_view value) {
    if (value == "user")
        return EvidenceOrigin::User;
    if (value == "document")
        return EvidenceOrigin::Document;
    if (value == "tool")
        return EvidenceOrigin::Tool;
    if (value == "model")
        return EvidenceOrigin::ModelInference;
    if (value == "verified_judgement")
        return EvidenceOrigin::VerifiedJudgement;
    throw std::invalid_argument("invalid evidence origin");
}
ClaimStance parse_stance(std::string_view value) {
    if (value == "asserted")
        return ClaimStance::Asserted;
    if (value == "proposed")
        return ClaimStance::Proposed;
    if (value == "verified")
        return ClaimStance::Verified;
    if (value == "retracted")
        return ClaimStance::Retracted;
    if (value == "superseded")
        return ClaimStance::Superseded;
    throw std::invalid_argument("invalid claim stance");
}
Evidence parse_evidence(const nlohmann::json &json) {
    Evidence out;
    out.id = json.at("id").get<std::string>();
    out.text = json.at("text").get<std::string>();
    out.source_id = json.at("source_id").get<std::string>();
    out.start = json.at("start").get<std::size_t>();
    out.end = json.at("end").get<std::size_t>();
    out.origin = parse_origin(json.at("origin").get<std::string>());
    out.content_verified = json.value("content_verified", false);
    if (out.id.empty() || out.text.empty() || out.source_id.empty() ||
        out.start >= out.end)
        throw std::invalid_argument("invalid evidence");
    return out;
}
Claim parse_claim(const nlohmann::json &json) {
    Claim out;
    out.id = json.at("id").get<std::string>();
    out.speaker = json.at("speaker").get<std::string>();
    out.subject = json.at("subject").get<std::string>();
    out.attribute = json.at("attribute").get<std::string>();
    out.time = json.value("time", "current");
    out.condition = json.value("condition", "default");
    out.value = json.at("value").get<double>();
    out.unit = json.at("unit").get<std::string>();
    out.stance = parse_stance(json.value("stance", "asserted"));
    out.evidence_ids = json.at("evidence_ids").get<std::vector<std::string>>();
    if (out.id.empty() || out.subject.empty() || out.attribute.empty() ||
        out.unit.empty() || !std::isfinite(out.value) ||
        out.evidence_ids.empty())
        throw std::invalid_argument("invalid claim");
    return out;
}
DiscussionConstraint parse_constraint(const nlohmann::json &json) {
    DiscussionConstraint out;
    out.id = json.at("id").get<std::string>();
    out.attribute = json.at("attribute").get<std::string>();
    out.operation = json.at("op").get<std::string>();
    out.value = json.at("value").get<double>();
    out.unit = json.at("unit").get<std::string>();
    out.required = json.value("required", true);
    out.evidence_ids = json.value("evidence_ids", std::vector<std::string>{});
    if (out.id.empty() || out.attribute.empty() || out.unit.empty() ||
        !std::isfinite(out.value) ||
        (out.operation != "<=" && out.operation != ">=" &&
         out.operation != "=="))
        throw std::invalid_argument("invalid constraint");
    return out;
}
bool usable(const Claim &claim,
            const std::map<std::string, Evidence, std::less<>> &evidence) {
    if (claim.stance == ClaimStance::Proposed ||
        claim.stance == ClaimStance::Retracted ||
        claim.stance == ClaimStance::Superseded)
        return false;
    for (const auto &id : claim.evidence_ids) {
        const auto found = evidence.find(id);
        if (found == evidence.end())
            return false;
        const auto origin = found->second.origin;
        if (origin == EvidenceOrigin::Document ||
            origin == EvidenceOrigin::Tool ||
            origin == EvidenceOrigin::VerifiedJudgement ||
            found->second.content_verified)
            return true;
    }
    return claim.stance == ClaimStance::Verified;
}
bool satisfies(double actual, std::string_view operation, double limit) {
    constexpr double epsilon = 1e-9;
    if (operation == "<=")
        return actual <= limit + epsilon;
    if (operation == ">=")
        return actual + epsilon >= limit;
    return std::abs(actual - limit) <= epsilon;
}
std::string key(const Claim &claim) {
    return claim.subject + "\x1f" + claim.attribute + "\x1f" + claim.time +
           "\x1f" + claim.condition + "\x1f" + claim.unit;
}
void require_known_evidence(const std::vector<std::string> &ids,
                            const DiscussionState &state) {
    for (const auto &id : ids)
        if (!state.evidence.contains(id))
            throw std::invalid_argument("unknown evidence id: " + id);
}
void add_unique(std::vector<std::string> &out, const std::string &value) {
    if (std::find(out.begin(), out.end(), value) == out.end())
        out.push_back(value);
}

Decision decide(const DiscussionState &state) {
    Decision result;
    std::map<std::string, std::vector<const Claim *>, std::less<>> grouped;
    for (const auto &[id, claim] : state.claims) {
        (void)id;
        if (usable(claim, state.evidence))
            grouped[key(claim)].push_back(&claim);
    }
    for (const auto &[claim_key, claims] : grouped) {
        (void)claim_key;
        for (std::size_t i = 1; i < claims.size(); ++i)
            if (std::abs(claims[i]->value - claims[0]->value) > 1e-9) {
                result.status = DecisionStatus::ConflictingEvidence;
                result.conclusion =
                    "同じ対象・時点・条件の根拠が食い違っています。";
                for (const auto *claim : claims)
                    for (const auto &evidence_id : claim->evidence_ids)
                        add_unique(result.evidence_ids, evidence_id);
                result.open_questions.push_back(
                    "どの根拠を採用するか確認してください。");
                return result;
            }
    }

    struct OptionResult {
            std::string id;
            bool feasible{true};
            bool complete{true};
            std::map<std::string, double, std::less<>> values;
    };
    std::vector<OptionResult> options;
    for (const auto &option : state.option_ids) {
        OptionResult checked;
        checked.id = option;
        for (const auto &[constraint_id, constraint] : state.constraints) {
            (void)constraint_id;
            if (!constraint.required)
                continue;
            const Claim *match = nullptr;
            for (const auto &[claim_id, claim] : state.claims) {
                (void)claim_id;
                if (claim.subject == option &&
                    claim.attribute == constraint.attribute &&
                    claim.time == "current" && claim.condition == "default" &&
                    claim.unit == constraint.unit &&
                    usable(claim, state.evidence)) {
                    match = &claim;
                    break;
                }
            }
            if (!match) {
                checked.complete = false;
                continue;
            }
            checked.values[constraint.attribute] = match->value;
            checked.feasible &=
                satisfies(match->value, constraint.operation, constraint.value);
            for (const auto &id : match->evidence_ids)
                add_unique(result.evidence_ids, id);
            for (const auto &id : constraint.evidence_ids)
                add_unique(result.evidence_ids, id);
        }
        for (const auto &attribute : state.requested_attributes) {
            bool found = false;
            for (const auto &[claim_id, claim] : state.claims) {
                (void)claim_id;
                if (claim.subject == option && claim.attribute == attribute &&
                    claim.time == "current" && claim.condition == "default" &&
                    usable(claim, state.evidence)) {
                    found = true;
                    checked.values[attribute] = claim.value;
                    for (const auto &id : claim.evidence_ids)
                        add_unique(result.evidence_ids, id);
                    break;
                }
            }
            checked.complete &= found;
        }
        options.push_back(std::move(checked));
    }
    if (options.empty()) {
        result.status = DecisionStatus::InsufficientEvidence;
        result.conclusion = "比較対象がありません。";
        result.open_questions.push_back("比較対象を提示してください。");
        return result;
    }
    if (std::any_of(options.begin(), options.end(),
                    [](const auto &option) { return !option.complete; })) {
        result.status = DecisionStatus::InsufficientEvidence;
        result.conclusion = "必要な情報が不足しているため判断を保留します。";
        result.open_questions.push_back(
            "不足している属性の資料を提示してください。");
        return result;
    }
    std::vector<OptionResult *> feasible;
    for (auto &option : options)
        if (option.feasible)
            feasible.push_back(&option);
    if (feasible.empty()) {
        result.status = DecisionStatus::NoFeasibleOption;
        result.conclusion = "すべての必須条件を満たす選択肢はありません。条件を"
                            "勝手に緩めません。";
        result.open_questions.push_back("変更可能な条件を確認してください。");
        return result;
    }
    if (feasible.size() == 1) {
        result.status = DecisionStatus::Supported;
        result.selected_option = feasible.front()->id;
        result.conclusion =
            feasible.front()->id + "だけが必須条件を満たします。";
        return result;
    }
    if (!state.priorities.is_array() || state.priorities.empty()) {
        result.status = DecisionStatus::InsufficientEvidence;
        result.conclusion = "複数の選択肢が条件を満たします。";
        result.open_questions.push_back("優先順位を指定してください。");
        return result;
    }
    auto candidates = feasible;
    for (const auto &priority : state.priorities) {
        const auto attribute = priority.at("attribute").get<std::string>();
        const auto direction = priority.at("direction").get<std::string>();
        if (direction != "min" && direction != "max")
            throw std::invalid_argument(
                "priority direction must be min or max");
        if (std::any_of(candidates.begin(), candidates.end(),
                        [&](const auto *option) {
                            return !option->values.contains(attribute);
                        })) {
            result.status = DecisionStatus::InsufficientEvidence;
            result.conclusion = "優先順位の評価に必要な情報が不足しています。";
            result.open_questions.push_back(attribute +
                                            "の資料を提示してください。");
            return result;
        }
        const auto best = std::minmax_element(
            candidates.begin(), candidates.end(),
            [&](const auto *a, const auto *b) {
                return a->values.at(attribute) < b->values.at(attribute);
            });
        const double target = direction == "min"
                                  ? (*best.first)->values.at(attribute)
                                  : (*best.second)->values.at(attribute);
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                           [&](const auto *option) {
                               return std::abs(option->values.at(attribute) -
                                               target) > 1e-9;
                           }),
            candidates.end());
        if (candidates.size() == 1)
            break;
    }
    if (candidates.size() != 1) {
        result.status = DecisionStatus::InsufficientEvidence;
        result.conclusion = "優先順位を適用しても選択肢を一意にできません。";
        result.open_questions.push_back("追加の優先条件を指定してください。");
        return result;
    }
    result.status = DecisionStatus::Supported;
    result.selected_option = candidates.front()->id;
    result.conclusion = "明示された条件と優先順位では" +
                        candidates.front()->id + "が選択されます。";
    return result;
}

nlohmann::json decision_json(const Decision &decision) {
    return {{"status", to_string(decision.status)},
            {"conclusion", decision.conclusion},
            {"selected_option", decision.selected_option},
            {"evidence_ids", decision.evidence_ids},
            {"open_questions", decision.open_questions}};
}
} // namespace

std::string to_string(EvidenceOrigin value) {
    switch (value) {
    case EvidenceOrigin::User:
        return "user";
    case EvidenceOrigin::Document:
        return "document";
    case EvidenceOrigin::Tool:
        return "tool";
    case EvidenceOrigin::ModelInference:
        return "model";
    case EvidenceOrigin::VerifiedJudgement:
        return "verified_judgement";
    }
    throw std::invalid_argument("unknown evidence origin");
}
std::string to_string(ClaimStance value) {
    switch (value) {
    case ClaimStance::Asserted:
        return "asserted";
    case ClaimStance::Proposed:
        return "proposed";
    case ClaimStance::Verified:
        return "verified";
    case ClaimStance::Retracted:
        return "retracted";
    case ClaimStance::Superseded:
        return "superseded";
    }
    throw std::invalid_argument("unknown claim stance");
}
std::string to_string(DecisionStatus value) {
    switch (value) {
    case DecisionStatus::Supported:
        return "supported";
    case DecisionStatus::InsufficientEvidence:
        return "insufficient_evidence";
    case DecisionStatus::ConflictingEvidence:
        return "conflicting_evidence";
    case DecisionStatus::NoFeasibleOption:
        return "no_feasible_option";
    }
    throw std::invalid_argument("unknown decision status");
}

nlohmann::json to_json(const DiscussionState &state) {
    nlohmann::json evidence = nlohmann::json::array();
    for (const auto &[id, item] : state.evidence)
        evidence.push_back({{"id", id},
                            {"text", item.text},
                            {"source_id", item.source_id},
                            {"start", item.start},
                            {"end", item.end},
                            {"origin", to_string(item.origin)},
                            {"content_verified", item.content_verified}});
    nlohmann::json claims = nlohmann::json::array();
    for (const auto &[id, item] : state.claims)
        claims.push_back({{"id", id},
                          {"speaker", item.speaker},
                          {"subject", item.subject},
                          {"attribute", item.attribute},
                          {"time", item.time},
                          {"condition", item.condition},
                          {"value", item.value},
                          {"unit", item.unit},
                          {"stance", to_string(item.stance)},
                          {"evidence_ids", item.evidence_ids}});
    nlohmann::json constraints = nlohmann::json::array();
    for (const auto &[id, item] : state.constraints)
        constraints.push_back({{"id", id},
                               {"attribute", item.attribute},
                               {"op", item.operation},
                               {"value", item.value},
                               {"unit", item.unit},
                               {"required", item.required},
                               {"evidence_ids", item.evidence_ids}});
    return {{"scenario_id", state.scenario_id},
            {"topic", state.topic},
            {"objective", state.objective},
            {"options", state.option_ids},
            {"evidence", evidence},
            {"claims", claims},
            {"constraints", constraints},
            {"priorities", state.priorities},
            {"requested_attributes", state.requested_attributes},
            {"applied_update_ids", state.applied_update_ids},
            {"revisions", state.revisions},
            {"decision", decision_json(state.decision)}};
}

bool validate_discussion_result(
    const nlohmann::json &result,
    const std::set<std::string, std::less<>> &known_evidence,
    std::string &error) {
    if (!result.is_object() || !result.contains("status") ||
        !result["status"].is_string() || !result.contains("conclusion") ||
        !result["conclusion"].is_string() ||
        result["conclusion"].get<std::string>().empty() ||
        !result.contains("evidence_ids") ||
        !result["evidence_ids"].is_array() ||
        !result.contains("open_questions") ||
        !result["open_questions"].is_array()) {
        error = "missing or invalid required decision field";
        return false;
    }
    const auto status = result["status"].get<std::string>();
    if (status != "supported" && status != "insufficient_evidence" &&
        status != "conflicting_evidence" && status != "no_feasible_option") {
        error = "invalid decision status";
        return false;
    }
    if (result.contains("selected_option") &&
        !(result["selected_option"].is_null() ||
          result["selected_option"].is_string())) {
        error = "invalid selected option";
        return false;
    }
    if (status == "supported" &&
        (!result.contains("selected_option") ||
         !result["selected_option"].is_string() ||
         result["selected_option"].get<std::string>().empty())) {
        error = "supported decision requires selected option";
        return false;
    }
    for (const auto &id : result["evidence_ids"])
        if (!id.is_string() ||
            !known_evidence.contains(id.get<std::string>())) {
            error = "unknown evidence id in decision";
            return false;
        }
    for (const auto &question : result["open_questions"])
        if (!question.is_string() || question.get<std::string>().empty()) {
            error = "invalid open question";
            return false;
        }
    return true;
}

ToolResult DiscussionEngine::execute(const Task &task,
                                     const std::vector<ToolResult> &) {
    ToolResult result;
    result.task_id = task.id;
    try {
        if (task.operation != "discussion.compare")
            throw std::invalid_argument("unsupported reasoning operation: " +
                                        task.operation);
        const auto &input = task.arguments;
        const auto scenario_id = input.at("scenario_id").get<std::string>();
        if (scenario_id.empty())
            throw std::invalid_argument("scenario_id is empty");
        const auto existing = _states.find(scenario_id);
        DiscussionState state =
            existing == _states.end() ? DiscussionState{} : existing->second;
        state.scenario_id = scenario_id;
        if (input.contains("topic"))
            state.topic = input.at("topic").get<std::string>();
        if (input.contains("objective"))
            state.objective = input.at("objective").get<std::string>();
        if (input.contains("options")) {
            for (const auto &option : input.at("options")) {
                const auto id = option.at("id").get<std::string>();
                if (id.empty())
                    throw std::invalid_argument("empty option id");
                if (std::find(state.option_ids.begin(), state.option_ids.end(),
                              id) == state.option_ids.end())
                    state.option_ids.push_back(id);
            }
        }
        if (input.contains("evidence"))
            for (const auto &item : input.at("evidence")) {
                auto evidence = parse_evidence(item);
                state.evidence[evidence.id] = std::move(evidence);
            }
        if (input.contains("claims"))
            for (const auto &item : input.at("claims")) {
                auto claim = parse_claim(item);
                require_known_evidence(claim.evidence_ids, state);
                state.claims[claim.id] = std::move(claim);
            }
        if (input.contains("constraints"))
            for (const auto &item : input.at("constraints")) {
                auto constraint = parse_constraint(item);
                require_known_evidence(constraint.evidence_ids, state);
                state.constraints[constraint.id] = std::move(constraint);
            }
        if (input.contains("priorities"))
            state.priorities = input.at("priorities");
        if (input.contains("requested_attributes"))
            state.requested_attributes = input.at("requested_attributes")
                                             .get<std::vector<std::string>>();
        if (input.contains("updates"))
            for (const auto &item : input.at("updates")) {
                const auto update_id = item.at("id").get<std::string>();
                if (update_id.empty())
                    throw std::invalid_argument("empty update id");
                if (state.applied_update_ids.contains(update_id))
                    continue;
                const auto kind = item.at("kind").get<std::string>();
                const auto target = item.at("target_id").get<std::string>();
                std::optional<Evidence> update_evidence;
                if (item.contains("evidence")) {
                    update_evidence = parse_evidence(item.at("evidence"));
                    state.evidence[update_evidence->id] = *update_evidence;
                }
                if (kind == "correct_constraint") {
                    auto found = state.constraints.find(target);
                    if (found == state.constraints.end())
                        throw std::invalid_argument(
                            "unknown constraint update target");
                    state.revisions.push_back(
                        {{"update_id", update_id},
                         {"kind", kind},
                         {"target_id", target},
                         {"old_value", found->second.value},
                         {"old_evidence_ids", found->second.evidence_ids},
                         {"new_value", item.at("value")},
                         {"new_evidence_id",
                          update_evidence ? nlohmann::json(update_evidence->id)
                                          : nlohmann::json(nullptr)}});
                    found->second.value = item.at("value").get<double>();
                    if (update_evidence)
                        found->second.evidence_ids = {update_evidence->id};
                } else if (kind == "correct_claim") {
                    auto found = state.claims.find(target);
                    if (found == state.claims.end())
                        throw std::invalid_argument(
                            "unknown claim update target");
                    state.revisions.push_back(
                        {{"update_id", update_id},
                         {"kind", kind},
                         {"target_id", target},
                         {"old_value", found->second.value},
                         {"old_evidence_ids", found->second.evidence_ids},
                         {"new_value", item.at("value")},
                         {"new_evidence_id",
                          update_evidence ? nlohmann::json(update_evidence->id)
                                          : nlohmann::json(nullptr)}});
                    found->second.value = item.at("value").get<double>();
                    found->second.stance = ClaimStance::Asserted;
                    if (update_evidence)
                        found->second.evidence_ids = {update_evidence->id};
                } else if (kind == "set_priorities") {
                    if (target != "priorities")
                        throw std::invalid_argument(
                            "priority update target must be priorities");
                    const auto &replacement = item.at("value");
                    if (!replacement.is_array())
                        throw std::invalid_argument(
                            "priority replacement must be an array");
                    for (const auto &priority : replacement) {
                        const auto attribute =
                            priority.at("attribute").get<std::string>();
                        const auto direction =
                            priority.at("direction").get<std::string>();
                        if (attribute.empty() ||
                            (direction != "min" && direction != "max"))
                            throw std::invalid_argument(
                                "invalid priority replacement");
                    }
                    state.revisions.push_back(
                        {{"update_id", update_id},
                         {"kind", kind},
                         {"target_id", target},
                         {"old_value", state.priorities},
                         {"new_value", replacement},
                         {"new_evidence_id",
                          update_evidence ? nlohmann::json(update_evidence->id)
                                          : nlohmann::json(nullptr)}});
                    state.priorities = replacement;
                } else if (kind == "retract_claim") {
                    auto found = state.claims.find(target);
                    if (found == state.claims.end())
                        throw std::invalid_argument(
                            "unknown claim update target");
                    state.revisions.push_back(
                        {{"update_id", update_id},
                         {"kind", kind},
                         {"target_id", target},
                         {"old_stance", to_string(found->second.stance)},
                         {"old_evidence_ids", found->second.evidence_ids}});
                    found->second.stance = ClaimStance::Retracted;
                } else {
                    throw std::invalid_argument("unsupported update kind");
                }
                state.applied_update_ids.insert(update_id);
            }
        state.decision = decide(state);
        auto decision = decision_json(state.decision);
        std::set<std::string, std::less<>> known;
        for (const auto &[id, evidence] : state.evidence) {
            (void)evidence;
            known.insert(id);
        }
        std::string error;
        if (!validate_discussion_result(decision, known, error))
            throw std::logic_error("invalid deterministic decision: " + error);
        result.status = ToolStatus::Success;
        result.value = decision;
        result.value["discussion_state"] = to_json(state);
        _states[scenario_id] = std::move(state);
    } catch (const std::exception &error) {
        result.status = ToolStatus::PermanentError;
        result.error = error.what();
    }
    return result;
}

nlohmann::json DiscussionEngine::state(std::string_view scenario_id) const {
    const auto found = _states.find(scenario_id);
    return found == _states.end() ? nlohmann::json(nullptr)
                                  : to_json(found->second);
}
} // namespace ai::agent
