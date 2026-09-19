#pragma once

#include "ai/agent/components.h"

#include <map>
#include <set>

namespace ai::agent {

enum class EvidenceOrigin {
    User,
    Document,
    Tool,
    ModelInference,
    VerifiedJudgement
};
enum class ClaimStance { Asserted, Proposed, Verified, Retracted, Superseded };
enum class DecisionStatus {
    Supported,
    InsufficientEvidence,
    ConflictingEvidence,
    NoFeasibleOption
};

struct Evidence {
        std::string id;
        std::string text;
        std::string source_id;
        std::size_t start{};
        std::size_t end{};
        EvidenceOrigin origin{EvidenceOrigin::User};
        bool content_verified{false};
};

struct Claim {
        std::string id;
        std::string speaker;
        std::string subject;
        std::string attribute;
        std::string time;
        std::string condition;
        double value{};
        std::string unit;
        ClaimStance stance{ClaimStance::Asserted};
        std::vector<std::string> evidence_ids;
};

struct DiscussionConstraint {
        std::string id;
        std::string attribute;
        std::string operation;
        double value{};
        std::string unit;
        bool required{true};
        std::vector<std::string> evidence_ids;
};

struct StateUpdate {
        std::string id;
        std::string kind;
        std::string target_id;
        nlohmann::json replacement;
        std::optional<Evidence> evidence;
};

struct Decision {
        DecisionStatus status{DecisionStatus::InsufficientEvidence};
        std::string conclusion;
        std::optional<std::string> selected_option;
        std::vector<std::string> evidence_ids;
        std::vector<std::string> open_questions;
};

struct DiscussionState {
        std::string scenario_id;
        std::string topic;
        std::string objective;
        std::vector<std::string> option_ids;
        std::map<std::string, Evidence, std::less<>> evidence;
        std::map<std::string, Claim, std::less<>> claims;
        std::map<std::string, DiscussionConstraint, std::less<>> constraints;
        nlohmann::json priorities = nlohmann::json::array();
        std::vector<std::string> requested_attributes;
        std::set<std::string, std::less<>> applied_update_ids;
        nlohmann::json revisions = nlohmann::json::array();
        Decision decision;
};

std::string to_string(EvidenceOrigin);
std::string to_string(ClaimStance);
std::string to_string(DecisionStatus);
nlohmann::json to_json(const DiscussionState &);

bool validate_discussion_result(
    const nlohmann::json &result,
    const std::set<std::string, std::less<>> &known_evidence,
    std::string &error);

class DiscussionEngine final : public IReasoningTaskExecutor {
    public:
        ToolResult execute(const Task &,
                           const std::vector<ToolResult> &) override;
        nlohmann::json state(std::string_view scenario_id) const;

    private:
        std::map<std::string, DiscussionState, std::less<>> _states;
};

} // namespace ai::agent
