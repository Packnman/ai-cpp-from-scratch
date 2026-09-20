#include "brain/preprocess/Preprocessor.hpp"
#include <sstream>
namespace ai::brain {
std::vector<ObjectDetectionResult>
FakeObjectDetector::detect(const ByteBuffer &b, TimePoint t) {
    if (b.empty())
        return {};
    return {{1, "fake_object", 1.F, t, {{"bytes", std::uint64_t(b.size())}}}};
}
SpeechRecognitionResult FakeSpeechRecognizer::recognize(const ByteBuffer &,
                                                        TimePoint t) {
    return {_text, 1.F, "ja", {}, t, t};
}
ContextRecognitionResult
RuleContextRecognizer::recognize(const std::string &text, TimePoint t) {
    ContextRecognitionResult r;
    r.intent = "statement";
    r.confidence = 1;
    constexpr std::string_view prefix = "goal:";
    if (text.starts_with(prefix)) {
        Goal g;
        g.type = text.substr(prefix.size());
        g.source = GoalSource::Human;
        g.priority = 1;
        g.createdAt = t;
        g.updatedAt = t;
        r.intent = "command";
        r.goals.push_back(std::move(g));
    }
    return r;
}
Preprocessor::Preprocessor(std::unique_ptr<IObjectDetector> o,
                           std::unique_ptr<ISpeechRecognizer> s,
                           std::unique_ptr<IContextRecognizer> c)
    : _objects(std::move(o)), _speech(std::move(s)), _context(std::move(c)) {}
std::vector<SemanticItem> Preprocessor::process(const BrainInput &i) {
    std::vector<SemanticItem> out;
    if (i.status != InputStatus::Valid)
        return out;
    if (i.type == BrainInputType::CameraFrame &&
        std::holds_alternative<ByteBuffer>(i.payload)) {
        for (const auto &o :
             _objects->detect(std::get<ByteBuffer>(i.payload), i.timestamp))
            out.push_back({_nextId++,
                           SemanticType::Perception,
                           o.timestamp,
                           o.confidence,
                           true,
                           o.objectId,
                           {{"class", o.objectClass}}});
        return out;
    }
    std::string text;
    if (i.type == BrainInputType::Voice &&
        std::holds_alternative<ByteBuffer>(i.payload))
        text = _speech->recognize(std::get<ByteBuffer>(i.payload), i.timestamp)
                   .text;
    else if (i.type == BrainInputType::Text &&
             std::holds_alternative<std::string>(i.payload))
        text = std::get<std::string>(i.payload);
    else
        return out;
    auto context = _context->recognize(text, i.timestamp);
    out.push_back({_nextId++,
                   SemanticType::Conversation,
                   i.timestamp,
                   context.confidence,
                   true,
                   {},
                   {{"text", text}, {"intent", context.intent}}});
    for (const auto &g : context.goals) {
        AttributeMap attributes{{"goal_type", g.type},
                                {"priority", std::int64_t(g.priority)}};
        const auto targetName =
            g.completionCondition.arguments.find("target_name");
        if (targetName != g.completionCondition.arguments.end())
            attributes["target_name"] = targetName->second;
        out.push_back({_nextId++, SemanticType::Goal, i.timestamp,
                       context.confidence, true, g.target,
                       std::move(attributes)});
    }
    for (const auto &condition : context.conditions) {
        auto attributes = condition.attributes;
        attributes["condition_type"] = condition.type;
        out.push_back({_nextId++,
                       SemanticType::Condition,
                       condition.timestamp,
                       condition.confidence,
                       condition.active,
                       {},
                       std::move(attributes)});
    }
    for (const auto &constraint : context.constraints) {
        AttributeMap attributes{
            {"constraint_type", constraint.type},
            {"critical", constraint.critical},
            {"scope_type", std::int64_t(constraint.scope.type)},
            {"source", std::int64_t(constraint.source)},
            {"expression", constraint.expression.expression}};
        if (constraint.scope.targetId)
            attributes["scope_target_id"] = *constraint.scope.targetId;
        for (const auto &[key, value] : constraint.expression.arguments)
            attributes["argument." + key] = value;
        out.push_back({_nextId++,
                       SemanticType::Constraint,
                       constraint.timestamp,
                       context.confidence,
                       constraint.active,
                       {},
                       std::move(attributes)});
    }
    return out;
}
} // namespace ai::brain
