#pragma once
#include "brain/common/Dtos.hpp"
namespace ai::brain {
struct ContextRecognitionResult {
        std::string intent;
        std::vector<Goal> goals;
        std::vector<Condition> conditions;
        std::vector<Constraint> constraints;
        float confidence{};
};
class IContextRecognizer {
    public:
        virtual ~IContextRecognizer() = default;
        virtual ContextRecognitionResult recognize(const std::string &,
                                                   TimePoint) = 0;
};
class RuleContextRecognizer final : public IContextRecognizer {
    public:
        ContextRecognitionResult recognize(const std::string &,
                                           TimePoint) override;
};
} // namespace ai::brain
