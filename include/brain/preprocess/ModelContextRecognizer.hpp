#pragma once

#include "brain/logging/LogManager.hpp"
#include "brain/preprocess/ContextModelBackend.hpp"
#include "brain/preprocess/ContextRecognizer.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ai::brain {

struct ModelContextRecognizerConfig {
        std::size_t maxResponseBytes{65'536};
        std::size_t maxStringBytes{128};
        std::size_t maxConditions{16};
        std::size_t maxConstraints{16};
        std::size_t maxAttributes{16};
        std::vector<std::string> knownIntents{"statement", "command",
                                              "question", "correction"};
        std::vector<std::string> knownGoalTypes{"AcquireObject", "FindObject",
                                                "MoveObject", "AnswerQuestion"};
};

class ModelContextRecognizer final : public IContextRecognizer {
    public:
        ModelContextRecognizer(std::unique_ptr<IContextModelBackend> backend,
                               std::unique_ptr<IContextRecognizer> fallback =
                                   std::make_unique<RuleContextRecognizer>(),
                               ModelContextRecognizerConfig config = {},
                               std::shared_ptr<ILogManager> logger = {});

        ContextRecognitionResult recognize(const std::string &,
                                           TimePoint) override;

    private:
        ContextRecognitionResult fallback(const std::string &, TimePoint,
                                          std::string reason);
        std::unique_ptr<IContextModelBackend> _backend;
        std::unique_ptr<IContextRecognizer> _fallback;
        ModelContextRecognizerConfig _config;
        std::shared_ptr<ILogManager> _logger;
        ErrorId _nextErrorId{1};
};

} // namespace ai::brain
