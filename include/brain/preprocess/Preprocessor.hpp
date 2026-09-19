#pragma once
#include "brain/preprocess/ContextRecognizer.hpp"
#include "brain/preprocess/ObjectDetector.hpp"
#include "brain/preprocess/SpeechRecognizer.hpp"
#include <memory>
namespace ai::brain {
class IPreprocessor {
    public:
        virtual ~IPreprocessor() = default;
        virtual std::vector<SemanticItem> process(const BrainInput &) = 0;
};
class Preprocessor final : public IPreprocessor {
    public:
        Preprocessor(std::unique_ptr<IObjectDetector>,
                     std::unique_ptr<ISpeechRecognizer>,
                     std::unique_ptr<IContextRecognizer>);
        std::vector<SemanticItem> process(const BrainInput &) override;

    private:
        std::unique_ptr<IObjectDetector> _objects;
        std::unique_ptr<ISpeechRecognizer> _speech;
        std::unique_ptr<IContextRecognizer> _context;
        SemanticId _nextId{1};
};
} // namespace ai::brain
