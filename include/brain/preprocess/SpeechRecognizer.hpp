#pragma once
#include "brain/common/Dtos.hpp"
namespace ai::brain {
struct SpeechRecognitionResult {
        std::string text;
        float confidence{};
        std::string language;
        std::optional<SpeakerId> speaker;
        TimePoint start{};
        TimePoint end{};
};
class ISpeechRecognizer {
    public:
        virtual ~ISpeechRecognizer() = default;
        virtual SpeechRecognitionResult recognize(const ByteBuffer &,
                                                  TimePoint) = 0;
};
class FakeSpeechRecognizer final : public ISpeechRecognizer {
    public:
        explicit FakeSpeechRecognizer(std::string text = "fake speech")
            : _text(std::move(text)) {}
        SpeechRecognitionResult recognize(const ByteBuffer &,
                                          TimePoint) override;

    private:
        std::string _text;
};
} // namespace ai::brain
