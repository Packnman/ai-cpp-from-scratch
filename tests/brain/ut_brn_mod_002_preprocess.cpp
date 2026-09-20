#include "TestSupport.hpp"
#include "brain/preprocess/Preprocessor.hpp"

using namespace ai::brain;

int main() {
    brain_test::Suite t{"UT-BRN-MOD-002"};
    const auto now = steady_now();
    Preprocessor pre(std::make_unique<FakeObjectDetector>(),
                     std::make_unique<FakeSpeechRecognizer>("goal:listen"),
                     std::make_unique<RuleContextRecognizer>());
    auto camera = pre.process({1, InputSource::Sensor, now, InputStatus::Valid,
                               BrainInputType::CameraFrame, ByteBuffer{1, 2}});
    t.expect(camera.size() == 1 && camera[0].type == SemanticType::Perception,
             "UT-002-001/004", "camera routed and object normalized");
    auto voice =
        pre.process({2, InputSource::HumanInterface, now, InputStatus::Valid,
                     BrainInputType::Voice, ByteBuffer{1}});
    t.expect(voice.size() == 2 && voice[1].type == SemanticType::Goal,
             "UT-002-002", "voice routed through ASR and context");
    auto text =
        pre.process({3, InputSource::HumanInterface, now, InputStatus::Valid,
                     BrainInputType::Text, std::string("goal:inspect")});
    t.expect(text.size() == 2 && text[1].type == SemanticType::Goal,
             "UT-002-003/005", "text command creates goal");
    t.skip("UT-002-006", "constraint language is not in RuleContextRecognizer");
    t.skip("UT-002-007/008", "recognizer confidence injection API is absent");
    t.skip("UT-002-009/010", "failure/timeout recognizer doubles are absent");
    auto again =
        pre.process({4, InputSource::HumanInterface, now, InputStatus::Valid,
                     BrainInputType::Text, std::string("goal:inspect")});
    t.expect(text[0].attributes == again[0].attributes &&
                 text[1].attributes == again[1].attributes,
             "UT-002-011", "fake semantic contents are deterministic");
    return t.finish();
}
