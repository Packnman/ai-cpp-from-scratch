#pragma once
#include "brain/common/Dtos.hpp"
namespace ai::brain {
struct ObjectDetectionResult {
        SemanticId objectId{};
        std::string objectClass;
        float confidence{};
        TimePoint timestamp{};
        AttributeMap attributes;
};
class IObjectDetector {
    public:
        virtual ~IObjectDetector() = default;
        virtual std::vector<ObjectDetectionResult> detect(const ByteBuffer &,
                                                          TimePoint) = 0;
};
class FakeObjectDetector final : public IObjectDetector {
    public:
        std::vector<ObjectDetectionResult> detect(const ByteBuffer &,
                                                  TimePoint) override;
};
} // namespace ai::brain
