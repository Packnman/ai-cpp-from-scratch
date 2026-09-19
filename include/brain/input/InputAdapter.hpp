#pragma once
#include "brain/common/Dtos.hpp"
#include "brain/common/Result.hpp"
#include "brain/logging/LogManager.hpp"
#include <array>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>
namespace ai::brain {
struct ExternalMessage {
        InputSource source{InputSource::Sensor};
        BrainInputType type{BrainInputType::Text};
        std::uint32_t schemaVersion{1};
        TimePoint timestamp{};
        Payload payload;
        std::uint64_t sequence{};
};
struct InputAdapterConfig {
        std::uint32_t schemaVersion{1};
        std::size_t queueCapacity{256};
        Duration defaultMaxAge{Duration{30'000}};
        Duration safetyMaxAge{Duration{1'000}};
};
class IInputAdapter {
    public:
        virtual ~IInputAdapter() = default;
        virtual Result<void> push(const ExternalMessage &) = 0;
        virtual std::vector<BrainInput> poll() = 0;
};
class InputAdapter final : public IInputAdapter {
    public:
        explicit InputAdapter(InputAdapterConfig config = {},
                              std::shared_ptr<ILogManager> logger = {});
        Result<void> push(const ExternalMessage &) override;
        std::vector<BrainInput> poll() override;
        std::size_t size() const;

    private:
        static std::size_t queue_index(InputSource) noexcept;
        InputStatus validate(const ExternalMessage &, TimePoint) const;
        BrainInput normalize(const ExternalMessage &, InputStatus);
        InputAdapterConfig _config;
        std::shared_ptr<ILogManager> _logger;
        mutable std::mutex _mutex;
        std::array<std::deque<BrainInput>, 5> _queues;
        std::uint64_t _nextId{1};
};
} // namespace ai::brain
