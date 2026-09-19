#pragma once
#include "brain/common/Dtos.hpp"
#include <shared_mutex>
namespace ai::brain {
class IWorldStateManager {
    public:
        virtual ~IWorldStateManager() = default;
        virtual bool update(const SemanticItem &) = 0;
        virtual WorldState snapshot() const = 0;
        virtual std::uint64_t version() const = 0;
        virtual std::size_t expire(TimePoint, Duration) = 0;
        virtual bool updateSafety(const SafetyState &) = 0;
        virtual bool updateCommunication(const CommunicationState &) = 0;
};
class WorldStateManager final : public IWorldStateManager {
    public:
        WorldStateManager();
        bool update(const SemanticItem &) override;
        WorldState snapshot() const override;
        std::uint64_t version() const override;
        std::size_t expire(TimePoint, Duration) override;
        bool updateSafety(const SafetyState &) override;
        bool updateCommunication(const CommunicationState &) override;

    private:
        mutable std::shared_mutex _mutex;
        WorldState _state;
};
} // namespace ai::brain
