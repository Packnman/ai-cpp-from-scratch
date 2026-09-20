#include "TestSupport.hpp"
#include "brain/world/WorldStateManager.hpp"

#include <thread>

using namespace ai::brain;

int main() {
    brain_test::Suite t{"UT-BRN-MOD-003"};
    const auto now = steady_now();
    WorldStateManager world;
    SemanticItem entity{1,  SemanticType::Perception,       now, .8F, true,
                        {}, {{"name", std::string("ball")}}};
    t.expect(world.update(entity) && world.snapshot().perceptions.size() == 1,
             "UT-003-001", "new entity added");
    entity.confidence = .9F;
    entity.timestamp += Duration{1};
    t.expect(world.update(entity) &&
                 world.snapshot().perceptions.at(1).confidence == .9F,
             "UT-003-002", "same ID updated");
    t.skip("UT-003-003",
           "tracking ID resolution is not exposed by SemanticItem");
    SemanticItem condition{2, SemanticType::Condition, now, 1.F, true, {}, {}};
    t.expect(world.update(condition) &&
                 world.snapshot().conditions.at(2).active,
             "UT-003-004", "condition added");
    t.expect(world.updateSafety({now, SafetyLevel::Warning, true, {}}) &&
                 world.snapshot().safetyState.level == SafetyLevel::Warning,
             "UT-003-005", "safety state updated");
    t.expect(world.expire(now + Duration{20}, Duration{10}) >= 1, "UT-003-006",
             "expired observations marked stale");
    t.skip("UT-003-007", "conflict metadata is not represented in WorldState");
    auto snapshot = world.snapshot();
    entity.id = 3;
    entity.timestamp = now + Duration{30};
    world.update(entity);
    t.expect(!snapshot.perceptions.contains(3), "UT-003-008",
             "snapshot is an immutable copy");
    const auto version = world.version();
    world.updateCommunication({now + Duration{31}, true, false, {}});
    t.expect(world.version() == version + 1, "UT-003-009",
             "version increments");
    std::thread writer([&] {
        for (SemanticId id = 10; id < 30; ++id)
            world.update({id,
                          SemanticType::Perception,
                          now + Duration{static_cast<long>(id)},
                          1.F,
                          true,
                          {},
                          {}});
    });
    std::thread reader([&] {
        for (int i = 0; i < 20; ++i)
            (void)world.snapshot();
    });
    writer.join();
    reader.join();
    t.expect(world.snapshot().perceptions.size() >= 20, "UT-003-010",
             "concurrent reads and writes complete");
    return t.finish();
}
