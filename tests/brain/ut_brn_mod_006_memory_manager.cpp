#include "TestSupport.hpp"
#include "brain/memory/MemoryManager.hpp"

using namespace ai::brain;

namespace {
MemoryItem item(MemoryId id, MemoryType type, float importance,
                std::string tag) {
    MemoryItem value;
    value.id = id;
    value.type = type;
    value.content = {id, SemanticType::Conversation,    steady_now(), 1.F, true,
                     {}, {{"text", std::to_string(id)}}};
    value.importance = importance;
    value.confidence = 1.F;
    value.tags = {std::move(tag)};
    return value;
}
} // namespace

int main() {
    brain_test::Suite t{"UT-BRN-MOD-006"};
    MemoryManager stm({}, 2);
    stm.remember(item(1, MemoryType::Knowledge, .1F, "a"));
    t.expect(stm.shortTermSize() == 1, "UT-006-001", "STM insert");
    stm.remember(item(2, MemoryType::Knowledge, .2F, "a"));
    stm.remember(item(3, MemoryType::Knowledge, .3F, "a"));
    auto recent = stm.recall({});
    t.expect(recent.size() == 2 && recent[0].id == 3, "UT-006-002",
             "STM evicts oldest item");

    auto backend = std::make_unique<SQLiteMemoryBackend>(":memory:");
    auto *db = backend.get();
    MemoryManager memory(std::move(backend));
    memory.remember(item(10, MemoryType::Knowledge, .2F, "blue"));
    memory.remember(item(11, MemoryType::Error, .9F, "failure"));
    t.expect(db->get(10).has_value(), "UT-006-003", "SQLite insert");
    auto updated = *db->get(10);
    updated.importance = .8F;
    t.expect(db->update(updated) && db->get(10)->importance == .8F &&
                 db->remove(10) && !db->get(10),
             "UT-006-004", "backend update and delete");
    t.expect(memory.recall({MemoryType::Error}).size() == 1, "UT-006-005",
             "search by type");
    t.expect(memory.recall({{}, {"failure"}}).size() == 1, "UT-006-006",
             "search by tag");
    t.expect(memory.recall({}).front().id == 11, "UT-006-007",
             "higher-ranked result first");
    t.skip("UT-006-008", "transaction fault injection is not exposed");
    auto unavailable = std::make_unique<SQLiteMemoryBackend>(
        "/path/that/does/not/exist/brain.sqlite3");
    MemoryManager fallback(std::move(unavailable));
    fallback.remember(item(20, MemoryType::Knowledge, .5F, "fallback"));
    t.expect(!fallback.persistent() && fallback.recall({}).size() == 1,
             "UT-006-009/010", "unavailable DB falls back to STM");
    t.skip("UT-006-011", "automatic STM promotion policy is not implemented");
    return t.finish();
}
