#include "TestSupport.hpp"
#include "brain/input/InputAdapter.hpp"

#include <thread>

using namespace ai::brain;

int main() {
    brain_test::Suite t{"UT-BRN-MOD-001"};
    const auto now = steady_now();
    auto logger = std::make_shared<InMemoryLogManager>();
    InputAdapter adapter{{1, 32, Duration{100}, Duration{20}}, logger};

    t.expect(
        bool(adapter.push({InputSource::Sensor, BrainInputType::CameraFrame, 1,
                           now, ByteBuffer{1}, 1})),
        "UT-001-001", "camera input accepted");
    t.expect(
        bool(adapter.push({InputSource::HumanInterface, BrainInputType::Text, 1,
                           now, std::string("hello"), 2})),
        "UT-001-002", "HMI text accepted");
    t.expect(
        bool(adapter.push({InputSource::Control, BrainInputType::ActionResult,
                           1, now, AttributeMap{{"ok", true}}, 3})),
        "UT-001-003", "control result accepted");
    t.expect(
        bool(adapter.push({InputSource::Safety, BrainInputType::StopRequest, 1,
                           now, std::string("stop"), 4})),
        "UT-001-004", "safety input accepted");
    t.expect(!adapter.push({static_cast<InputSource>(99), BrainInputType::Text,
                            1, now, std::string("bad"), 5}),
             "UT-001-005", "unknown source rejected");
    t.expect(
        !adapter.push({InputSource::Sensor, static_cast<BrainInputType>(99), 1,
                       now, std::string("bad"), 6}),
        "UT-001-006", "unknown type rejected");
    t.expect(!adapter.push({InputSource::Sensor, BrainInputType::Text, 2, now,
                            std::string("bad"), 7}) &&
                 !logger->errors().empty(),
             "UT-001-007", "schema mismatch rejected and logged");
    t.expect(!adapter.push({InputSource::Sensor,
                            BrainInputType::Text,
                            1,
                            {},
                            std::string("bad"),
                            8}),
             "UT-001-008", "missing timestamp rejected");
    t.expect(!adapter.push({InputSource::Sensor, BrainInputType::Text, 1,
                            now - Duration{101}, std::string("old"), 9}),
             "UT-001-009", "stale sensor rejected");
    auto polled = adapter.poll();
    t.expect(polled.size() == 4 && polled.front().source == InputSource::Safety,
             "UT-001-010", "safety queue has priority");

    InputAdapter fifo;
    fifo.push({InputSource::Sensor, BrainInputType::Text, 1, now,
               std::string("first"), 1});
    fifo.push({InputSource::Sensor, BrainInputType::Text, 1, now,
               std::string("second"), 2});
    polled = fifo.poll();
    t.expect(std::get<std::string>(polled[0].payload) == "first" &&
                 std::get<std::string>(polled[1].payload) == "second",
             "UT-001-011", "same-priority input is FIFO");

    InputAdapter bounded{{1, 1, Duration{100}, Duration{20}}};
    t.expect(bool(bounded.push({InputSource::Sensor, BrainInputType::Text, 1,
                                now, std::string("one"), 1})) &&
                 !bounded.push({InputSource::Sensor, BrainInputType::Text, 1,
                                now, std::string("two"), 2}),
             "UT-001-012", "queue capacity enforced");

    InputAdapter concurrent{{1, 64, Duration{100}, Duration{20}}};
    std::vector<std::thread> workers;
    for (std::uint64_t i = 1; i <= 16; ++i)
        workers.emplace_back([&, i] {
            concurrent.push({InputSource::Sensor, BrainInputType::Text, 1, now,
                             std::to_string(i), i});
        });
    for (auto &worker : workers)
        worker.join();
    t.expect(concurrent.poll().size() == 16, "UT-001-013",
             "concurrent pushes are not lost");
    return t.finish();
}
