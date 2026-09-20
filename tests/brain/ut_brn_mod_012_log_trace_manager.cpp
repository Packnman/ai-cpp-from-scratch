#include "TestSupport.hpp"
#include "brain/logging/LogManager.hpp"

#include <stdexcept>

using namespace ai::brain;

namespace {
class Backend final : public ILogBackend {
    public:
        void write(const TraceEvent &event) override {
            events.push_back(event);
        }
        void write(const BrainError &error) override {
            errors.push_back(error);
        }
        void flush() override { flushed = true; }
        std::vector<TraceEvent> events;
        std::vector<BrainError> errors;
        bool flushed{};
};
class FailingBackend final : public ILogBackend {
    public:
        void write(const TraceEvent &) override {
            throw std::runtime_error("fail");
        }
        void write(const BrainError &) override {
            throw std::runtime_error("fail");
        }
        void flush() override { throw std::runtime_error("fail"); }
};
} // namespace

int main() {
    brain_test::Suite t{"UT-BRN-MOD-012"};
    auto backend = std::make_shared<Backend>();
    LogManager log(backend);
    TraceEvent event;
    event.id = 1;
    event.type = TraceType::Execution;
    event.timestamp = steady_now();
    event.goalId = 2;
    event.planId = 3;
    event.actionId = 4;
    event.data = {{"software_version", std::string("test")}};
    log.log(event);
    t.expect(backend->events.size() == 1, "UT-012-001",
             "trace written to backend");
    t.expect(backend->events[0].goalId == 2, "UT-012-002",
             "goal correlation retained");
    t.expect(backend->events[0].planId == 3, "UT-012-003",
             "plan correlation retained");
    t.expect(backend->events[0].actionId == 4, "UT-012-004",
             "action correlation retained");
    BrainError error{1,  ErrorLevel::Critical,    12, steady_now(), "error",
                     {}, RecoveryAction::SafeStop};
    log.report(error);
    t.expect(backend->errors.size() == 1, "UT-012-005",
             "error written to backend");
    t.expect(backend->events[0].data.contains("software_version"), "UT-012-006",
             "version metadata retained");
    LogManager failing(std::make_shared<FailingBackend>());
    failing.log(event);
    failing.report(error);
    failing.flush();
    t.expect(failing.dropped() == 3, "UT-012-007",
             "backend failure does not escape");
    t.skip("UT-012-008/009", "bounded priority log queue is not implemented");
    t.expect(backend->events[0].goalId && backend->events[0].planId &&
                 backend->events[0].actionId,
             "UT-012-010", "replay correlation metadata complete");
    return t.finish();
}
