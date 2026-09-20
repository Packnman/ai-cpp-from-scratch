#include "TestSupport.hpp"
#include "brain/external/ExternalAIAdapter.hpp"

using namespace ai::brain;

namespace {
ExternalAIRequest request(TimePoint now) {
    ExternalAIRequest value;
    value.createdAt = now;
    value.timeout = Duration{20};
    value.payload = std::string("payload");
    value.goalId = 1;
    return value;
}
} // namespace

int main() {
    brain_test::Suite t{"UT-BRN-MOD-011"};
    const auto now = steady_now();
    FakeExternalAI immediate;
    auto id = immediate.submit(request(now));
    auto response = immediate.poll(id, now);
    t.expect(response && response->status == ResponseStatus::Succeeded,
             "UT-011-001", "normal response succeeds");
    t.expect(!immediate.poll(999, now), "UT-011-002",
             "unknown request ID rejected");
    auto badSchema = request(now);
    badSchema.schemaVersion = 2;
    t.expect(immediate.submit(badSchema) == 0, "UT-011-003",
             "schema mismatch rejected");
    FakeExternalAI slow{Duration{50}};
    id = slow.submit(request(now));
    t.expect(slow.poll(id, now + Duration{21})->status ==
                 ResponseStatus::Timeout,
             "UT-011-004", "request times out");
    t.expect(slow.poll(id, now + Duration{60})->status ==
                 ResponseStatus::Timeout,
             "UT-011-005", "late response is not accepted");
    t.skip("UT-011-006/007", "retry policy is not implemented");
    t.skip("UT-011-008",
           "active Goal validation belongs to missing coordinator API");
    FakeExternalAI failing{Duration{0}, true};
    t.expect(failing.poll(failing.submit(request(now)), now)->status ==
                 ResponseStatus::Failed,
             "UT-011-009", "failure is surfaced for local fallback");
    FakeExternalAI deterministic;
    auto one = deterministic.poll(deterministic.submit(request(now)), now);
    auto two = deterministic.poll(deterministic.submit(request(now)), now);
    t.expect(one->payload == two->payload && one->modelId == two->modelId,
             "UT-011-010", "fake result contents deterministic");
    return t.finish();
}
