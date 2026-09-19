#include "ai/agent/agent.h"
#include "ai/agent/memory.h"
#include "ai/agent/reasoner.h"
#include "ai/agent/tools.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace ai::agent;
namespace {
#define CHECK(x)                                                               \
    do {                                                                       \
        if (!(x))                                                              \
            throw std::runtime_error(std::string("check failed: ") + #x);      \
    } while (false)
struct StaticPlanner final : IPlanner {
        int replans{};
        Plan create(const ParsedInput &p,
                    const std::vector<MemoryRecord> &) override {
            return {p.goal, {{"one", TaskType::Tool, "fake", {}, {}}}};
        }
        Plan replan(const ParsedInput &p, const Plan &, const ToolResult &,
                    const EvaluationResult &) override {
            ++replans;
            return create(p, {});
        }
};
struct SequenceExecutor final : IExecutor {
        std::vector<ToolStatus> statuses;
        int calls{};
        ToolResult execute(const Task &t,
                           const std::vector<ToolResult> &) override {
            auto status =
                statuses[std::min<std::size_t>(calls, statuses.size() - 1)];
            ++calls;
            return {t.id,
                    status,
                    {{"answer", 42}},
                    status == ToolStatus::Success ? "" : "injected"};
        }
};
struct CountingEvaluator final : IEvaluator {
        int calls{};
        RuleReasoner rule;
        EvaluationResult evaluate(const Task &t, const ToolResult &r) override {
            ++calls;
            return rule.evaluate(t, r);
        }
};
std::shared_ptr<Agent> make(std::shared_ptr<StaticPlanner> planner,
                            std::shared_ptr<SequenceExecutor> executor,
                            std::shared_ptr<CountingEvaluator> evaluator,
                            const std::filesystem::path &db) {
    auto reasoner = std::make_shared<RuleReasoner>();
    auto memory = std::make_shared<SqliteMemory>(db);
    return std::make_shared<Agent>(
        std::make_shared<DefaultInputParser>(reasoner),
        std::make_shared<DefaultRouter>(), planner, executor, evaluator,
        std::make_shared<DefaultAggregator>(), memory, reasoner);
}
} // namespace
int main() {
    try {
        auto dir =
            std::filesystem::temp_directory_path() / "ai_cpp_control_check";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        auto p1 = std::make_shared<StaticPlanner>();
        auto x1 = std::make_shared<SequenceExecutor>();
        x1->statuses = {ToolStatus::RetryableError, ToolStatus::RetryableError,
                        ToolStatus::Success};
        auto e1 = std::make_shared<CountingEvaluator>();
        auto a1 = make(p1, x1, e1, dir / "retry.sqlite");
        CHECK(a1->process("説明して").success);
        CHECK(x1->calls == 3 && e1->calls == 3);
        auto p2 = std::make_shared<StaticPlanner>();
        auto x2 = std::make_shared<SequenceExecutor>();
        x2->statuses = {ToolStatus::NeedsReplan};
        auto e2 = std::make_shared<CountingEvaluator>();
        auto a2 = make(p2, x2, e2, dir / "replan.sqlite");
        auto failed = a2->process("説明して");
        CHECK(!failed.success);
        CHECK(p2->replans == 3 && x2->calls == 4 && e2->calls == 4);
        DefaultAggregator agg;
        auto j =
            agg.aggregate({{"a", ToolStatus::Success, {{"state", true}}, {}},
                           {"b", ToolStatus::Success, {{"state", false}}, {}}});
        CHECK(j["contradictions"].empty());
        std::ofstream(dir / "large.txt") << std::string(1024 * 1024 + 1, 'x');
        FileReadTool files(dir);
        CHECK(files.execute({{"path", "large.txt"}}).status ==
              ToolStatus::PermanentError);
        std::error_code ec;
        std::filesystem::create_symlink("/etc/passwd", dir / "escape", ec);
        if (!ec)
            CHECK(files.execute({{"path", "escape"}}).status ==
                  ToolStatus::PermanentError);
        std::filesystem::remove_all(dir);
        std::cout << "control_flow_check passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
