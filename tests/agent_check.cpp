#include "ai/agent/agent.h"
#include "ai/agent/context_builder.h"
#include "ai/agent/memory.h"
#include "ai/agent/reasoner.h"
#include "ai/agent/tools.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace ai::agent;
namespace {
int checks = 0;
#define CHECK(x)                                                               \
    do {                                                                       \
        ++checks;                                                              \
        if (!(x))                                                              \
            throw std::runtime_error(std::string("check failed: ") + #x);      \
    } while (false)
template <class F> void throws(F f) {
    bool did = false;
    try {
        f();
    } catch (...) {
        did = true;
    }
    CHECK(did);
}
struct ScriptModel final : ILanguageModel {
        std::vector<std::string> answers;
        std::vector<ModelMode> modes;
        std::string complete(ModelMode m, std::string_view) override {
            modes.push_back(m);
            if (answers.empty())
                throw std::runtime_error("script exhausted");
            auto a = answers.front();
            answers.erase(answers.begin());
            return a;
        }
};
std::shared_ptr<Agent> make_agent(const std::filesystem::path &root,
                                  const std::filesystem::path &db) {
    auto reasoner = std::make_shared<RuleReasoner>();
    auto memory = std::make_shared<SqliteMemory>(db);
    auto tools = std::make_shared<ToolRegistry>();
    tools->add("calculator.calculate", std::make_shared<CalculatorTool>());
    tools->add("file.read", std::make_shared<FileReadTool>(root));
    tools->add("memory.retrieve", std::make_shared<MemoryRetrieveTool>(memory));
    return std::make_shared<Agent>(
        std::make_shared<DefaultInputParser>(reasoner),
        std::make_shared<DefaultRouter>(),
        std::make_shared<DefaultPlanner>(reasoner),
        std::make_shared<DefaultExecutor>(tools),
        std::make_shared<DefaultEvaluator>(reasoner),
        std::make_shared<DefaultAggregator>(), memory, reasoner);
}
void parser_router() {
    RuleReasoner r;
    DefaultRouter router;
    auto p = r.parse("赤い球を3個取って。ただし人がいたら動かないで。");
    CHECK(p.raw.find("3個") != std::string::npos);
    CHECK(p.constraints.size() == 1 && p.constraints[0].critical);
    CHECK(router.route(r.parse("今日は疲れた")) ==
          RequestType::SimpleConversation);
    CHECK(router.route(r.parse("Attentionを説明して")) ==
          RequestType::QuestionAnswer);
    CHECK(router.route(r.parse("これを分解して")) ==
          RequestType::ComplexReasoning);
    CHECK(router.route(r.parse("ロボットを動かす")) == RequestType::RobotTask);
    CHECK(router.route(r.parse("/calc 1+2")) == RequestType::ToolTask);
    CHECK(router.route(r.parse("/recall 設計")) == RequestType::MemoryRecall);
}
void plans() {
    std::string e;
    Plan p{"g",
           {{"a", TaskType::Reasoning, "a", {}, {}},
            {"b", TaskType::Reasoning, "b", {}, {"a"}}}};
    CHECK(validate_plan(p, e));
    auto o = topological_order(p);
    CHECK(o.size() == 2 && o[0] == 0 && o[1] == 1);
    p.tasks[0].depends_on = {"b"};
    CHECK(!validate_plan(p, e));
    p.tasks[0].depends_on = {"missing"};
    CHECK(!validate_plan(p, e));
    p.tasks = {{"x"}, {"x"}};
    CHECK(!validate_plan(p, e));
    p.tasks.clear();
    for (int i = 0; i < 33; ++i)
        p.tasks.push_back({std::to_string(i)});
    CHECK(!validate_plan(p, e));
}
void tool_checks(const std::filesystem::path &dir) {
    CalculatorTool c;
    auto ok = c.execute({{"expression", "-(2 + 3) * 4 / 2"}});
    CHECK(ok.status == ToolStatus::Success && ok.value["value"] == -10);
    CHECK(c.execute({{"expression", "1/0"}}).status ==
          ToolStatus::PermanentError);
    CHECK(c.execute({{"expression", "1 + system()"}}).status ==
          ToolStatus::PermanentError);
    std::ofstream(dir / "ok.txt") << "日本語 UTF-8";
    std::ofstream bad(dir / "bad.txt", std::ios::binary);
    bad.put(static_cast<char>(0xff));
    bad.close();
    FileReadTool f(dir);
    CHECK(f.execute({{"path", "ok.txt"}}).status == ToolStatus::Success);
    CHECK(f.execute({{"path", "../outside"}}).status ==
          ToolStatus::PermanentError);
    CHECK(f.execute({{"path", "bad.txt"}}).status ==
          ToolStatus::PermanentError);
    CHECK(f.execute({{"path", "/etc/passwd"}}).status ==
          ToolStatus::PermanentError);
}
void memory_checks(const std::filesystem::path &db) {
    {
        SqliteMemory m(db);
        m.store({MemoryType::Project, "日本語の検索テスト", .7, .8});
        m.store({MemoryType::Project, "日本語の検索テスト", .9, .6});
        m.store({MemoryType::Semantic, "AI設計", .8, .9});
        auto a = m.retrieve("日本語");
        CHECK(a.size() == 1 && a[0].importance == .9 && a[0].confidence == .8);
        auto q = m.retrieve("AI");
        CHECK(q.size() == 1 && q[0].type == MemoryType::Semantic);
        throws([&] {
            m.store_batch({{MemoryType::Project, "rollback target", .9, .9},
                           {MemoryType::Project, "", .9, .9}});
        });
        CHECK(m.retrieve("rollback target").empty());
    }
    {
        SqliteMemory m(db);
        CHECK(m.retrieve("日本語").size() == 1);
        CHECK(m.retrieve("日本語", MemoryType::Semantic).empty());
    }
}
void context_checks() {
    ContextInput in;
    in.current_input = "入力";
    in.goal = "目標";
    in.current_task = "作業";
    in.constraints = {{"絶対に削除しない", true}};
    in.summary = std::string(200, 's');
    in.recent = {{"u", "a"}};
    ContextBuilder b(100);
    auto x = b.build(in);
    CHECK(x.find("絶対に削除しない") != std::string::npos);
    CHECK(x.find(std::string(200, 's')) == std::string::npos);
    CHECK(x == b.build(in));
    ContextBuilder tiny(2);
    throws([&] { tiny.build(in); });
    CHECK(ContextBuilder::utf8_codepoints("日本a") == 3);
}
void model_checks() {
    auto lm = std::make_shared<ScriptModel>();
    lm->answers = {
        "bad", R"({"raw":"x","intent":"chat","goal":"x","constraints":[]})"};
    ModelReasoner r(lm);
    CHECK(r.parse("x").raw == "x");
    CHECK(lm->modes.size() == 2 && lm->modes[0] == ModelMode::Parse);
    auto fail = std::make_shared<ScriptModel>();
    fail->answers = {"{}", "{}"};
    ModelReasoner r2(fail);
    throws([&] { r2.parse("x"); });
    auto nested = std::make_shared<ScriptModel>();
    nested->answers = {
        R"({"raw":"x","intent":"chat","goal":"x","constraints":[{"text":7}]})",
        R"({"raw":"x","intent":"chat","goal":"x","constraints":[{"text":"safe","critical":true}]})"};
    ModelReasoner r3(nested);
    CHECK(r3.parse("x").constraints.front().critical);
    CHECK(nested->modes.size() == 2);
    auto overflow = std::make_shared<ScriptModel>();
    ModelReasoner r4(overflow);
    throws([&] { r4.parse(std::string(1023, 'x')); });
    CHECK(overflow->modes.empty());
}
void e2e(const std::filesystem::path &dir) {
    auto a = make_agent(dir, dir / "e2e.sqlite");
    auto chat = a->process("こんにちは");
    CHECK(chat.success &&
          chat.state.request_type == RequestType::SimpleConversation &&
          !chat.state.current_plan);
    auto calc = a->process("/calc (2+3)*4");
    CHECK(calc.success && calc.text == "20.0");
    CHECK(a->process("/remember project このAgentはC++20を使う").success);
    auto recall = a->process("/recall C++20");
    CHECK(recall.success &&
          recall.text.find("このAgentはC++20") != std::string::npos);
    auto unknown = a->process("ロボットを動かす");
    CHECK(!unknown.success &&
          unknown.error.find("unknown tool") != std::string::npos);
}
} // namespace
int main() {
    try {
        auto dir =
            std::filesystem::temp_directory_path() / "ai_cpp_agent_check";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        parser_router();
        plans();
        tool_checks(dir);
        memory_checks(dir / "memory.sqlite");
        context_checks();
        model_checks();
        e2e(dir);
        std::filesystem::remove_all(dir);
        std::cout << "agent_check: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
