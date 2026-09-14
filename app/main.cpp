#include "ai/agent/agent.h"
#include "ai/agent/memory.h"
#include "ai/agent/reasoner.h"
#include "ai/agent/tools.h"

#include <cstdlib>
#include <iostream>

using namespace ai::agent;
int main(int argc, char **argv) {
    try {
        std::filesystem::path root =
            argc > 1 ? argv[1] : std::filesystem::current_path();
        std::filesystem::path db =
            argc > 2 ? argv[2] : std::filesystem::path("agent_memory.sqlite3");
        auto reasoner = std::make_shared<RuleReasoner>();
        auto memory = std::make_shared<SqliteMemory>(db);
        auto tools = std::make_shared<ToolRegistry>();
        tools->add("calculator.calculate", std::make_shared<CalculatorTool>());
        tools->add("file.read", std::make_shared<FileReadTool>(root));
        tools->add("memory.retrieve",
                   std::make_shared<MemoryRetrieveTool>(memory));
        Agent agent(std::make_shared<DefaultInputParser>(reasoner),
                    std::make_shared<DefaultRouter>(),
                    std::make_shared<DefaultPlanner>(reasoner),
                    std::make_shared<DefaultExecutor>(tools),
                    std::make_shared<DefaultEvaluator>(reasoner),
                    std::make_shared<DefaultAggregator>(), memory, reasoner);
        std::cout << "ai_cpp Agent rule backend (/quit to exit)\n";
        std::string line;
        while (std::cout << "> " && std::getline(std::cin, line)) {
            if (line == "/quit")
                break;
            auto r = agent.process(line);
            std::cout << r.text << '\n';
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "fatal: " << e.what() << '\n';
        return 1;
    }
}
