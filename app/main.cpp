#include "ai/agent/agent.h"
#include "ai/agent/discussion.h"
#include "ai/agent/memory.h"
#include "ai/agent/reasoner.h"
#include "ai/agent/tools.h"
#include "ai/ner/extractor.h"
#ifdef AI_CPP_BUILD_MODEL
#include "ai/model/model_language_model.h"
#endif

#include <cstdlib>
#include <iostream>
#include <map>

using namespace ai::agent;
int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string, std::less<>> options;
        for (int index = 1; index < argc; index += 2) {
            const std::string name = argv[index];
            if (name == "--help") {
                std::cout
                    << "agent_cli [--backend rule|hybrid|model] [--model PATH] "
                       "[--file-root PATH] [--memory-db PATH] [--seed N] "
                       "[--temperature F] [--top-p F] [--ner "
                       "off|rule|model|hybrid] (limited comparison: /compare "
                       "JSON) [--ner-bundle PATH]\n";
                return 0;
            }
            if (!name.starts_with("--") || index + 1 >= argc)
                throw std::invalid_argument(
                    "options must be named and have a value");
            options[name] = argv[index + 1];
        }
        const auto value = [&](const std::string &name, std::string fallback) {
            const auto found = options.find(name);
            return found == options.end() ? std::move(fallback) : found->second;
        };
        const std::filesystem::path root =
            value("--file-root", std::filesystem::current_path().string());
        const std::filesystem::path db =
            value("--memory-db", "agent_memory.sqlite3");
        const std::string backend = value("--backend", "rule");
        std::shared_ptr<IReasoner> reasoner;
        if (backend == "rule") {
            reasoner = std::make_shared<RuleReasoner>();
        } else if (backend == "model" || backend == "hybrid") {
#ifdef AI_CPP_BUILD_MODEL
            const auto model_path = value("--model", "");
            if (model_path.empty())
                throw std::invalid_argument("--model PATH is required");
            ai::model::GenerationConfig generation;
            generation.seed = std::stoull(value("--seed", "42"));
            generation.temperature = std::stof(value("--temperature", "0.8"));
            generation.top_p = std::stof(value("--top-p", "0.9"));
            auto language_model =
                std::make_shared<ai::model::ModelLanguageModel>(model_path,
                                                                generation);
            reasoner =
                backend == "hybrid"
                    ? std::shared_ptr<IReasoner>(
                          std::make_shared<HybridReasoner>(language_model))
                    : std::shared_ptr<IReasoner>(
                          std::make_shared<ModelReasoner>(language_model));
#else
            throw std::invalid_argument(
                "model backend requires -DAI_CPP_BUILD_MODEL=ON");
#endif
        } else {
            throw std::invalid_argument(
                "--backend must be rule, hybrid, or model");
        }
        std::shared_ptr<ai::ner::IEntityExtractor> entity_extractor;
        const auto ner_mode = value("--ner", "off");
        const auto ner_bundle = value("--ner-bundle", "");
        if (ner_mode == "rule")
            entity_extractor = std::make_shared<ai::ner::RuleEntityExtractor>();
        else if (ner_mode == "model") {
            if (ner_bundle.empty())
                throw std::invalid_argument(
                    "--ner-bundle is required for model NER");
            entity_extractor =
                std::make_shared<ai::ner::ModelEntityExtractor>(ner_bundle);
        } else if (ner_mode == "hybrid") {
            if (ner_bundle.empty())
                throw std::invalid_argument(
                    "--ner-bundle is required for hybrid NER");
            entity_extractor = std::make_shared<ai::ner::HybridEntityExtractor>(
                std::make_shared<ai::ner::RuleEntityExtractor>(),
                std::make_shared<ai::ner::ModelEntityExtractor>(ner_bundle));
        } else if (ner_mode != "off")
            throw std::invalid_argument(
                "--ner must be off, rule, model, or hybrid");
        auto memory = std::make_shared<SqliteMemory>(db);
        auto discussion = std::make_shared<DiscussionEngine>();
        auto tools = std::make_shared<ToolRegistry>();
        tools->add("calculator.calculate", std::make_shared<CalculatorTool>());
        tools->add("file.read", std::make_shared<FileReadTool>(root));
        tools->add("memory.retrieve",
                   std::make_shared<MemoryRetrieveTool>(memory));
        Agent agent(std::make_shared<DefaultInputParser>(reasoner),
                    std::make_shared<DefaultRouter>(),
                    std::make_shared<DefaultPlanner>(reasoner),
                    std::make_shared<DefaultExecutor>(tools, discussion),
                    std::make_shared<DefaultEvaluator>(reasoner),
                    std::make_shared<DefaultAggregator>(), memory, reasoner,
                    entity_extractor);
        std::cout << "ai_cpp Agent " << backend << " backend (/quit to exit)\n";
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
