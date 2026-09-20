#include "TestSupport.hpp"
#include "brain/preprocess/ModelContextRecognizer.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

using namespace ai::brain;

namespace {
std::string environment(const char *name, std::string fallback) {
    const auto *value = std::getenv(name);
    return value && *value ? value : std::move(fallback);
}
} // namespace

int main() {
    brain_test::Suite t{"Qwen context integration"};
    QwenContextBackendConfig config;
    config.endpoint = environment("AI_CPP_QWEN_ENDPOINT", config.endpoint);
    config.model = environment("AI_CPP_QWEN_MODEL", config.model);
    config.timeout = Duration{60'000};
    auto backend = std::make_unique<QwenContextBackend>(config);
    auto logger = std::make_shared<InMemoryLogManager>();
    ModelContextRecognizer recognizer(std::move(backend),
                                      std::make_unique<RuleContextRecognizer>(),
                                      {}, logger);

    const auto result =
        recognizer.recognize("青いボールを取って", steady_now());
    if (result.intent != "command" || result.goals.size() != 1 ||
        result.goals.front().type != "AcquireObject" ||
        !result.goals.front().target) {
        std::cerr << "Qwen result: intent=" << result.intent
                  << " goals=" << result.goals.size();
        if (!result.goals.empty())
            std::cerr << " goal_type=" << result.goals.front().type
                      << " target="
                      << (result.goals.front().target ? "present" : "missing");
        std::cerr << '\n';
        for (const auto &error : logger->errors())
            std::cerr << "Qwen fallback: " << error.cause << '\n';
    }
    t.expect(result.intent == "command" && result.goals.size() == 1 &&
                 result.goals.front().type == "AcquireObject" &&
                 result.goals.front().target,
             "QWEN-CTX-001",
             "running Qwen server returns a validated AcquireObject goal");
    return t.finish();
}
