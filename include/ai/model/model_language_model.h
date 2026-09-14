#pragma once

#include "ai/agent/components.h"
#include "ai/model/agent_tokenizer.h"
#include "ai/model/agent_transformer.h"

#include <filesystem>
#include <memory>
#include <random>

namespace ai::model {

struct GenerationConfig {
        float temperature = 0.8F;
        float top_p = 0.9F;
        std::uint64_t seed = 42;
        int max_tokens = 256;
        bool greedy = false;
        void validate() const;
};

struct AgentBundle {
        std::unique_ptr<AgentTransformer> model;
        AgentTokenizer tokenizer;
};

void save_bundle(AgentTransformer &model, const AgentTokenizer &tokenizer,
                 const std::filesystem::path &directory);
AgentBundle load_bundle(const std::filesystem::path &directory);
std::vector<std::int32_t> generate(AgentTransformer &model,
                                   const AgentTokenizer &tokenizer,
                                   const std::vector<std::int32_t> &prompt,
                                   const GenerationConfig &config);

class ModelLanguageModel final : public ai::agent::ILanguageModel {
    public:
        explicit ModelLanguageModel(const std::filesystem::path &bundle,
                                    GenerationConfig sampling = {});
        std::string complete(ai::agent::ModelMode mode,
                             std::string_view prompt) override;
        std::size_t token_count(std::string_view text) const override;
        const AgentTransformerConfig &model_config() const noexcept {
            return _bundle.model->config();
        }

    private:
        static AgentMode convert(ai::agent::ModelMode mode);
        AgentBundle _bundle;
        GenerationConfig _sampling;
        std::uint64_t _request_counter = 0;
};

} // namespace ai::model
