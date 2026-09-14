#pragma once

#include "ai/model/agent_dataset.h"
#include "ai/model/model_language_model.h"

#include <filesystem>
#include <iosfwd>

namespace ai::model {

struct TrainingConfig {
        std::size_t steps = 1;
        std::size_t batch_size = 1;
        std::size_t accumulation_steps = 1;
        std::size_t token_budget = 0;
        float learning_rate = 3.0e-4F;
        float clip_norm = 1.0F;
        std::uint64_t seed = 42;
        void validate() const;
};

struct ValidationMetrics {
        double loss = 0.0;
        double perplexity = 0.0;
        double tokens_per_second = 0.0;
        double padding_ratio = 0.0;
        std::size_t valid_tokens = 0;
};

ValidationMetrics validate(AgentTransformer &model, const AgentDataset &dataset,
                           std::size_t batch_size);
void train(AgentTransformer &model, const AgentTokenizer &tokenizer,
           const AgentDataset &training, const AgentDataset &validation,
           const std::filesystem::path &output, const TrainingConfig &config,
           bool resume, std::ostream &log);

} // namespace ai::model
