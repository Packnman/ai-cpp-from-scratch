#pragma once

#include "ai/model/agent_tokenizer.h"

#include <filesystem>
#include <string>
#include <vector>

namespace ai::model {

struct AgentWindow {
        std::vector<std::int32_t> inputs;
        std::vector<std::int32_t> targets;
        AgentMode mode = AgentMode::Chat;
        std::string document_id;
        std::size_t valid_tokens = 0;
};

class AgentDataset {
    public:
        static AgentDataset jawiki(const std::filesystem::path &train_jsonl,
                                   const AgentTokenizer &tokenizer,
                                   std::size_t context,
                                   std::size_t token_limit = 0);
        static AgentDataset sft(const std::filesystem::path &train_jsonl,
                                const AgentTokenizer &tokenizer,
                                std::size_t context);
        const std::vector<AgentWindow> &windows() const noexcept {
            return _windows;
        }
        std::vector<std::size_t> order(std::uint64_t seed,
                                       std::size_t epoch) const;
        double padding_ratio(std::size_t batch_size) const;
        std::string fingerprint() const;

    private:
        std::vector<AgentWindow> _windows;
};

void generate_sft_corpus(const std::filesystem::path &conversation_train,
                         const std::filesystem::path &output,
                         std::uint64_t seed = 42);

} // namespace ai::model
