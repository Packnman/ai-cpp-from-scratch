#pragma once

#include "ai/agent/components.h"
#include "ai/model/agent_tokenizer.h"
#include "ai/model/agent_transformer.h"

#include <filesystem>
#include <memory>
#include <random>

namespace ai::model {

struct GenerationConfig {
        float temperature = 0.8F; // サンプリング分布の温度
        float top_p = 0.9F; // nucleus samplingの累積確率閾値
        std::uint64_t seed = 42; // 生成に用いる乱数シード
        int max_tokens = 256; // 生成する最大トークン数
        bool greedy = false; // 常に最高確率のトークンを選ぶか
        void validate() const;
};

struct AgentBundle {
        std::unique_ptr<AgentTransformer> model; // 学習済みTransformerモデル
        AgentTokenizer tokenizer; // モデルと組み合わせるトークナイザー
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
        AgentBundle _bundle; // 推論に用いるモデルとトークナイザー
        GenerationConfig _sampling; // 応答生成時のサンプリング設定
        std::uint64_t _request_counter = 0; // シードをずらすための処理要求数
};

} // namespace ai::model
