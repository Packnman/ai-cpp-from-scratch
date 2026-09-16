#pragma once

#include "ai/model/agent_dataset.h"
#include "ai/model/model_language_model.h"

#include <filesystem>
#include <iosfwd>

namespace ai::model {

struct TrainingConfig {
    std::size_t steps = 1; // 実行する最適化ステップ数
    std::size_t batch_size = 1; // 1ミニバッチ当たりの系列数
    std::size_t accumulation_steps = 1; // 更新前に勾配を蓄積する回数
    std::size_t token_budget = 0; // 処理する最大トークン数（0は無制限）
    float learning_rate = 3.0e-4F; // Adam最適化の最大学習率
    float minimum_learning_rate = 3.0e-5F; // cosine decay後の学習率
    std::size_t warmup_updates = 0; // 線形warmupするAdam更新数
    std::size_t decay_updates = 0; // cosine decay完了時のAdam更新番号
    std::uint64_t max_device_bytes = 0; // 推定peak VRAM上限（0は無制限）
    float clip_norm = 1.0F; // 勾配クリッピングの最大ノルム
    std::uint64_t seed = 42; // データ順序などに用いる乱数シード
    void validate() const;
};

struct ValidationMetrics {
    double loss = 0.0; // 有効トークン当たりの平均損失
    double perplexity = 0.0; // 損失から算出したパープレキシティ
    double tokens_per_second = 0.0; // 1秒当たりの処理トークン数
    double padding_ratio = 0.0; // 全トークンに占めるパディング率
    std::size_t valid_tokens = 0; // 評価に使った有効トークン数
    std::size_t input_bytes = 0; // 評価テキストのUTF-8 byte数
    double bits_per_byte = 0.0; // tokenizer非依存のbase-2交差entropy
};

struct ShardedTrainingConfig {
    TrainingConfig training; // 各シャードに適用する学習設定
    std::size_t epochs = 1; // 全シャードを巡回する回数
    std::size_t max_shards = 1; // 1回の呼び出しで処理する最大シャード数
};

ValidationMetrics validate(
    AgentTransformer &model,
    const AgentDataset &dataset,
    std::size_t batch_size
);
void train(
    AgentTransformer &model,
    const AgentTokenizer &tokenizer,
    const AgentDataset &training,
    const AgentDataset &validation,
    const std::filesystem::path &output,
    const TrainingConfig &config,
    bool resume, std::ostream &log
);
void pretrain_sharded(
    AgentTransformer &model,
    const AgentTokenizer &tokenizer,
    const AgentDataset &validation,
    const std::filesystem::path &shard_manifest,
    const std::filesystem::path &output,
    const ShardedTrainingConfig &config, bool resume,
    std::ostream &log
);

} // namespace ai::model
