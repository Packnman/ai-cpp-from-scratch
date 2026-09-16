#pragma once

#include "cuda_function.h"
#include "module.h"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace ai::model {

struct AgentTransformerConfig {
        int vocabulary = 8192; // 語彙に含めるトークン数
        int layers = 4; // Transformerブロックの段数
        int embedding = 256; // トークン埋め込みの次元数
        int heads = 4; // Attentionヘッド数
        int feed_forward = 1024; // Feed Forward層の中間次元数
        int context = 1024; // 入力できる最大トークン数
        float dropout = 0.1F; // Dropoutで無効化する確率
        std::uint64_t seed = 42; // パラメーター初期化の乱数シード
        void validate() const;
        std::uint64_t parameter_count() const noexcept;
};

class AgentAttention final : public Module {
    public:
        AgentAttention(int embedding, int heads, float dropout,
                       std::uint64_t seed);
        void initialize(std::mt19937 &random);
        std::uint64_t dropout_counter() const noexcept {
            return _dropout_counter;
        }
        void set_dropout_counter(std::uint64_t value) noexcept {
            _dropout_counter = value;
        }
        TensorPtr forward(TensorList &inputs) override;

    private:
        int _embedding; // 入出力の埋め込み次元数
        int _heads; // Attentionヘッド数
        int _head_size; // 1ヘッド当たりの次元数
        float _dropout; // Attention重みへ適用するDropout率
        std::uint64_t _dropout_counter; // 再現用のDropout呼び出し位置
        Tensor _query_weight, _query_bias, _key_weight, _key_bias; // Query・Key射影の重みとバイアス
        Tensor _value_weight, _value_bias, _output_weight, _output_bias; // Value・出力射影の重みとバイアス
        Linear _query, _key, _value, _output; // Query・Key・Value・出力の線形層
        BatchMatMul _query_key, _attention_value; // 注意スコアと重み付きValueの積演算
        Scale _scale; // 注意スコアの次元スケーリング
        Softmax _softmax; // 注意スコアの確率化
};

class AgentFeedForward final : public Module {
    public:
        AgentFeedForward(int embedding, int hidden, float dropout,
                         std::uint64_t seed);
        void initialize(std::mt19937 &random);
        std::uint64_t dropout_counter() const noexcept {
            return _dropout_counter;
        }
        void set_dropout_counter(std::uint64_t value) noexcept {
            _dropout_counter = value;
        }
        TensorPtr forward(TensorList &inputs) override;

    private:
        int _embedding; // 入出力の埋め込み次元数
        int _hidden; // 中間層の次元数
        float _dropout; // 中間表現へ適用するDropout率
        std::uint64_t _dropout_counter; // 再現用のDropout呼び出し位置
        Tensor _input_weight, _input_bias, _output_weight, _output_bias; // 入出力射影の重みとバイアス
        Linear _input, _output; // 中間層への入力変換と出力変換
        GELU _gelu; // 中間層の活性化関数
};

class AgentTransformerBlock final : public Module {
    public:
        AgentTransformerBlock(int embedding, int heads, int hidden,
                              float dropout, std::uint64_t seed);
        void initialize(std::mt19937 &random);
        std::vector<std::uint64_t> dropout_counters() const;
        void set_dropout_counters(const std::vector<std::uint64_t> &values);
        TensorPtr forward(TensorList &inputs) override;

    private:
        Tensor _gamma1, _beta1, _gamma2, _beta2; // 2つのLayerNormのスケールとバイアス
        LayerNorm _attention_norm, _feed_forward_norm; // Attention・Feed Forward入力の正規化層
        AgentAttention _attention; // Multi-Head Attention層
        AgentFeedForward _feed_forward; // 位置ごとのFeed Forward層
        Add _residual; // 残差接続の加算演算
};

class AgentTransformer final : public Model {
    public:
        explicit AgentTransformer(const AgentTransformerConfig &config);
        const AgentTransformerConfig &config() const noexcept {
            return _config;
        }
        TensorPtr forward(TensorList &inputs) override;
        TensorPtr forward(const std::shared_ptr<const cunMat> &ids);
        TensorPtr loss(const std::shared_ptr<const cunMat> &ids,
                       const cunMat &targets, int ignored_id = 0);
        std::vector<std::uint64_t> dropout_counters() const;
        void set_dropout_counters(const std::vector<std::uint64_t> &values);

    private:
        AgentTransformerConfig _config; // モデル構造と生成時の基本設定
        Tensor _tokens, _positions, _gamma, _beta, _output_weight, _output_bias; // 埋め込み・正規化・出力射影のパラメーター
        Embedding _token_embedding, _position_embedding; // トークンIDと位置の埋め込み層
        LayerNorm _final_norm; // 出力直前の正規化層
        Linear _output; // 隠れ表現を語彙ロジットへ射影する層
        Add _add_position; // トークン埋め込みと位置埋め込みの加算
        std::vector<std::unique_ptr<AgentTransformerBlock>> _blocks; // 順に適用するTransformerブロック
};

} // namespace ai::model
