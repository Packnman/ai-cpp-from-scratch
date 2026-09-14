#pragma once

#include "cuda_function.h"
#include "module.h"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace ai::model {

struct AgentTransformerConfig {
        int vocabulary = 8192;
        int layers = 4;
        int embedding = 256;
        int heads = 4;
        int feed_forward = 1024;
        int context = 1024;
        float dropout = 0.1F;
        std::uint64_t seed = 42;
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
        int _embedding;
        int _heads;
        int _head_size;
        float _dropout;
        std::uint64_t _dropout_counter;
        Tensor _query_weight, _query_bias, _key_weight, _key_bias;
        Tensor _value_weight, _value_bias, _output_weight, _output_bias;
        Linear _query, _key, _value, _output;
        BatchMatMul _query_key, _attention_value;
        Scale _scale;
        Softmax _softmax;
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
        int _embedding;
        int _hidden;
        float _dropout;
        std::uint64_t _dropout_counter;
        Tensor _input_weight, _input_bias, _output_weight, _output_bias;
        Linear _input, _output;
        GELU _gelu;
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
        Tensor _gamma1, _beta1, _gamma2, _beta2;
        LayerNorm _attention_norm, _feed_forward_norm;
        AgentAttention _attention;
        AgentFeedForward _feed_forward;
        Add _residual;
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
        AgentTransformerConfig _config;
        Tensor _tokens, _positions, _gamma, _beta, _output_weight, _output_bias;
        Embedding _token_embedding, _position_embedding;
        LayerNorm _final_norm;
        Linear _output;
        Add _add_position;
        std::vector<std::unique_ptr<AgentTransformerBlock>> _blocks;
};

} // namespace ai::model
