#include "ai/model/agent_transformer.h"

#include "cuda_function_IndexCrossEntropy.h"

#include <cmath>
#include <stdexcept>

namespace ai::model {
namespace {
template <class Operation, class... Arguments>
TensorPtr graph_apply(const TensorList &inputs, Arguments &&...arguments)
{
    auto operation =
        std::make_shared<Operation>(std::forward<Arguments>(arguments)...);
    auto result = (*operation)(inputs);
    result->_spContext->_spFunction = std::move(operation);
    return result;
}
TensorPtr reshape(const TensorPtr &input, std::vector<std::int64_t> shape)
{
    return graph_apply<Reshape>(TensorList{input}, std::move(shape));
}
TensorPtr permute(const TensorPtr &input, std::vector<std::size_t> axes)
{
    return graph_apply<Permute>(TensorList{input}, std::move(axes));
}
void initialize(Tensor &tensor, int fan_in, std::mt19937 &random)
{
    std::normal_distribution<float> distribution(
        0.0F, 1.0F / std::sqrt(static_cast<float>(fan_in)));
    std::vector<float> values(tensor._mData.numel());
    for (auto &value : values)
        value = distribution(random);
    tensor._mData.copyFromHost(values.data(), values.size());
}
class OwnedMask final : public Mask
{
    public:
        explicit OwnedMask(TensorPtr mask)
            : Mask(mask.get()), _mask(std::move(mask)) {}

    private:
        TensorPtr _mask;
};
} // namespace

void AgentTransformerConfig::validate() const
{
    if (vocabulary < 269 || layers <= 0 || embedding <= 0 || heads <= 0 ||
        embedding % heads != 0 || feed_forward <= 0 || context <= 0 ||
        !std::isfinite(dropout) || dropout < 0.0F || dropout >= 1.0F)
        throw std::invalid_argument("invalid AgentTransformer configuration");
}

std::uint64_t AgentTransformerConfig::parameter_count() const noexcept
{
    const auto e = static_cast<std::uint64_t>(embedding);
    const auto v = static_cast<std::uint64_t>(vocabulary);
    const auto f = static_cast<std::uint64_t>(feed_forward);
    const auto per_layer = 4 * (e * e + e) + 2 * e * f + f + e + 4 * e;
    return e * v + e * context + 2 * e + v * e + v + layers * per_layer;
}

AgentAttention::AgentAttention(int embedding, int heads, float dropout,
                               std::uint64_t seed)
    : _embedding(embedding),
      _heads(heads),
      _head_size(heads ? embedding / heads : 0),
      _dropout(dropout),
      _dropout_counter(seed),
      _query_weight(embedding, embedding),
      _query_bias(embedding, 1),
      _key_weight(embedding, embedding),
      _key_bias(embedding, 1),
      _value_weight(embedding, embedding),
      _value_bias(embedding, 1),
      _output_weight(embedding, embedding),
      _output_bias(embedding, 1),
      _query(&_query_weight, &_query_bias),
      _key(&_key_weight, &_key_bias),
      _value(&_value_weight, &_value_bias),
      _output(&_output_weight, &_output_bias),
      _scale(_head_size ? 1.0F / std::sqrt(static_cast<float>(_head_size))
                        : 1.0F),
      _softmax(1)
{
    if (embedding <= 0 || heads <= 0 || embedding % heads != 0)
        throw std::invalid_argument("invalid attention dimensions");
    registerParameter("query_weight", &_query_weight);
    registerParameter("query_bias", &_query_bias);
    registerParameter("key_weight", &_key_weight);
    registerParameter("key_bias", &_key_bias);
    registerParameter("value_weight", &_value_weight);
    registerParameter("value_bias", &_value_bias);
    registerParameter("output_weight", &_output_weight);
    registerParameter("output_bias", &_output_bias);
}

void AgentAttention::initialize(std::mt19937 &random)
{
    for (auto *weight :
         {&_query_weight, &_key_weight, &_value_weight, &_output_weight})
        ai::model::initialize(*weight, _embedding, random);
    for (auto *bias : {&_query_bias, &_key_bias, &_value_bias, &_output_bias})
        cuda_fill(bias->_mData, 0.0F);
}

TensorPtr AgentAttention::forward(TensorList &inputs)
{
    if (inputs.size() != 1 || !inputs[0] || inputs[0]->_mData.dim() != 3 ||
        inputs[0]->_mData.size(0) != _embedding)
        throw std::invalid_argument(
            "attention expects [embedding, sequence, batch]");
    const int sequence = static_cast<int>(inputs[0]->_mData.size(1));
    const int batch = static_cast<int>(inputs[0]->_mData.size(2));
    const std::vector<std::int64_t> heads = {_heads, _head_size, sequence,
                                             batch};
    auto query = permute(reshape(_query(inputs), heads), {2, 1, 0, 3});
    auto key = permute(reshape(_key(inputs), heads), {1, 2, 0, 3});
    auto value = permute(reshape(_value(inputs), heads), {2, 1, 0, 3});
    auto scores = _scale({_query_key({query, key})});
    auto mask = std::make_shared<Tensor>(sequence, sequence);
    std::vector<float> values(static_cast<std::size_t>(sequence) * sequence,
                              0.0F);
    for (int q = 0; q < sequence; ++q)
        for (int k = 0; k <= q; ++k)
            values[static_cast<std::size_t>(q) * sequence + k] = 1.0F;
    mask->_mData.copyFromHost(values.data(), values.size());
    auto masked = graph_apply<OwnedMask>(TensorList{scores}, mask);
    auto weights = _softmax(TensorList{masked});
    if (isTraining() && _dropout > 0.0F)
        weights = graph_apply<Dropout>(TensorList{weights}, _dropout,
                                       _dropout_counter++);
    auto attended = _attention_value(TensorList{weights, value});
    auto merged =
        reshape(permute(attended, {2, 1, 0, 3}), {_embedding, sequence, batch});
    return _output(TensorList{merged});
}

AgentFeedForward::AgentFeedForward(int embedding, int hidden, float dropout,
                                   std::uint64_t seed)
    : _embedding(embedding),
      _hidden(hidden),
      _dropout(dropout),
      _dropout_counter(seed),
      _input_weight(hidden, embedding),
      _input_bias(hidden, 1),
      _output_weight(embedding, hidden),
      _output_bias(embedding, 1),
      _input(&_input_weight, &_input_bias),
      _output(&_output_weight, &_output_bias)
{
    registerParameter("input_weight", &_input_weight);
    registerParameter("input_bias", &_input_bias);
    registerParameter("output_weight", &_output_weight);
    registerParameter("output_bias", &_output_bias);
}
void AgentFeedForward::initialize(std::mt19937 &random)
{
    ai::model::initialize(_input_weight, _embedding, random);
    ai::model::initialize(_output_weight, _hidden, random);
    cuda_fill(_input_bias._mData, 0.0F);
    cuda_fill(_output_bias._mData, 0.0F);
}
TensorPtr AgentFeedForward::forward(TensorList &inputs)
{
    auto result = _output({_gelu({_input(inputs)})});
    if (isTraining() && _dropout > 0.0F)
        result = graph_apply<Dropout>(TensorList{result}, _dropout,
                                      _dropout_counter++);
    return result;
}

AgentTransformerBlock::AgentTransformerBlock(int embedding, int heads,
                                             int hidden, float dropout,
                                             std::uint64_t seed)
    : _gamma1(embedding, 1),
      _beta1(embedding, 1),
      _gamma2(embedding, 1),
      _beta2(embedding, 1),
      _attention_norm(&_gamma1, &_beta1),
      _feed_forward_norm(&_gamma2, &_beta2),
      _attention(embedding, heads, dropout, seed),
      _feed_forward(embedding, hidden, dropout, seed + 1)
{
    registerParameter("gamma1", &_gamma1);
    registerParameter("beta1", &_beta1);
    registerParameter("gamma2", &_gamma2);
    registerParameter("beta2", &_beta2);
    registerModule("attention", &_attention);
    registerModule("feed_forward", &_feed_forward);
}
void AgentTransformerBlock::initialize(std::mt19937 &random)
{
    cuda_fill(_gamma1._mData, 1.0F);
    cuda_fill(_gamma2._mData, 1.0F);
    cuda_fill(_beta1._mData, 0.0F);
    cuda_fill(_beta2._mData, 0.0F);
    _attention.initialize(random);
    _feed_forward.initialize(random);
}
TensorPtr AgentTransformerBlock::forward(TensorList &inputs)
{
    auto normalized = _attention_norm(inputs);
    TensorList attention_inputs{normalized};
    auto first = _residual({inputs.at(0), _attention.forward(attention_inputs)});
    normalized = _feed_forward_norm({first});
    TensorList feed_forward_inputs{normalized};
    return _residual({first, _feed_forward.forward(feed_forward_inputs)});
}
std::vector<std::uint64_t> AgentTransformerBlock::dropout_counters() const
{
    return {_attention.dropout_counter(), _feed_forward.dropout_counter()};
}
void AgentTransformerBlock::set_dropout_counters(
    const std::vector<std::uint64_t> &values
)
{
    if (values.size() != 2)
        throw std::invalid_argument("invalid block dropout state");
    _attention.set_dropout_counter(values[0]);
    _feed_forward.set_dropout_counter(values[1]);
}

AgentTransformer::AgentTransformer(const AgentTransformerConfig &config)
    : _config(config),
      _tokens(config.embedding, config.vocabulary),
      _positions(config.embedding, config.context),
      _gamma(config.embedding, 1),
      _beta(config.embedding, 1),
      _output_weight(config.vocabulary, config.embedding),
      _output_bias(config.vocabulary, 1),
      _token_embedding(&_tokens),
      _position_embedding(&_positions),
      _final_norm(&_gamma, &_beta),
      _output(&_output_weight, &_output_bias)
{
    _config.validate();
    registerParameter("tokens", &_tokens);
    registerParameter("positions", &_positions);
    registerParameter("gamma", &_gamma);
    registerParameter("beta", &_beta);
    registerParameter("output_weight", &_output_weight);
    registerParameter("output_bias", &_output_bias);
    std::mt19937 random(static_cast<std::mt19937::result_type>(_config.seed));
    ai::model::initialize(_tokens, _config.embedding, random);
    ai::model::initialize(_positions, _config.embedding, random);
    ai::model::initialize(_output_weight, _config.embedding, random);
    cuda_fill(_gamma._mData, 1.0F);
    cuda_fill(_beta._mData, 0.0F);
    cuda_fill(_output_bias._mData, 0.0F);
    for (int layer = 0; layer < _config.layers; ++layer)
    {
        auto block = std::make_unique<AgentTransformerBlock>(
            _config.embedding, _config.heads, _config.feed_forward,
            _config.dropout, _config.seed + 2 * layer);
        block->initialize(random);
        registerModule("block" + std::to_string(layer), block.get());
        _blocks.push_back(std::move(block));
    }
}
TensorPtr AgentTransformer::forward(TensorList &)
{
    throw std::invalid_argument("AgentTransformer requires integer token IDs");
}
TensorPtr AgentTransformer::forward(const std::shared_ptr<const cunMat> &ids)
{
    if (!ids || ids->dim() != 2 || ids->size(0) <= 0 ||
        ids->size(0) > _config.context || ids->size(1) <= 0)
    {
        throw std::invalid_argument(
            "token IDs must have shape [1..context, positive batch]"
        );
    }
    auto stable_ids = std::make_shared<cunMat>(*ids);
    auto positions = std::make_shared<cunMat>(ids->shape());
    std::vector<std::int32_t> position_ids(positions->numel());
    for (std::size_t i = 0; i < position_ids.size(); ++i)
    {
        position_ids[i] = static_cast<std::int32_t>(i / static_cast<std::size_t>(ids->size(1)));
    }
    positions->copyFromHost(position_ids.data(), position_ids.size());
    auto result = _add_position( {_token_embedding(stable_ids), _position_embedding(positions)} );
    for (auto &block : _blocks)
    {
        TensorList inputs{result};
        result = block->forward(inputs);
    }
    return _output({_final_norm({result})});
}
TensorPtr AgentTransformer::loss(
    const std::shared_ptr<const cunMat> &ids,
    const cunMat &targets, int ignored_id
)
{
    return graph_apply<IndexCrossEntropy>(
        TensorList{forward(ids)}, targets,
        ignored_id
    );
}
std::vector<std::uint64_t> AgentTransformer::dropout_counters() const
{
    std::vector<std::uint64_t> result;
    for (const auto &block : _blocks)
    {
        auto values = block->dropout_counters();
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}
void AgentTransformer::set_dropout_counters(
    const std::vector<std::uint64_t> &values
)
{
    if (values.size() != 2 * _blocks.size())
        throw std::invalid_argument("dropout state count mismatch");
    for (std::size_t i = 0; i < _blocks.size(); ++i)
        _blocks[i]->set_dropout_counters({values[2 * i], values[2 * i + 1]});
}
} // namespace ai::model
