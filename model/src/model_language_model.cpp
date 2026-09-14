#include "ai/model/model_language_model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

namespace ai::model {
namespace {
nlohmann::json special_ids() {
    return {{"pad", 0},          {"unk", 1},           {"bos", 2},
            {"eos", 3},          {"chat", 4},          {"parse", 5},
            {"plan", 6},         {"evaluate", 7},      {"summarize", 8},
            {"memory_write", 9}, {"memory_query", 10}, {"final", 11},
            {"tool", 12}};
}
struct TrainingRestore {
        Module &model;
        bool training;
        ~TrainingRestore() { model.setTraining(training); }
};
} // namespace

void GenerationConfig::validate() const {
    if (!std::isfinite(temperature) || temperature <= 0.0F ||
        !std::isfinite(top_p) || top_p <= 0.0F || top_p > 1.0F ||
        max_tokens <= 0)
        throw std::invalid_argument("invalid generation configuration");
}

void save_bundle(AgentTransformer &model, const AgentTokenizer &tokenizer,
                 const std::filesystem::path &directory) {
    if (model.config().vocabulary !=
        static_cast<int>(tokenizer.vocabulary_size()))
        throw std::invalid_argument("model/tokenizer vocabulary mismatch");
    std::filesystem::create_directories(directory);
    model.save((directory / "weights.bin").c_str());
    tokenizer.save(directory / "tokenizer.model");
    const auto &config = model.config();
    nlohmann::json manifest = {
        {"format", "ai_cpp_agent_model"},
        {"version", 1},
        {"dtype", "float32"},
        {"special_ids", special_ids()},
        {"tokenizer_fingerprint", tokenizer.fingerprint()},
        {"config",
         {{"vocabulary", config.vocabulary},
          {"layers", config.layers},
          {"embedding", config.embedding},
          {"heads", config.heads},
          {"feed_forward", config.feed_forward},
          {"context", config.context},
          {"dropout", config.dropout},
          {"seed", config.seed},
          {"parameter_count", config.parameter_count()}}}};
    const auto temporary = directory / "manifest.json.tmp";
    std::ofstream output(temporary, std::ios::trunc);
    output << manifest.dump(2) << '\n';
    if (!output)
        throw std::runtime_error("cannot write model manifest");
    output.close();
    std::filesystem::rename(temporary, directory / "manifest.json");
}

AgentBundle load_bundle(const std::filesystem::path &directory) {
    std::ifstream input(directory / "manifest.json");
    nlohmann::json manifest;
    if (!input || !(input >> manifest))
        throw std::runtime_error("cannot read model manifest");
    if (manifest.value("format", "") != "ai_cpp_agent_model" ||
        manifest.value("version", 0) != 1 ||
        manifest.value("dtype", "") != "float32" ||
        manifest.at("special_ids") != special_ids())
        throw std::invalid_argument("unsupported agent model bundle");
    auto tokenizer = AgentTokenizer::load(directory / "tokenizer.model");
    if (manifest.at("tokenizer_fingerprint") != tokenizer.fingerprint())
        throw std::invalid_argument("tokenizer fingerprint mismatch");
    const auto &json = manifest.at("config");
    AgentTransformerConfig config;
    config.vocabulary = json.at("vocabulary");
    config.layers = json.at("layers");
    config.embedding = json.at("embedding");
    config.heads = json.at("heads");
    config.feed_forward = json.at("feed_forward");
    config.context = json.at("context");
    config.dropout = json.at("dropout");
    config.seed = json.at("seed");
    if (config.vocabulary != static_cast<int>(tokenizer.vocabulary_size()) ||
        json.at("parameter_count") != config.parameter_count())
        throw std::invalid_argument("noncanonical model manifest");
    auto model = std::make_unique<AgentTransformer>(config);
    model->load((directory / "weights.bin").c_str());
    model->setTraining(false);
    return {std::move(model), std::move(tokenizer)};
}

std::vector<std::int32_t> generate(AgentTransformer &model,
                                   const AgentTokenizer &tokenizer,
                                   const std::vector<std::int32_t> &prompt,
                                   const GenerationConfig &config) {
    config.validate();
    if (model.config().vocabulary !=
        static_cast<int>(tokenizer.vocabulary_size()))
        throw std::invalid_argument(
            "generation model/tokenizer vocabulary mismatch");
    if (prompt.empty() ||
        prompt.size() > static_cast<std::size_t>(model.config().context))
        throw std::length_error("generation prompt exceeds model context");
    TrainingRestore restore{model, model.isTraining()};
    model.setTraining(false);
    std::mt19937 random(static_cast<std::mt19937::result_type>(config.seed));
    auto history = prompt;
    std::vector<std::int32_t> result;
    for (int step = 0; step < config.max_tokens; ++step) {
        const auto sequence = static_cast<int>(history.size());
        auto input = std::make_shared<cunMat>(sequence, 1);
        input->copyFromHost(history.data(), history.size());
        const auto logits = model.forward(input)->_mData.toHost();
        auto logit = [&](int id) {
            const float value =
                logits[static_cast<std::size_t>(id) * sequence + sequence - 1];
            if (!std::isfinite(value))
                throw std::runtime_error("nonfinite generation logit");
            return value;
        };
        std::vector<int> candidates = {AgentTokenizer::eos_id};
        for (int id = AgentTokenizer::special_count;
             id < model.config().vocabulary; ++id)
            candidates.push_back(id);
        std::sort(
            candidates.begin(), candidates.end(),
            [&](int left, int right) { return logit(left) > logit(right); });
        int next = candidates.front();
        if (!config.greedy) {
            const double maximum = logit(candidates.front());
            std::vector<double> weights;
            weights.reserve(candidates.size());
            double total = 0.0;
            for (const int id : candidates) {
                const double weight =
                    std::exp((logit(id) - maximum) / config.temperature);
                weights.push_back(weight);
                total += weight;
            }
            double cumulative = 0.0;
            std::size_t keep = 0;
            for (; keep < weights.size(); ++keep) {
                cumulative += weights[keep];
                if (cumulative / total >= config.top_p) {
                    ++keep;
                    break;
                }
            }
            keep = std::max<std::size_t>(1, keep);
            candidates.resize(keep);
            weights.resize(keep);
            std::discrete_distribution<std::size_t> distribution(
                weights.begin(), weights.end());
            next = candidates[distribution(random)];
        }
        result.push_back(next);
        if (next == AgentTokenizer::eos_id)
            break;
        if (history.size() == static_cast<std::size_t>(model.config().context))
            throw std::length_error(
                "generation reached context limit before EOS");
        history.push_back(next);
    }
    return result;
}

AgentMode ModelLanguageModel::convert(ai::agent::ModelMode mode) {
    return static_cast<AgentMode>(static_cast<int>(mode) + 4);
}
ModelLanguageModel::ModelLanguageModel(const std::filesystem::path &bundle,
                                       GenerationConfig sampling)
    : _bundle(load_bundle(bundle)), _sampling(sampling) {
    _sampling.validate();
}
std::size_t ModelLanguageModel::token_count(std::string_view text) const {
    return _bundle.tokenizer.count(text);
}
std::string ModelLanguageModel::complete(ai::agent::ModelMode mode,
                                         std::string_view prompt) {
    std::vector<std::int32_t> ids = {AgentTokenizer::bos_id,
                                     _bundle.tokenizer.mode_id(convert(mode))};
    const auto text_ids = _bundle.tokenizer.encode(prompt);
    ids.insert(ids.end(), text_ids.begin(), text_ids.end());
    auto generation = _sampling;
    const bool structured = mode != ai::agent::ModelMode::Chat &&
                            mode != ai::agent::ModelMode::Final;
    generation.greedy = structured;
    static constexpr int limits[] = {256, 256, 384, 128, 256,
                                     256, 128, 384, 256};
    generation.max_tokens =
        std::min(generation.max_tokens, limits[static_cast<int>(mode)]);
    generation.seed += _request_counter++;
    const auto room =
        _bundle.model->config().context - static_cast<int>(ids.size());
    if (room <= 0)
        throw std::length_error("model prompt exceeds 1024-token context");
    generation.max_tokens = std::min(generation.max_tokens, room);
    return _bundle.tokenizer.decode(
        generate(*_bundle.model, _bundle.tokenizer, ids, generation));
}
} // namespace ai::model
