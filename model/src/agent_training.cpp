#include "ai/model/agent_training.h"

#include "optimizer_adam.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace ai::model {
namespace {
struct Batch {
        int sequence = 0;
        int batch = 0;
        std::size_t valid = 0;
        std::vector<std::int32_t> inputs, targets;
};
Batch make_batch(const AgentDataset &dataset,
                 const std::vector<std::size_t> &order, std::size_t start,
                 std::size_t requested) {
    const auto end = std::min(order.size(), start + requested);
    Batch result;
    result.batch = static_cast<int>(end - start);
    for (std::size_t i = start; i < end; ++i)
        result.sequence = std::max(
            result.sequence,
            static_cast<int>(dataset.windows().at(order[i]).inputs.size()));
    result.inputs.assign(
        static_cast<std::size_t>(result.sequence) * result.batch, 0);
    result.targets.assign(result.inputs.size(), 0);
    for (int column = 0; column < result.batch; ++column) {
        const auto &window = dataset.windows().at(order[start + column]);
        for (std::size_t token = 0; token < window.inputs.size(); ++token) {
            const auto index = token * result.batch + column;
            result.inputs[index] = window.inputs[token];
            result.targets[index] = window.targets[token];
            if (window.targets[token] != AgentTokenizer::pad_id)
                ++result.valid;
        }
    }
    return result;
}
std::string fingerprint(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    std::uint64_t hash = 1469598103934665603ULL;
    char byte;
    while (input.get(byte)) {
        hash ^= static_cast<unsigned char>(byte);
        hash *= 1099511628211ULL;
    }
    std::ostringstream text;
    text << std::hex << std::setw(16) << std::setfill('0') << hash;
    return text.str();
}
float batch_loss(AgentTransformer &model, const Batch &batch, bool backward) {
    auto inputs = std::make_shared<cunMat>(batch.sequence, batch.batch);
    cunMat targets(batch.sequence, batch.batch);
    inputs->copyFromHost(batch.inputs.data(), batch.inputs.size());
    targets.copyFromHost(batch.targets.data(), batch.targets.size());
    auto loss = model.loss(inputs, targets);
    const float value = loss->_mData.toHost().at(0);
    if (!std::isfinite(value))
        throw std::runtime_error("nonfinite training loss");
    if (backward)
        loss->backward();
    return value;
}
void clip(AgentTransformer &model, float maximum) {
    double sum = 0.0;
    for (const auto &parameter : model.namedParameters())
        for (const float value : parameter.lpTensor->_mGrad.toHost())
            sum += static_cast<double>(value) * value;
    const double norm = std::sqrt(sum);
    if (norm > maximum)
        for (const auto &parameter : model.namedParameters())
            cuda_scale(parameter.lpTensor->_mGrad,
                       static_cast<float>(maximum / norm));
}
} // namespace

void TrainingConfig::validate() const {
    if (steps == 0 || batch_size == 0 || accumulation_steps == 0 ||
        !std::isfinite(learning_rate) || learning_rate <= 0.0F ||
        !std::isfinite(clip_norm) || clip_norm <= 0.0F)
        throw std::invalid_argument("invalid training configuration");
}

ValidationMetrics validate(AgentTransformer &model, const AgentDataset &dataset,
                           std::size_t batch_size) {
    if (batch_size == 0)
        throw std::invalid_argument("validation batch must be positive");
    const bool previous = model.isTraining();
    model.setTraining(false);
    const auto started = std::chrono::steady_clock::now();
    const auto order = dataset.order(0, 0);
    double weighted = 0.0;
    std::size_t valid = 0;
    for (std::size_t start = 0; start < order.size(); start += batch_size) {
        const auto batch = make_batch(dataset, order, start, batch_size);
        weighted += batch_loss(model, batch, false) * batch.valid;
        valid += batch.valid;
    }
    model.setTraining(previous);
    if (!valid)
        throw std::invalid_argument("validation has no target tokens");
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    const double loss = weighted / valid;
    return {loss, std::exp(loss), valid / seconds,
            dataset.padding_ratio(batch_size), valid};
}

void train(AgentTransformer &model, const AgentTokenizer &tokenizer,
           const AgentDataset &training, const AgentDataset &validation,
           const std::filesystem::path &output, const TrainingConfig &config,
           bool resume, std::ostream &log) {
    config.validate();
    if (config.steps % config.accumulation_steps != 0)
        throw std::invalid_argument(
            "steps must end on a gradient-accumulation boundary");
    std::filesystem::create_directories(output / "checkpoint");
    Adam optimizer(&model, config.learning_rate);
    optimizer.init();
    std::size_t first_step = 0, optimized_tokens = 0;
    std::mt19937 shuffle(static_cast<std::mt19937::result_type>(config.seed));
    if (resume) {
        std::ifstream latest_stream(output / "checkpoint/latest.json");
        nlohmann::json latest;
        latest_stream >> latest;
        if (latest.value("format", "") != "ai_cpp_agent_training_latest" ||
            latest.value("version", 0) != 1)
            throw std::runtime_error("invalid latest checkpoint");
        std::ifstream state_stream(output / "checkpoint" /
                                   latest.at("checkpoint").get<std::string>());
        nlohmann::json state;
        state_stream >> state;
        if (state.value("format", "") != "ai_cpp_agent_training_checkpoint" ||
            state.value("version", 0) != 1 ||
            state.at("tokenizer_fingerprint") != tokenizer.fingerprint() ||
            state.at("data_fingerprints").at("train") !=
                training.fingerprint() ||
            state.at("data_fingerprints").at("validation") !=
                validation.fingerprint())
            throw std::runtime_error("incompatible checkpoint");
        const auto weights =
            output / "checkpoint" / state.at("weights").get<std::string>();
        const auto adam =
            output / "checkpoint" / state.at("adam").get<std::string>();
        if (state.at("weights_fingerprint") != fingerprint(weights) ||
            state.at("adam_fingerprint") != fingerprint(adam))
            throw std::runtime_error("checkpoint fingerprint mismatch");
        model.load(weights.c_str());
        optimizer.loadState(adam.string());
        model.set_dropout_counters(
            state.at("dropout_counters").get<std::vector<std::uint64_t>>());
        std::istringstream random_state(
            state.at("shuffle_state").get<std::string>());
        random_state >> shuffle;
        first_step = state.at("step");
        optimized_tokens = state.at("optimized_tokens");
    }
    model.setTraining(true);
    std::size_t pending = 0, completed = 0;
    model.zero_grads();
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t local = 0; local < config.steps; ++local) {
        const auto order = training.order(config.seed, first_step + local);
        const auto batch = make_batch(training, order, 0, config.batch_size);
        const auto loss = batch_loss(model, batch, true);
        optimized_tokens += batch.valid;
        ++pending;
        ++completed;
        (void)shuffle();
        if (pending == config.accumulation_steps || local + 1 == config.steps) {
            for (const auto &parameter : model.namedParameters())
                cuda_scale(parameter.lpTensor->_mGrad, 1.0F / pending);
            clip(model, config.clip_norm);
            optimizer.update();
            model.zero_grads();
            pending = 0;
        }
        log << nlohmann::json{{"split", "train"},
                              {"step", first_step + local + 1},
                              {"loss", loss},
                              {"valid_tokens", batch.valid}}
                   .dump()
            << '\n';
        // Never checkpoint a partially accumulated gradient. A token budget
        // is therefore a soft limit and may exceed one accumulation group.
        if (config.token_budget && optimized_tokens >= config.token_budget &&
            pending == 0)
            break;
    }
    const auto metrics = validate(model, validation, config.batch_size);
    log << nlohmann::json{{"split", "validation"},
                          {"loss", metrics.loss},
                          {"perplexity", metrics.perplexity},
                          {"tokens_per_second", metrics.tokens_per_second},
                          {"padding_ratio", metrics.padding_ratio},
                          {"valid_tokens", metrics.valid_tokens}}
               .dump()
        << '\n';
    save_bundle(model, tokenizer, output);
    const bool is_sft =
        std::any_of(training.windows().begin(), training.windows().end(),
                    [](const AgentWindow &window) {
                        return window.mode != AgentMode::Chat;
                    });
    if (is_sft) {
        std::ifstream input(output / "manifest.json");
        nlohmann::json manifest;
        input >> manifest;
        manifest["sft_mix_weights"] = {
            {"CHAT", 20},        {"PARSE", 15},     {"PLAN", 15},
            {"EVALUATE", 10},    {"SUMMARIZE", 10}, {"MEMORY_WRITE", 10},
            {"MEMORY_QUERY", 5}, {"FINAL", 10},     {"TOOL", 5}};
        const auto temporary = output / "manifest.json.tmp";
        {
            std::ofstream file(temporary);
            file << manifest.dump(2) << '\n';
        }
        std::filesystem::rename(temporary, output / "manifest.json");
    }
    const auto step = first_step + completed;
    const auto stem = "step-" + std::to_string(step);
    const auto weights_name = stem + ".weights.bin";
    const auto adam_name = stem + ".adam.bin";
    model.save((output / "checkpoint" / weights_name).c_str());
    optimizer.saveState((output / "checkpoint" / adam_name).string());
    std::ostringstream random_state;
    random_state << shuffle;
    nlohmann::json state = {
        {"format", "ai_cpp_agent_training_checkpoint"},
        {"version", 1},
        {"step", step},
        {"adam_step", optimizer.step()},
        {"optimized_tokens", optimized_tokens},
        {"weights", weights_name},
        {"adam", adam_name},
        {"weights_fingerprint",
         fingerprint(output / "checkpoint" / weights_name)},
        {"adam_fingerprint", fingerprint(output / "checkpoint" / adam_name)},
        {"tokenizer_fingerprint", tokenizer.fingerprint()},
        {"data_fingerprints",
         {{"train", training.fingerprint()},
          {"validation", validation.fingerprint()}}},
        {"dropout_counters", model.dropout_counters()},
        {"shuffle_state", random_state.str()},
        {"elapsed_seconds", std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - started)
                                .count()},
        {"training",
         {{"seed", config.seed},
          {"batch_size", config.batch_size},
          {"accumulation_steps", config.accumulation_steps},
          {"clip_norm", config.clip_norm},
          {"learning_rate", config.learning_rate},
          {"token_budget", config.token_budget}}}};
    const auto state_name = stem + ".json";
    {
        std::ofstream file(output / "checkpoint" / state_name);
        file << state.dump(2) << '\n';
    }
    const auto temporary = output / "checkpoint/latest.json.tmp";
    {
        std::ofstream file(temporary);
        file << nlohmann::json{{"format", "ai_cpp_agent_training_latest"},
                               {"version", 1},
                               {"checkpoint", state_name}}
                    .dump(2)
             << '\n';
    }
    std::filesystem::rename(temporary, output / "checkpoint/latest.json");
}
} // namespace ai::model
