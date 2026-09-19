#include "ai/model/agent_training.h"

#include "cuda_memory.h"
#include "optimizer_adam.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
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
std::string fingerprint_text(std::string_view bytes) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream text;
    text << std::hex << std::setw(16) << std::setfill('0') << hash;
    return text.str();
}
void atomic_json(const std::filesystem::path &path,
                 const nlohmann::json &value) {
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::trunc);
        output << value.dump(2) << '\n';
        if (!output)
            throw std::runtime_error("cannot write " + path.string());
    }
    std::filesystem::rename(temporary, path);
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
float scheduled_learning_rate(const TrainingConfig &config,
                              std::uint64_t update) {
    if (config.warmup_updates && update <= config.warmup_updates)
        return config.learning_rate * static_cast<float>(update) /
               static_cast<float>(config.warmup_updates);
    if (config.decay_updates > config.warmup_updates &&
        update < config.decay_updates) {
        const double progress =
            static_cast<double>(update - config.warmup_updates) /
            static_cast<double>(config.decay_updates - config.warmup_updates);
        const double weight =
            0.5 * (1.0 + std::cos(3.14159265358979323846 * progress));
        return static_cast<float>(
            config.minimum_learning_rate +
            (config.learning_rate - config.minimum_learning_rate) * weight);
    }
    return config.decay_updates ? config.minimum_learning_rate
                                : config.learning_rate;
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
        !std::isfinite(minimum_learning_rate) ||
        minimum_learning_rate <= 0.0F ||
        minimum_learning_rate > learning_rate ||
        (decay_updates && decay_updates <= warmup_updates) ||
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
    const auto input_bytes = dataset.input_bytes();
    const double bits_per_byte =
        input_bytes ? loss * static_cast<double>(valid) /
                          static_cast<double>(input_bytes) / std::log(2.0)
                    : 0.0;
    return {loss,
            std::exp(loss),
            valid / seconds,
            dataset.padding_ratio(batch_size),
            valid,
            input_bytes,
            bits_per_byte};
}

void train(AgentTransformer &model, const AgentTokenizer &tokenizer,
           const AgentDataset &training, const AgentDataset &validation,
           const std::filesystem::path &output, const TrainingConfig &config,
           bool resume, std::ostream &log) {
    config.validate();
    if (!training.sft_profile().empty() &&
        training.sft_profile() != validation.sft_profile())
        throw std::invalid_argument("train and validation SFT profiles differ");
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
    const bool is_sft = !training.sft_profile().empty();
    const auto jawiki_batches =
        (training.windows().size() + config.batch_size - 1) / config.batch_size;
    std::size_t cached_epoch = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> cached_order;
    model.zero_grads();
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t local = 0; local < config.steps; ++local) {
        const auto global_step = first_step + local;
        Batch batch;
        if (is_sft) {
            const auto order = training.order(config.seed, global_step);
            batch = make_batch(training, order, 0, config.batch_size);
        } else {
            const auto epoch = global_step / jawiki_batches;
            if (epoch != cached_epoch) {
                cached_order = training.order(config.seed, epoch);
                cached_epoch = epoch;
            }
            const auto cursor =
                (global_step % jawiki_batches) * config.batch_size;
            batch =
                make_batch(training, cached_order, cursor, config.batch_size);
        }
        const auto loss = batch_loss(model, batch, true);
        optimized_tokens += batch.valid;
        ++pending;
        ++completed;
        (void)shuffle();
        if (pending == config.accumulation_steps || local + 1 == config.steps) {
            for (const auto &parameter : model.namedParameters())
                cuda_scale(parameter.lpTensor->_mGrad, 1.0F / pending);
            clip(model, config.clip_norm);
            optimizer.setLearningRate(
                scheduled_learning_rate(config, optimizer.step() + 1));
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
                          {"valid_tokens", metrics.valid_tokens},
                          {"input_bytes", metrics.input_bytes},
                          {"bits_per_byte", metrics.bits_per_byte}}
               .dump()
        << '\n';
    save_bundle(model, tokenizer, output);
    if (is_sft) {
        std::ifstream input(output / "manifest.json");
        nlohmann::json manifest;
        input >> manifest;
        manifest["sft_mix_weights"] =
            training.sft_profile() == "agent"
                ? nlohmann::json{{"CHAT", 20},      {"PARSE", 15},
                                 {"PLAN", 15},      {"EVALUATE", 10},
                                 {"SUMMARIZE", 20}, {"MEMORY_WRITE", 10},
                                 {"FINAL", 10}}
                : nlohmann::json{{"CHAT", 100}};
        manifest["trained_modes"] = training.trained_modes();
        manifest["sft_profile"] = training.sft_profile();
        nlohmann::json sources = nlohmann::json::array();
        for (const auto &source : training.source_provenance())
            sources.push_back(
                {{"name", source.name},
                 {"revision", source.revision},
                 {"converter_version", source.converter_version}});
        manifest["sft_generation"] = {{"generator", "local-template-v3"},
                                      {"converter_version", "summary-sft-v3"},
                                      {"seed", training.generation_seed()},
                                      {"sources", std::move(sources)}};
        manifest["sft_exclusions"] = {
            {"train", {{"context_overflow", training.excluded_too_long()}}},
            {"validation",
             {{"context_overflow", validation.excluded_too_long()}}}};
        manifest["split_fingerprints"] = {
            {"train", training.fingerprint()},
            {"validation", validation.fingerprint()}};
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

void pretrain_sharded(AgentTransformer &model, const AgentTokenizer &tokenizer,
                      const AgentDataset &validation,
                      const std::filesystem::path &shard_manifest,
                      const std::filesystem::path &output,
                      const ShardedTrainingConfig &config, bool resume,
                      std::ostream &log) {
    config.training.validate();
    if (config.epochs != 1)
        throw std::invalid_argument(
            "sharded pretraining currently requires --epochs 1");
    if (!config.max_shards)
        throw std::invalid_argument("max shards must be positive");
    if (!validation.sft_profile().empty())
        throw std::invalid_argument(
            "sharded pretraining validation must be Jawiki");

    std::ifstream manifest_stream(shard_manifest);
    nlohmann::json corpus;
    if (!manifest_stream || !(manifest_stream >> corpus) ||
        corpus.value("format", "") != "ai_cpp_jawiki_shards" ||
        corpus.value("version", 0) != 1 || !corpus.contains("shards") ||
        corpus.at("shards").empty())
        throw std::runtime_error("invalid Jawiki shard manifest");
    const auto recorded_manifest_fingerprint =
        corpus.at("manifest_fingerprint").get<std::string>();
    auto fingerprint_payload = corpus;
    fingerprint_payload.erase("manifest_fingerprint");
    const auto serialized_manifest = fingerprint_payload.dump();
    if (fingerprint_text(serialized_manifest) != recorded_manifest_fingerprint)
        throw std::runtime_error("Jawiki shard manifest fingerprint mismatch");
    const auto shard_root = shard_manifest.parent_path();
    std::uintmax_t covered_bytes = 0;
    std::size_t covered_documents = 0;
    for (std::size_t index = 0; index < corpus.at("shards").size(); ++index) {
        const auto &entry = corpus.at("shards").at(index);
        const auto relative =
            std::filesystem::path(entry.at("file").get<std::string>());
        const auto path = shard_root / relative;
        if (entry.at("index") != index || relative.is_absolute() ||
            relative.has_parent_path() ||
            !std::filesystem::is_regular_file(path) ||
            std::filesystem::file_size(path) != entry.at("bytes") ||
            fingerprint(path) != entry.at("fingerprint").get<std::string>())
            throw std::runtime_error("Jawiki shard is missing or modified");
        covered_bytes += entry.at("bytes").get<std::uintmax_t>();
        covered_documents += entry.at("documents").get<std::size_t>();
    }
    if (covered_bytes != corpus.at("source").at("bytes") ||
        covered_documents != corpus.at("source").at("documents"))
        throw std::runtime_error(
            "Jawiki shard manifest has incomplete coverage");

    const nlohmann::json training_parameters = {
        {"seed", config.training.seed},
        {"batch_size", config.training.batch_size},
        {"accumulation_steps", config.training.accumulation_steps},
        {"clip_norm", config.training.clip_norm},
        {"learning_rate", config.training.learning_rate},
        {"minimum_learning_rate", config.training.minimum_learning_rate},
        {"warmup_updates", config.training.warmup_updates},
        {"decay_updates", config.training.decay_updates},
        {"max_device_bytes", config.training.max_device_bytes}};
    const auto &model_config = model.config();
    const nlohmann::json run_identity = {
        {"format", "ai_cpp_sharded_pretraining_run"},
        {"version", 1},
        {"corpus_manifest_fingerprint", recorded_manifest_fingerprint},
        {"tokenizer_fingerprint", tokenizer.fingerprint()},
        {"validation_fingerprint", validation.fingerprint()},
        {"training", training_parameters},
        {"model",
         {{"vocabulary", model_config.vocabulary},
          {"layers", model_config.layers},
          {"embedding", model_config.embedding},
          {"heads", model_config.heads},
          {"feed_forward", model_config.feed_forward},
          {"context", model_config.context},
          {"dropout", model_config.dropout},
          {"seed", model_config.seed}}}};
    const auto checkpoint_dir = output / "checkpoint";
    if (resume) {
        std::ifstream run_stream(output / "run.json");
        nlohmann::json existing_run;
        if (!run_stream || !(run_stream >> existing_run) ||
            existing_run != run_identity)
            throw std::runtime_error("incompatible sharded training run");
    } else if (std::filesystem::exists(output) &&
               (!std::filesystem::is_directory(output) ||
                std::filesystem::directory_iterator(output) !=
                    std::filesystem::directory_iterator())) {
        std::ifstream run_stream(output / "run.json");
        nlohmann::json existing_run;
        if (!run_stream || !(run_stream >> existing_run) ||
            existing_run != run_identity)
            throw std::runtime_error(
                "pretraining output already exists; refusing a new start");
    }
    std::filesystem::create_directories(checkpoint_dir);
    if (!resume)
        atomic_json(output / "run.json", run_identity);
    Adam optimizer(&model, config.training.learning_rate);
    optimizer.init();
    const auto initial_memory = cu_memory::statistics();
    const auto initial_device_used =
        initial_memory.deviceTotalBytes - initial_memory.deviceFreeBytes;
    const auto non_pool_device_bytes =
        initial_device_used > initial_memory.usedBytes
            ? initial_device_used - initial_memory.usedBytes
            : 0;
    std::mt19937 shuffle(
        static_cast<std::mt19937::result_type>(config.training.seed));
    std::size_t next_shard = 0, step = 0, optimized_tokens = 0;
    double elapsed_seconds = 0.0;
    std::vector<std::size_t> completed_shards;
    std::vector<std::size_t> shard_window_counts;
    nlohmann::json validation_history = nlohmann::json::array();
    if (resume) {
        std::ifstream latest_stream(checkpoint_dir / "latest.json");
        nlohmann::json latest;
        if (!latest_stream || !(latest_stream >> latest) ||
            latest.value("format", "") != "ai_cpp_agent_training_latest" ||
            latest.value("version", 0) != 2)
            throw std::runtime_error("invalid sharded training checkpoint");
        std::ifstream state_stream(checkpoint_dir /
                                   latest.at("checkpoint").get<std::string>());
        nlohmann::json state;
        if (!state_stream || !(state_stream >> state) ||
            state.value("format", "") != "ai_cpp_agent_training_checkpoint" ||
            state.value("version", 0) != 2 ||
            state.at("corpus_manifest_fingerprint") !=
                recorded_manifest_fingerprint ||
            state.at("corpus_fingerprint") !=
                corpus.at("source").at("fingerprint") ||
            state.at("tokenizer_fingerprint") != tokenizer.fingerprint() ||
            state.at("validation_fingerprint") != validation.fingerprint() ||
            state.at("training") != training_parameters)
            throw std::runtime_error("incompatible sharded checkpoint");
        const auto weights =
            checkpoint_dir / state.at("weights").get<std::string>();
        const auto adam = checkpoint_dir / state.at("adam").get<std::string>();
        if (state.at("weights_fingerprint") != fingerprint(weights) ||
            state.at("adam_fingerprint") != fingerprint(adam))
            throw std::runtime_error("checkpoint fingerprint mismatch");
        model.load(weights.c_str());
        optimizer.loadState(adam.string());
        if (state.at("adam_step") != optimizer.step())
            throw std::runtime_error("Adam checkpoint step mismatch");
        model.set_dropout_counters(
            state.at("dropout_counters").get<std::vector<std::uint64_t>>());
        std::istringstream random_state(
            state.at("shuffle_state").get<std::string>());
        if (!(random_state >> shuffle))
            throw std::runtime_error("invalid shuffle RNG checkpoint");
        next_shard = state.at("next_shard");
        step = state.at("step");
        optimized_tokens = state.at("optimized_tokens");
        elapsed_seconds = state.value("elapsed_seconds", 0.0);
        completed_shards =
            state.at("completed_shards").get<std::vector<std::size_t>>();
        shard_window_counts =
            state.at("shard_window_counts").get<std::vector<std::size_t>>();
        validation_history = state.at("validation_history");
        if (next_shard != completed_shards.size() ||
            next_shard != shard_window_counts.size())
            throw std::runtime_error("invalid completed shard cursor");
        for (std::size_t i = 0; i < completed_shards.size(); ++i)
            if (completed_shards[i] != i)
                throw std::runtime_error("noncontiguous completed shards");
    }

    const auto total_shards = corpus.at("shards").size();
    if (next_shard >= total_shards) {
        log << nlohmann::json{{"event", "pretraining_complete"},
                              {"shards", total_shards},
                              {"no_op", true}}
                   .dump()
            << '\n';
        return;
    }

    model.setTraining(true);
    const auto invocation_started = std::chrono::steady_clock::now();
    const auto stop = std::min(total_shards, next_shard + config.max_shards);
    for (; next_shard < stop; ++next_shard) {
        const auto &entry = corpus.at("shards").at(next_shard);
        const auto shard_path =
            shard_root / entry.at("file").get<std::string>();
        const auto training = AgentDataset::jawiki(
            shard_path, tokenizer,
            static_cast<std::size_t>(model.config().context));
        std::vector<std::size_t> order(training.windows().size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), shuffle);
        model.zero_grads();
        std::size_t pending = 0;
        for (std::size_t cursor = 0; cursor < order.size();
             cursor += config.training.batch_size) {
            const auto batch =
                make_batch(training, order, cursor, config.training.batch_size);
            const auto loss = batch_loss(model, batch, true);
            optimized_tokens += batch.valid;
            ++pending;
            ++step;
            if (pending == config.training.accumulation_steps ||
                cursor + config.training.batch_size >= order.size()) {
                for (const auto &parameter : model.namedParameters())
                    cuda_scale(parameter.lpTensor->_mGrad, 1.0F / pending);
                clip(model, config.training.clip_norm);
                optimizer.setLearningRate(scheduled_learning_rate(
                    config.training, optimizer.step() + 1));
                optimizer.update();
                model.zero_grads();
                pending = 0;
            }
            log << nlohmann::json{{"split", "train"},
                                  {"shard", next_shard},
                                  {"window_cursor",
                                   std::min(order.size(),
                                            cursor +
                                                config.training.batch_size)},
                                  {"step", step},
                                  {"loss", loss},
                                  {"valid_tokens", batch.valid}}
                       .dump()
                << '\n';
        }
        shard_window_counts.push_back(order.size());
        completed_shards.push_back(next_shard);
        const auto metrics =
            validate(model, validation, config.training.batch_size);
        const auto memory = cu_memory::statistics();
        const auto estimated_peak_device_bytes =
            non_pool_device_bytes + memory.peakUsedBytes;
        if (config.training.max_device_bytes &&
            estimated_peak_device_bytes > config.training.max_device_bytes)
            throw std::runtime_error(
                "estimated peak VRAM exceeds configured limit");
        const nlohmann::json validation_event = {
            {"split", "validation"},
            {"shard", next_shard},
            {"loss", metrics.loss},
            {"perplexity", metrics.perplexity},
            {"tokens_per_second", metrics.tokens_per_second},
            {"padding_ratio", metrics.padding_ratio},
            {"valid_tokens", metrics.valid_tokens},
            {"input_bytes", metrics.input_bytes},
            {"bits_per_byte", metrics.bits_per_byte},
            {"learning_rate", optimizer.learningRate()},
            {"adam_update", optimizer.step()},
            {"pool_peak_used_bytes", memory.peakUsedBytes},
            {"pool_peak_reserved_bytes", memory.peakReservedBytes},
            {"estimated_peak_device_bytes", estimated_peak_device_bytes}};
        validation_history.push_back(validation_event);
        log << validation_event.dump() << '\n';

        const auto completed_count = next_shard + 1;
        const bool complete = completed_count == total_shards;
        const auto stem = "shard-" + std::to_string(next_shard);
        const auto weights_name = stem + ".weights.bin";
        const auto adam_name = stem + ".adam.bin";
        const auto weights_tmp = checkpoint_dir / (weights_name + ".tmp");
        const auto adam_tmp = checkpoint_dir / (adam_name + ".tmp");
        model.save(weights_tmp.c_str());
        optimizer.saveState(adam_tmp.string());
        std::filesystem::rename(weights_tmp, checkpoint_dir / weights_name);
        std::filesystem::rename(adam_tmp, checkpoint_dir / adam_name);
        save_bundle(model, tokenizer, output);
        std::ifstream model_manifest_stream(output / "manifest.json");
        nlohmann::json model_manifest;
        model_manifest_stream >> model_manifest;
        model_manifest["pretraining_complete"] = complete;
        model_manifest["corpus_manifest_fingerprint"] =
            recorded_manifest_fingerprint;
        model_manifest["corpus_fingerprint"] =
            corpus.at("source").at("fingerprint");
        model_manifest["coverage"] = {
            {"completed_shards", completed_count},
            {"total_shards", total_shards},
            {"documents", complete ? corpus.at("source").at("documents")
                                   : nlohmann::json(nullptr)},
            {"bytes", complete ? corpus.at("source").at("bytes")
                               : nlohmann::json(nullptr)}};
        model_manifest["validation_history"] = validation_history;
        atomic_json(output / "manifest.json", model_manifest);

        std::ostringstream random_state;
        random_state << shuffle;
        const double total_elapsed =
            elapsed_seconds +
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          invocation_started)
                .count();
        const nlohmann::json state = {
            {"format", "ai_cpp_agent_training_checkpoint"},
            {"version", 2},
            {"epoch", complete ? 1 : 0},
            {"current_shard", next_shard},
            {"next_shard", completed_count},
            {"completed_shards", completed_shards},
            {"shard_window_counts", shard_window_counts},
            {"validation_history", validation_history},
            {"step", step},
            {"adam_step", optimizer.step()},
            {"optimized_tokens", optimized_tokens},
            {"weights", weights_name},
            {"adam", adam_name},
            {"weights_fingerprint", fingerprint(checkpoint_dir / weights_name)},
            {"adam_fingerprint", fingerprint(checkpoint_dir / adam_name)},
            {"tokenizer_fingerprint", tokenizer.fingerprint()},
            {"validation_fingerprint", validation.fingerprint()},
            {"corpus_manifest_fingerprint", recorded_manifest_fingerprint},
            {"corpus_fingerprint", corpus.at("source").at("fingerprint")},
            {"dropout_counters", model.dropout_counters()},
            {"shuffle_state", random_state.str()},
            {"elapsed_seconds", total_elapsed},
            {"training", training_parameters}};
        const auto state_name = stem + ".json";
        atomic_json(checkpoint_dir / state_name, state);
        atomic_json(checkpoint_dir / "latest.json",
                    {{"format", "ai_cpp_agent_training_latest"},
                     {"version", 2},
                     {"checkpoint", state_name}});
        atomic_json(output / "progress.json", state);
    }
}
} // namespace ai::model
