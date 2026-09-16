#include "ai/model/agent_training.h"
#include "ai/model/jawiki_sharding.h"

#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace ai::model;
namespace {
using Options = std::map<std::string, std::string, std::less<>>;
Options parse(int argc, char **argv) {
    Options result;
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 == argc || !std::string_view(argv[i]).starts_with("--"))
            throw std::invalid_argument("expected named --option VALUE pairs");
        result[argv[i]] = argv[i + 1];
    }
    return result;
}
std::string required(const Options &options, const std::string &name) {
    const auto found = options.find(name);
    if (found == options.end())
        throw std::invalid_argument("missing " + name);
    return found->second;
}
std::string optional(const Options &options, const std::string &name,
                     std::string fallback) {
    const auto found = options.find(name);
    return found == options.end() ? std::move(fallback) : found->second;
}
ai::agent::ModelMode validation_mode(std::string_view name) {
    using M = ai::agent::ModelMode;
    if (name == "PARSE")
        return M::Parse;
    if (name == "PLAN")
        return M::Plan;
    if (name == "EVALUATE")
        return M::Evaluate;
    if (name == "MEMORY_WRITE")
        return M::MemoryWrite;
    throw std::invalid_argument("mode is not structured validation target");
}
struct HeldOutStat {
        std::size_t total = 0, syntax = 0, semantic = 0;
};
bool semantic_match(std::string_view mode, const nlohmann::json &expected,
                    const nlohmann::json &actual) {
    if (mode == "PARSE")
        return actual.value("intent", "") == expected.value("intent", "") &&
               actual.value("arguments", nlohmann::json::object()) ==
                   expected.value("arguments", nlohmann::json::object());
    if (mode == "PLAN")
        return actual.contains("tasks") && actual["tasks"].is_array() &&
               !actual["tasks"].empty() && expected.contains("tasks") &&
               actual["tasks"][0].value("operation", "") ==
                   expected["tasks"][0].value("operation", "") &&
               actual["tasks"][0].value("arguments",
                                        nlohmann::json::object()) ==
                   expected["tasks"][0].value("arguments",
                                              nlohmann::json::object());
    if (mode == "EVALUATE")
        return actual.value("status", "") == expected.value("status", "");
    return actual.value("memories", nlohmann::json::array()) ==
           expected.value("memories", nlohmann::json::array());
}
bool autoregressive_validation(AgentBundle &bundle,
                               const std::filesystem::path &path,
                               std::size_t requested, std::ostream &out) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot read held-out SFT split");
    std::map<std::string, HeldOutStat, std::less<>> stats;
    std::string line;
    std::size_t checked = 0;
    while (checked < requested && std::getline(input, line)) {
        const auto example = nlohmann::json::parse(line);
        const auto mode = example.at("mode").get<std::string>();
        if (mode != "PARSE" && mode != "PLAN" && mode != "EVALUATE" &&
            mode != "MEMORY_WRITE")
            continue;
        auto &stat = stats[mode];
        const auto quota = requested / 4;
        if (stat.total >= quota)
            continue;
        ++stat.total;
        ++checked;
        const auto runtime_mode = validation_mode(mode);
        std::vector<std::int32_t> ids = {
            AgentTokenizer::bos_id,
            bundle.tokenizer.mode_id(
                static_cast<AgentMode>(static_cast<int>(runtime_mode) + 4))};
        const auto prompt =
            bundle.tokenizer.encode(example.at("prompt").get<std::string>());
        GenerationConfig config;
        config.greedy = true;
        config.max_tokens = 384;
        const auto room =
            bundle.model->config().context - static_cast<int>(ids.size());
        if (room <= 0)
            continue;
        config.max_tokens = std::min(config.max_tokens, room);
        try {
            const auto answer = bundle.tokenizer.decode(
                generate(*bundle.model, bundle.tokenizer, ids, config));
            const auto actual = nlohmann::json::parse(answer);
            if (!actual.is_object())
                continue;
            const auto expected =
                nlohmann::json::parse(example.at("output").get<std::string>());
            bool schema = false;
            if (mode == "PARSE")
                schema = actual.contains("raw") && actual.contains("intent") &&
                         actual.contains("goal") &&
                         actual.contains("constraints");
            if (mode == "PLAN")
                schema = actual.contains("goal") && actual.contains("tasks") &&
                         actual["tasks"].is_array();
            if (mode == "EVALUATE")
                schema = actual.contains("status") && actual.contains("reason");
            if (mode == "MEMORY_WRITE")
                schema = actual.contains("memories") &&
                         actual["memories"].is_array();
            if (!schema)
                continue;
            ++stat.syntax;
            if (semantic_match(mode, expected, actual))
                ++stat.semantic;
        } catch (const std::exception &) {
        }
    }
    std::size_t total = 0, syntax = 0, semantic = 0;
    for (const auto &[mode, stat] : stats) {
        total += stat.total;
        syntax += stat.syntax;
        semantic += stat.semantic;
        out << nlohmann::json{{"validation", "autoregressive"},
                              {"mode", mode},
                              {"examples", stat.total},
                              {"json_syntax_rate",
                               stat.total ? double(stat.syntax) / stat.total
                                          : 0.0},
                              {"semantic_rate",
                               stat.total ? double(stat.semantic) / stat.total
                                          : 0.0}}
                   .dump()
            << '\n';
    }
    const bool passed =
        total >= 500 && syntax == total && semantic * 100 >= total * 95;
    out << nlohmann::json{{"validation", "autoregressive"},
                          {"examples", total},
                          {"json_syntax_rate",
                           total ? double(syntax) / total : 0.0},
                          {"semantic_rate",
                           total ? double(semantic) / total : 0.0},
                          {"promotion_ready", passed}}
               .dump()
        << '\n';
    return passed;
}
TrainingConfig training_config(const Options &options) {
    TrainingConfig value;
    value.steps = std::stoull(optional(options, "--steps", "1"));
    value.batch_size = std::stoull(optional(options, "--batch-size", "1"));
    value.accumulation_steps =
        std::stoull(optional(options, "--accumulation", "1"));
    value.token_budget = std::stoull(optional(options, "--token-budget", "0"));
    value.learning_rate =
        std::stof(optional(options, "--learning-rate", "0.0003"));
    value.minimum_learning_rate =
        std::stof(optional(options, "--minimum-learning-rate",
                            optional(options, "--learning-rate", "0.0003")));
    value.warmup_updates =
        std::stoull(optional(options, "--warmup-updates", "0"));
    value.decay_updates =
        std::stoull(optional(options, "--decay-updates", "0"));
    value.max_device_bytes =
        std::stoull(optional(options, "--max-device-bytes", "0"));
    value.clip_norm = std::stof(optional(options, "--clip-norm", "1.0"));
    value.seed = std::stoull(optional(options, "--seed", "42"));
    return value;
}
AgentTransformerConfig model_config(const Options &options, int vocabulary) {
    AgentTransformerConfig value;
    value.vocabulary = vocabulary;
    value.layers = std::stoi(optional(options, "--layers", "4"));
    value.embedding = std::stoi(optional(options, "--embedding", "256"));
    value.heads = std::stoi(optional(options, "--heads", "4"));
    value.feed_forward = std::stoi(optional(options, "--feed-forward", "1024"));
    value.context = std::stoi(optional(options, "--context", "1024"));
    value.dropout = std::stof(optional(options, "--dropout", "0.1"));
    value.seed = std::stoull(optional(options, "--seed", "42"));
    return value;
}
AgentDataset dataset(std::string_view kind, const std::string &path,
                     const AgentTokenizer &tokenizer, std::size_t context,
                     std::size_t token_limit = 0) {
    if (kind == "jawiki")
        return AgentDataset::jawiki(path, tokenizer, context, token_limit);
    if (kind == "sft")
        return AgentDataset::sft(path, tokenizer, context);
    throw std::invalid_argument("--kind must be jawiki or sft");
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 2)
            throw std::invalid_argument(
                "command: "
                "tokenizer|tokenizer-balanced|tokenizer-evaluate|generate-sft|"
                "shard-jawiki|pretrain|pretrain-sharded|"
                "sft|resume|validate");
        const std::string command = argv[1];
        const auto options = parse(argc, argv);
        if (command == "shard-jawiki") {
            shard_jawiki(
                required(options, "--source"), required(options, "--output"),
                std::stoull(optional(options, "--shard-bytes", "134217728")));
            return 0;
        }
        if (command == "tokenizer") {
            const auto output =
                std::filesystem::path(required(options, "--output"));
            std::filesystem::create_directories(output.parent_path());
            AgentTokenizer::train(
                required(options, "--train"),
                std::stoull(optional(options, "--vocabulary", "8192")))
                .save(output);
            return 0;
        }
        if (command == "tokenizer-balanced") {
            const auto output =
                std::filesystem::path(required(options, "--output"));
            std::filesystem::create_directories(output.parent_path());
            BalancedTokenizerConfig config;
            config.vocabulary_size =
                std::stoull(optional(options, "--vocabulary", "8192"));
            config.jawiki_bytes =
                std::stoull(optional(options, "--jawiki-bytes", "234881024"));
            config.conversation_bytes = std::stoull(
                optional(options, "--conversation-bytes", "33554432"));
            config.seed = std::stoull(optional(options, "--seed", "42"));
            AgentTokenizer::train_balanced(
                required(options, "--jawiki-train"),
                required(options, "--conversation-train"), config).save(output);
            return 0;
        }
        if (command == "tokenizer-evaluate") {
            const auto tokenizer =
                AgentTokenizer::load(required(options, "--tokenizer"));
            const auto metrics = evaluate_tokenizer(
                tokenizer, required(options, "--data"),
                std::stoull(optional(options, "--context", "1024")));
            std::cout << nlohmann::json{
                {"documents", metrics.documents},
                {"input_bytes", metrics.input_bytes},
                {"characters", metrics.characters},
                {"tokens", metrics.tokens},
                {"tokens_per_character", metrics.characters
                    ? static_cast<double>(metrics.tokens) / metrics.characters : 0.0},
                {"byte_fallback_tokens", metrics.byte_fallback_tokens},
                {"byte_fallback_rate", metrics.tokens
                    ? static_cast<double>(metrics.byte_fallback_tokens) / metrics.tokens : 0.0},
                {"over_context", metrics.over_context},
                {"over_context_rate", metrics.documents
                    ? static_cast<double>(metrics.over_context) / metrics.documents : 0.0}}.dump()
                      << '\n';
            return 0;
        }
        if (command == "generate-sft") {
            generate_sft_corpus(required(options, "--conversation-train"),
                                required(options, "--output"),
                                std::stoull(optional(options, "--seed", "42")),
                                optional(options, "--profile", "agent"),
                                optional(options, "--split", "train"));
            return 0;
        }
        if (command == "pretrain") {
            const auto tokenizer =
                AgentTokenizer::load(required(options, "--tokenizer"));
            const auto config = model_config(
                options, static_cast<int>(tokenizer.vocabulary_size()));
            AgentTransformer model(config);
            const auto common_limit =
                optional(options, "--dataset-token-limit", "0");
            const auto train_limit = std::stoull(
                optional(options, "--train-token-limit", common_limit));
            const auto validation_limit = std::stoull(
                optional(options, "--validation-token-limit", common_limit));
            const auto training =
                dataset("jawiki", required(options, "--train"), tokenizer,
                        config.context, train_limit);
            const auto validation =
                dataset("jawiki", required(options, "--validation"), tokenizer,
                        config.context, validation_limit);
            train(model, tokenizer, training, validation,
                  required(options, "--output"), training_config(options),
                  false, std::cout);
            return 0;
        }
        if (command == "pretrain-sharded") {
            const auto output =
                std::filesystem::path(required(options, "--output"));
            const bool resume = std::filesystem::is_regular_file(
                output / "checkpoint/latest.json");
            auto tokenizer =
                AgentTokenizer::load(required(options, "--tokenizer"));
            std::unique_ptr<AgentTransformer> model;
            if (resume) {
                auto bundle = load_bundle(output);
                if (bundle.tokenizer.fingerprint() != tokenizer.fingerprint())
                    throw std::runtime_error(
                        "requested tokenizer does not match checkpoint");
                model = std::move(bundle.model);
            } else {
                const auto config = model_config(
                    options, static_cast<int>(tokenizer.vocabulary_size()));
                model = std::make_unique<AgentTransformer>(config);
            }
            const auto validation = AgentDataset::jawiki(
                required(options, "--validation"), tokenizer,
                static_cast<std::size_t>(model->config().context),
                std::stoull(
                    optional(options, "--validation-token-limit", "260000")));
            ShardedTrainingConfig sharded;
            sharded.training = training_config(options);
            sharded.epochs = std::stoull(optional(options, "--epochs", "1"));
            sharded.max_shards =
                std::stoull(optional(options, "--max-shards", "1"));
            pretrain_sharded(*model, tokenizer, validation,
                             required(options, "--shard-manifest"), output,
                             sharded, resume, std::cout);
            return 0;
        }
        if (command == "sft") {
            auto bundle = load_bundle(required(options, "--source"));
            const auto training = AgentDataset::sft(
                required(options, "--train"), bundle.tokenizer,
                bundle.model->config().context);
            const auto validation = AgentDataset::sft(
                required(options, "--validation"), bundle.tokenizer,
                bundle.model->config().context);
            train(*bundle.model, bundle.tokenizer, training, validation,
                  required(options, "--output"), training_config(options),
                  false, std::cout);
            return 0;
        }
        if (command == "resume") {
            const auto directory =
                std::filesystem::path(required(options, "--model"));
            auto bundle = load_bundle(directory);
            const auto kind = optional(options, "--kind", "jawiki");
            const auto common_limit =
                optional(options, "--dataset-token-limit", "0");
            const auto train_limit = std::stoull(
                optional(options, "--train-token-limit", common_limit));
            const auto validation_limit = std::stoull(
                optional(options, "--validation-token-limit", common_limit));
            const auto training =
                dataset(kind, required(options, "--train"), bundle.tokenizer,
                        bundle.model->config().context, train_limit);
            const auto validation = dataset(
                kind, required(options, "--validation"), bundle.tokenizer,
                bundle.model->config().context, validation_limit);
            train(*bundle.model, bundle.tokenizer, training, validation,
                  directory, training_config(options), true, std::cout);
            return 0;
        }
        if (command == "validate") {
            auto bundle = load_bundle(required(options, "--model"));
            const auto data =
                dataset(optional(options, "--kind", "jawiki"),
                        required(options, "--data"), bundle.tokenizer,
                        bundle.model->config().context);
            const auto metric =
                validate(*bundle.model, data,
                         std::stoull(optional(options, "--batch-size", "1")));
            std::cout << nlohmann::json{{"loss", metric.loss},
                                        {"perplexity", metric.perplexity},
                                        {"tokens_per_second",
                                         metric.tokens_per_second},
                                        {"padding_ratio", metric.padding_ratio},
                                        {"valid_tokens", metric.valid_tokens},
                                        {"input_bytes", metric.input_bytes},
                                        {"bits_per_byte", metric.bits_per_byte}}
                             .dump()
                      << '\n';
            if (optional(options, "--kind", "jawiki") == "sft" &&
                optional(options, "--autoregressive", "true") != "false")
                return autoregressive_validation(
                           bundle, required(options, "--data"),
                           std::stoull(optional(options, "--held-out", "500")),
                           std::cout)
                           ? 0
                           : 2;
            return 0;
        }
        throw std::invalid_argument("unknown command: " + command);
    } catch (const std::exception &error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
