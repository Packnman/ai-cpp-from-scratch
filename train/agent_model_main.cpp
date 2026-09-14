#include "ai/model/agent_training.h"

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
TrainingConfig training_config(const Options &options) {
    TrainingConfig value;
    value.steps = std::stoull(optional(options, "--steps", "1"));
    value.batch_size = std::stoull(optional(options, "--batch-size", "1"));
    value.accumulation_steps =
        std::stoull(optional(options, "--accumulation", "1"));
    value.token_budget = std::stoull(optional(options, "--token-budget", "0"));
    value.learning_rate =
        std::stof(optional(options, "--learning-rate", "0.0003"));
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
                "command: tokenizer|generate-sft|pretrain|sft|resume|validate");
        const std::string command = argv[1];
        const auto options = parse(argc, argv);
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
        if (command == "generate-sft") {
            generate_sft_corpus(required(options, "--conversation-train"),
                                required(options, "--output"),
                                std::stoull(optional(options, "--seed", "42")));
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
                                        {"valid_tokens", metric.valid_tokens}}
                             .dump()
                      << '\n';
            return 0;
        }
        throw std::invalid_argument("unknown command: " + command);
    } catch (const std::exception &error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
