#include "ai/model/agent_dataset.h"
#include "ai/model/agent_tokenizer.h"
#include "ai/model/agent_training.h"
#include "ai/model/model_language_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

using namespace ai::model;
using namespace std::string_literals;
namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace

int main() {
    try {
        const auto root = std::filesystem::temp_directory_path() /
                          ("ai-cpp-agent-model-" + std::to_string(getpid()));
        std::filesystem::create_directories(root);
        {
            std::ofstream train(root / "train.jsonl");
            train
                << R"({"id":1,"text":"日本語とemoji 🤖を学習する。日本語を学習する。"})"
                << '\n';
        }
        auto tokenizer = AgentTokenizer::train(root / "train.jsonl", 512);
        require(tokenizer.vocabulary_size() >=
                        AgentTokenizer::special_count + 256 &&
                    tokenizer.vocabulary_size() <= 512,
                "SentencePiece vocabulary bounds");
        const std::string text = "未知の文字🤖\0byte"s;
        const auto encoded = tokenizer.encode(text);
        require(tokenizer.decode(encoded) == text, "byte fallback round-trip");
        tokenizer.save(root / "tokenizer.json");
        require(AgentTokenizer::load(root / "tokenizer.json").decode(encoded) ==
                    text,
                "tokenizer serialization");
        require(tokenizer.mode_id(AgentMode::Tool) == 12, "fixed mode IDs");
        {
            std::ofstream conversations(root / "conversation-train.jsonl");
            conversations
                << R"({"id":1,"utterances":[{"speaker":0,"text":"こんにちは"},{"speaker":1,"text":"こんにちは。"}]})"
                << '\n';
        }
        generate_sft_corpus(root / "conversation-train.jsonl",
                            root / "sft.jsonl", 19);
        const auto sft = AgentDataset::sft(root / "sft.jsonl", tokenizer, 512);
        std::array<bool, 9> modes{};
        for (const auto &window : sft.windows()) {
            modes[static_cast<std::size_t>(window.mode) - 4] = true;
            require(window.targets.front() == AgentTokenizer::pad_id,
                    "prompt loss mask");
            require(window.targets.back() == AgentTokenizer::eos_id,
                    "SFT EOS target");
        }
        require(std::all_of(modes.begin(), modes.end(),
                            [](bool value) { return value; }),
                "all nine modes covered");
        require(sft.order(7, 3) == sft.order(7, 3), "deterministic shuffle");
        {
            std::ofstream wiki(root / "wiki.jsonl");
            wiki << R"({"id":"a","text":"AAAAAAAAAAAAAAAAAAAAAAAA"})" << '\n'
                 << R"({"id":"b","text":"BBBBBBBBBBBBBBBBBBBBBBBB"})" << '\n';
        }
        const auto wiki =
            AgentDataset::jawiki(root / "wiki.jsonl", tokenizer, 4);
        for (const auto &window : wiki.windows())
            require(window.document_id == "\"a\"" ||
                        window.document_id == "\"b\"",
                    "window remains within one document");

        AgentTransformerConfig config;
        config.vocabulary = static_cast<int>(tokenizer.vocabulary_size());
        config.layers = 1;
        config.embedding = 16;
        config.heads = 2;
        config.feed_forward = 32;
        config.context = 16;
        config.dropout = 0.0F;
        AgentTransformer model(config);
        require(model.namedParameters().size() == 22, "parameter registration");
        auto ids = std::make_shared<cunMat>(4, 1);
        const std::vector<std::int32_t> input = {2, 4, 23, 24};
        ids->copyFromHost(input.data(), input.size());
        cunMat targets(4, 1);
        const std::vector<std::int32_t> expected = {4, 23, 24, 3};
        targets.copyFromHost(expected.data(), expected.size());
        auto loss = model.loss(ids, targets);
        require(std::isfinite(loss->_mData.toHost().at(0)), "finite loss");
        loss->backward();
        for (const auto &parameter : model.namedParameters())
            for (float gradient : parameter.lpTensor->_mGrad.toHost())
                require(std::isfinite(gradient), "finite gradient");
        const auto reference = model.forward(ids);
        require(reference->_mData.shape() ==
                    std::vector<std::int64_t>({config.vocabulary, 4, 1}),
                "logit shape");
        auto changed_ids = std::make_shared<cunMat>(4, 1);
        const std::vector<std::int32_t> changed_input = {2, 4, 23, 25};
        changed_ids->copyFromHost(changed_input.data(), changed_input.size());
        const auto changed = model.forward(changed_ids)->_mData.toHost();
        const auto unchanged = reference->_mData.toHost();
        for (int vocabulary = 0; vocabulary < config.vocabulary; ++vocabulary)
            for (int position = 0; position < 3; ++position)
                require(changed[vocabulary * 4 + position] ==
                            unchanged[vocabulary * 4 + position],
                        "causal mask");
        const auto before = model.forward(ids)->_mData.toHost();
        save_bundle(model, tokenizer, root / "bundle");
        auto loaded = load_bundle(root / "bundle");
        require(loaded.model->forward(ids)->_mData.toHost() == before,
                "bundle round-trip");

        GenerationConfig generation;
        generation.greedy = true;
        generation.max_tokens = 2;
        require(
            generate(*loaded.model, loaded.tokenizer, input, generation) ==
                generate(*loaded.model, loaded.tokenizer, input, generation),
            "greedy determinism");
        generation.greedy = false;
        generation.seed = 73;
        require(
            generate(*loaded.model, loaded.tokenizer, input, generation) ==
                generate(*loaded.model, loaded.tokenizer, input, generation),
            "sampling seed reproducibility");
        for (const auto &parameter : loaded.model->namedParameters()) {
            if (parameter.strName == "output_weight")
                cuda_fill(parameter.lpTensor->_mData, 0.0F);
            if (parameter.strName == "output_bias") {
                std::vector<float> bias(config.vocabulary, -100.0F);
                bias[AgentTokenizer::eos_id] = 100.0F;
                parameter.lpTensor->_mData.copyFromHost(bias.data(),
                                                        bias.size());
            }
        }
        generation.greedy = true;
        generation.max_tokens = 7;
        require(generate(*loaded.model, loaded.tokenizer, input, generation) ==
                    std::vector<std::int32_t>{AgentTokenizer::eos_id},
                "EOS stopping");
        bool context_failed = false;
        try {
            generate(*loaded.model, loaded.tokenizer,
                     std::vector<std::int32_t>(17, AgentTokenizer::bos_id),
                     generation);
        } catch (const std::length_error &) {
            context_failed = true;
        }
        require(context_failed, "context overflow");

        AgentTransformerConfig resume_model_config;
        resume_model_config.vocabulary =
            static_cast<int>(tokenizer.vocabulary_size());
        resume_model_config.layers = 1;
        resume_model_config.embedding = 16;
        resume_model_config.heads = 2;
        resume_model_config.feed_forward = 32;
        resume_model_config.context = 512;
        resume_model_config.dropout = 0.1F;
        resume_model_config.seed = 101;
        AgentTransformer continuous(resume_model_config);
        AgentTransformer resumed(resume_model_config);
        TrainingConfig schedule;
        schedule.steps = 4;
        schedule.seed = 29;
        std::ostringstream training_log;
        train(continuous, tokenizer, sft, sft, root / "continuous", schedule,
              false, training_log);
        schedule.steps = 2;
        train(resumed, tokenizer, sft, sft, root / "resumed", schedule, false,
              training_log);
        train(resumed, tokenizer, sft, sft, root / "resumed", schedule, true,
              training_log);
        const auto continuous_parameters = continuous.namedParameters();
        const auto resumed_parameters = resumed.namedParameters();
        require(continuous_parameters.size() == resumed_parameters.size(),
                "resume parameter count");
        for (std::size_t i = 0; i < continuous_parameters.size(); ++i)
            require(continuous_parameters[i].lpTensor->_mData.toHost() ==
                        resumed_parameters[i].lpTensor->_mData.toHost(),
                    "continuous and resumed weights");
        require(continuous.dropout_counters() == resumed.dropout_counters(),
                "continuous and resumed dropout counters");
        std::ifstream latest_stream(root / "resumed/checkpoint/latest.json");
        nlohmann::json latest;
        latest_stream >> latest;
        std::ifstream state_stream(root / "resumed/checkpoint" /
                                   latest.at("checkpoint").get<std::string>());
        nlohmann::json state;
        state_stream >> state;
        require(state.at("step") == 4 && state.at("adam_step") == 4,
                "resumed step and Adam step");

        AgentTransformerConfig production;
        require(production.parameter_count() > 7000000 &&
                    production.parameter_count() < 9000000,
                "approximately 8M parameters");
        std::filesystem::remove_all(root);
        std::cout
            << "agent tokenizer/transformer/bundle/generation checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
