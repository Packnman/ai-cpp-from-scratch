#include "ai/agent/context_builder.h"
#include "ai/model/agent_dataset.h"
#include "ai/model/agent_tokenizer.h"
#include "ai/model/agent_training.h"
#include "ai/model/jawiki_sharding.h"
#include "ai/model/model_language_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
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
        BalancedTokenizerConfig balanced_config;
        balanced_config.vocabulary_size = 512;
        balanced_config.jawiki_bytes = 32;
        balanced_config.conversation_bytes = 16;
        balanced_config.seed = 19;
        auto balanced = AgentTokenizer::train_balanced(
            root / "train.jsonl", root / "conversation-train.jsonl",
            balanced_config);
        balanced.save(root / "balanced.model");
        require(std::filesystem::is_regular_file(
                    root / "balanced.model.metadata.json"),
                "balanced tokenizer metadata sidecar");
        auto balanced_loaded = AgentTokenizer::load(root / "balanced.model");
        const auto metadata =
            nlohmann::json::parse(balanced_loaded.training_metadata());
        require(metadata.at("seed") == 19 &&
                    metadata.at("sources").at("jawiki").at(
                        "scanned_documents") == 1 &&
                    metadata.at("sources").at("jawiki").at(
                        "selected_bytes") <= 32,
                "balanced tokenizer provenance");
        const auto efficiency = evaluate_tokenizer(
            balanced_loaded, root / "train.jsonl", 1);
        require(efficiency.documents == 1 && efficiency.characters > 0 &&
                    efficiency.tokens > 0 && efficiency.over_context == 1,
                "tokenizer efficiency metrics");
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
        const std::array<bool, 9> required_modes = {
            true, true, true, true, true, true, false, true, false};
        require(modes == required_modes,
                "agent profile covers runtime modes only");
        require(sft.sft_profile() == "agent", "agent SFT profile");

        generate_sft_corpus(root / "conversation-train.jsonl",
                            root / "validation-sft.jsonl", 19, "agent",
                            "validation");
        generate_sft_corpus(root / "conversation-train.jsonl",
                            root / "repeat-sft.jsonl", 19, "agent", "train");
        const auto slurp = [](const std::filesystem::path &path) {
            std::ifstream in(path);
            return std::string(std::istreambuf_iterator<char>(in), {});
        };
        require(slurp(root / "sft.jsonl") == slurp(root / "repeat-sft.jsonl"),
                "SFT generator reproducibility");
        std::set<std::string> train_ids, validation_ids, train_prompts,
            validation_prompts;
        std::map<std::string, std::size_t> train_counts, validation_counts;
        const auto inspect =
            [&](const std::filesystem::path &path, std::string_view split,
                std::set<std::string> &ids, std::set<std::string> &prompts,
                std::map<std::string, std::size_t> &counts) {
                std::ifstream in(path);
                std::string row;
                while (std::getline(in, row)) {
                    const auto value = nlohmann::json::parse(row);
                    require(value.at("split") == split, "SFT split metadata");
                    ids.insert(value.at("id"));
                    if (value.at("mode") != "CHAT")
                        prompts.insert(value.at("prompt"));
                    ++counts[value.at("mode").get<std::string>()];
                    if (value.at("mode") != "CHAT" &&
                        value.at("mode") != "SUMMARIZE" &&
                        value.at("mode") != "FINAL")
                        require(nlohmann::json::parse(
                                    value.at("output").get<std::string>())
                                    .is_object(),
                                "structured output JSON");
                }
            };
        inspect(root / "sft.jsonl", "train", train_ids, train_prompts,
                train_counts);
        inspect(root / "validation-sft.jsonl", "validation", validation_ids,
                validation_prompts, validation_counts);
        for (const auto *mode : {"PARSE", "PLAN", "EVALUATE", "SUMMARIZE",
                                 "MEMORY_WRITE", "FINAL"}) {
            require(train_counts[mode] == 2000, "train mode count");
            require(validation_counts[mode] == 200, "validation mode count");
        }
        for (const auto &id : validation_ids)
            require(!train_ids.contains(id), "train/validation ID separation");
        for (const auto &prompt : validation_prompts)
            require(!train_prompts.contains(prompt),
                    "train/validation prompt separation");
        generate_sft_corpus(root / "conversation-train.jsonl",
                            root / "chat-sft.jsonl", 19, "chat", "train");
        const auto chat_sft =
            AgentDataset::sft(root / "chat-sft.jsonl", tokenizer, 512);
        require(chat_sft.trained_modes() == std::vector<std::string>{"CHAT"},
                "chat profile only trains CHAT");
        require(sft.order(7, 3) == sft.order(7, 3), "deterministic shuffle");

        std::array<std::size_t, 9> sampled_modes{};
        for (const auto index : sft.order(7, 3))
            ++sampled_modes[static_cast<std::size_t>(
                                sft.windows()[index].mode) -
                            4];
        require(sampled_modes == std::array<std::size_t, 9>{20, 20, 20, 10, 10,
                                                            10, 0, 10, 0},
                "agent sampling weights");
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

        {
            std::ifstream manifest_stream(root / "continuous/manifest.json");
            nlohmann::json manifest;
            manifest_stream >> manifest;
            require(manifest.at("sft_profile") == "agent",
                    "manifest SFT profile");
            require(manifest.at("trained_modes").size() == 7,
                    "manifest trained modes");
            require(manifest.at("sft_generation").at("generator") ==
                        "local-template-v2",
                    "manifest generator");
            require(
                manifest.at("split_fingerprints").contains("train") &&
                    manifest.at("split_fingerprints").contains("validation"),
                "manifest split fingerprints");
        }
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

        {
            const auto source = root / "sharded-source.jsonl";
            {
                std::ofstream out(source, std::ios::binary);
                out << R"({"id":"s0","text":"日本語alpha alpha alpha"})" << '\n'
                    << R"({"id":"s1","text":"日本語beta beta beta beta"})"
                    << '\n'
                    << R"({"id":"s2","text":"日本語gamma gamma gamma"})"
                    << '\n';
            }
            shard_jawiki(source, root / "shards", 64);
            shard_jawiki(source, root / "shards", 64);
            std::ifstream shard_manifest_stream(root / "shards/manifest.json");
            nlohmann::json shard_manifest;
            shard_manifest_stream >> shard_manifest;
            require(shard_manifest.at("source").at("documents") == 3,
                    "shard document coverage");
            std::string joined;
            for (const auto &entry : shard_manifest.at("shards"))
                joined += slurp(root / "shards" /
                                entry.at("file").get<std::string>());
            require(joined == slurp(source), "shards concatenate exactly");

            {
                std::ofstream out(root / "sharded-validation.jsonl");
                out << R"({"id":"v0","text":"検証用の日本語テキスト"})" << '\n';
            }
            const auto shard_validation = AgentDataset::jawiki(
                root / "sharded-validation.jsonl", tokenizer, 16);
            AgentTransformerConfig shard_model_config = resume_model_config;
            shard_model_config.context = 16;
            shard_model_config.dropout = 0.1F;
            shard_model_config.seed = 303;
            ShardedTrainingConfig shard_schedule;
            shard_schedule.training.steps = 1;
            shard_schedule.training.batch_size = 2;
            shard_schedule.training.accumulation_steps = 2;
            shard_schedule.training.seed = 71;
            shard_schedule.training.learning_rate = 3.0e-4F;
            shard_schedule.training.minimum_learning_rate = 3.0e-5F;
            shard_schedule.training.warmup_updates = 2;
            shard_schedule.training.decay_updates = 20;
            shard_schedule.max_shards = shard_manifest.at("shards").size();

            AgentTransformer all_at_once(shard_model_config);
            std::ostringstream all_log;
            pretrain_sharded(all_at_once, tokenizer, shard_validation,
                             root / "shards/manifest.json",
                             root / "sharded-continuous", shard_schedule, false,
                             all_log);

            AgentTransformer interrupted(shard_model_config);
            auto one_shard = shard_schedule;
            one_shard.max_shards = 1;
            std::ostringstream resumed_log;
            pretrain_sharded(interrupted, tokenizer, shard_validation,
                             root / "shards/manifest.json",
                             root / "sharded-resumed", one_shard, false,
                             resumed_log);
            {
                std::ofstream out(root / "wrong-validation.jsonl");
                out << R"({"id":"wrong","text":"異なる検証データ"})" << '\n';
            }
            const auto wrong_validation = AgentDataset::jawiki(
                root / "wrong-validation.jsonl", tokenizer, 16);
            bool validation_rejected = false;
            try {
                pretrain_sharded(interrupted, tokenizer, wrong_validation,
                                 root / "shards/manifest.json",
                                 root / "sharded-resumed", one_shard, true,
                                 resumed_log);
            } catch (const std::runtime_error &) {
                validation_rejected = true;
            }
            require(validation_rejected,
                    "validation fingerprint mismatch rejected");
            AgentTransformer restored(shard_model_config);
            shard_schedule.max_shards = shard_manifest.at("shards").size();
            pretrain_sharded(restored, tokenizer, shard_validation,
                             root / "shards/manifest.json",
                             root / "sharded-resumed", shard_schedule, true,
                             resumed_log);
            const auto all_parameters = all_at_once.namedParameters();
            const auto restored_parameters = restored.namedParameters();
            for (std::size_t i = 0; i < all_parameters.size(); ++i)
                require(all_parameters[i].lpTensor->_mData.toHost() ==
                            restored_parameters[i].lpTensor->_mData.toHost(),
                        "sharded resume weights");
            require(all_at_once.dropout_counters() ==
                        restored.dropout_counters(),
                    "sharded resume RNG counters");

            std::ifstream progress_stream(root /
                                          "sharded-resumed/progress.json");
            nlohmann::json progress;
            progress_stream >> progress;
            std::size_t expected_steps = 0;
            for (const auto windows : progress.at("shard_window_counts"))
                expected_steps += (windows.get<std::size_t>() +
                                   shard_schedule.training.batch_size - 1) /
                                  shard_schedule.training.batch_size;
            require(progress.at("step") == expected_steps,
                    "every shard window consumed once");
            const auto &last_validation = progress.at("validation_history").back();
            require(last_validation.at("bits_per_byte").get<double>() > 0.0 &&
                        last_validation.at("input_bytes").get<std::size_t>() > 0 &&
                        last_validation.at("adam_update").get<std::size_t>() > 0,
                    "normalized validation and schedule metrics");
            require(progress.at("next_shard") ==
                            shard_manifest.at("shards").size() &&
                        progress.at("epoch") == 1,
                    "all shard cursor complete");
            std::ifstream completed_manifest_stream(
                root / "sharded-resumed/manifest.json");
            nlohmann::json completed_manifest;
            completed_manifest_stream >> completed_manifest;
            require(completed_manifest.at("pretraining_complete") == true &&
                        completed_manifest.at("coverage").at("documents") == 3,
                    "full pretraining coverage manifest");
            std::ostringstream no_op_log;
            pretrain_sharded(restored, tokenizer, shard_validation,
                             root / "shards/manifest.json",
                             root / "sharded-resumed", one_shard, true,
                             no_op_log);
            require(no_op_log.str().find("\"no_op\":true") != std::string::npos,
                    "completed pretraining is a no-op");

            const auto first_shard =
                root / "shards" /
                shard_manifest.at("shards").at(0).at("file").get<std::string>();
            const auto original_shard = slurp(first_shard);
            {
                auto modified_shard = original_shard;
                modified_shard.at(0) = '[';
                std::ofstream out(first_shard,
                                  std::ios::binary | std::ios::trunc);
                out.write(modified_shard.data(), modified_shard.size());
            }
            bool shard_change_rejected = false;
            try {
                pretrain_sharded(restored, tokenizer, shard_validation,
                                 root / "shards/manifest.json",
                                 root / "sharded-resumed", one_shard, true,
                                 no_op_log);
            } catch (const std::runtime_error &) {
                shard_change_rejected = true;
            }
            require(shard_change_rejected, "modified shard rejected");
            {
                std::ofstream out(first_shard,
                                  std::ios::binary | std::ios::trunc);
                out.write(original_shard.data(), original_shard.size());
            }
            const auto missing_shard = first_shard.string() + ".missing";
            std::filesystem::rename(first_shard, missing_shard);
            bool missing_shard_rejected = false;
            try {
                pretrain_sharded(restored, tokenizer, shard_validation,
                                 root / "shards/manifest.json",
                                 root / "sharded-resumed", one_shard, true,
                                 no_op_log);
            } catch (const std::runtime_error &) {
                missing_shard_rejected = true;
            }
            require(missing_shard_rejected, "missing shard rejected");
            std::filesystem::rename(missing_shard, first_shard);

            std::ifstream latest_shard_stream(
                root / "sharded-resumed/checkpoint/latest.json");
            nlohmann::json latest_shard;
            latest_shard_stream >> latest_shard;
            const auto shard_state_path =
                root / "sharded-resumed/checkpoint" /
                latest_shard.at("checkpoint").get<std::string>();
            const auto original_state = slurp(shard_state_path);
            auto incompatible_state = nlohmann::json::parse(original_state);
            incompatible_state["tokenizer_fingerprint"] = "different";
            {
                std::ofstream out(shard_state_path, std::ios::trunc);
                out << incompatible_state.dump(2) << '\n';
            }
            bool tokenizer_rejected = false;
            try {
                pretrain_sharded(restored, tokenizer, shard_validation,
                                 root / "shards/manifest.json",
                                 root / "sharded-resumed", one_shard, true,
                                 no_op_log);
            } catch (const std::runtime_error &) {
                tokenizer_rejected = true;
            }
            require(tokenizer_rejected, "tokenizer mismatch rejected");
            {
                std::ofstream out(shard_state_path,
                                  std::ios::binary | std::ios::trunc);
                out.write(original_state.data(), original_state.size());
            }

            const auto original_corpus_manifest =
                slurp(root / "shards/manifest.json");
            auto changed_corpus =
                nlohmann::json::parse(original_corpus_manifest);
            changed_corpus["source"]["fingerprint"] = "different";
            {
                std::ofstream out(root / "shards/manifest.json",
                                  std::ios::trunc);
                out << changed_corpus.dump(2) << '\n';
            }
            bool corpus_rejected = false;
            try {
                pretrain_sharded(restored, tokenizer, shard_validation,
                                 root / "shards/manifest.json",
                                 root / "sharded-resumed", one_shard, true,
                                 no_op_log);
            } catch (const std::runtime_error &) {
                corpus_rejected = true;
            }
            require(corpus_rejected, "corpus mismatch rejected");
            {
                std::ofstream out(root / "shards/manifest.json",
                                  std::ios::binary | std::ios::trunc);
                out.write(original_corpus_manifest.data(),
                          original_corpus_manifest.size());
            }

            std::filesystem::create_directories(root / "occupied-output");
            {
                std::ofstream marker(root / "occupied-output/keep.txt");
                marker << "owned";
            }
            bool occupied_rejected = false;
            AgentTransformer fresh(shard_model_config);
            try {
                pretrain_sharded(fresh, tokenizer, shard_validation,
                                 root / "shards/manifest.json",
                                 root / "occupied-output", one_shard, false,
                                 no_op_log);
            } catch (const std::runtime_error &) {
                occupied_rejected = true;
            }
            require(occupied_rejected, "occupied output rejected");
        }

        AgentTransformerConfig production;
        require(production.parameter_count() > 7000000 &&
                    production.parameter_count() < 9000000,
                "approximately 8M parameters");
        AgentTransformerConfig next = production;
        next.layers = 6;
        next.embedding = 320;
        next.heads = 5;
        next.feed_forward = 1280;
        require(next.parameter_count() == 12977152,
                "next model parameter count");
        std::filesystem::remove_all(root);
        std::cout
            << "agent tokenizer/transformer/bundle/generation checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
