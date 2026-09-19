#include "ai/ner/extractor.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace ai::ner;
namespace {
int checks = 0;
#define CHECK(condition)                                                       \
    do {                                                                       \
        ++checks;                                                              \
        if (!(condition))                                                      \
            throw std::runtime_error("check failed: " #condition);             \
    } while (false)

const EntityMention *find(const std::vector<EntityMention> &mentions,
                          EntityType type, std::string_view surface) {
    for (const auto &mention : mentions)
        if (mention.type == type && mention.surface == surface)
            return &mention;
    return nullptr;
}

void rule_checks() {
    RuleEntityExtractor extractor;
    const std::string text = "予算は5万円、予備も5万円。２０２６年９月１８日の"
                             "１４時３０分に3名で、2週間試す。";
    const auto mentions = extractor.extract(text, "turn-7");
    CHECK(find(mentions, EntityType::Money, "5万円"));
    CHECK(std::count_if(mentions.begin(), mentions.end(), [](const auto &m) {
              return m.type == EntityType::Money && m.surface == "5万円";
          }) == 2);
    const auto *date = find(mentions, EntityType::Date, "２０２６年９月１８日");
    CHECK(date && date->normalized == "2026-09-18");
    const auto *time = find(mentions, EntityType::Time, "１４時３０分");
    CHECK(time && time->normalized == "14:30");
    CHECK(find(mentions, EntityType::Quantity, "3名"));
    CHECK(find(mentions, EntityType::Duration, "2週間"));
    for (const auto &mention : mentions) {
        CHECK(valid_mention(text, mention));
        CHECK(mention.utterance_id == "turn-7");
    }
    const auto relative = extractor.extract("明日までに終える");
    CHECK(find(relative, EntityType::Date, "明日") == nullptr);
    const auto condition = extractor.extract("ただし、雨の場合は実行しない。");
    CHECK(std::any_of(condition.begin(), condition.end(), [](const auto &m) {
        return m.type == EntityType::Condition;
    }));
    const auto nested = extractor.extract("ただし予算は5万円とする。");
    CHECK(find(nested, EntityType::Money, "5万円"));
    CHECK(std::any_of(nested.begin(), nested.end(), [](const auto &m) {
        return m.type == EntityType::Condition;
    }));
}

void utf8_checks() {
    const auto cps = decode_utf8("A日𠮷");
    CHECK(cps.size() == 3);
    CHECK(cps[1].byte_start == 1 && cps[1].byte_end == 4);
    CHECK(cps[2].byte_start == 4 && cps[2].byte_end == 8);
    bool threw = false;
    try {
        decode_utf8(std::string("\xff", 1));
    } catch (...) {
        threw = true;
    }
    CHECK(threw);
}

void model_bundle_check(const std::filesystem::path &path) {
    std::vector<std::string> labels{"O"};
    for (const auto &type : {"人名", "法人名", "政治的組織名", "その他の組織名",
                             "地名", "施設名", "製品名", "イベント名"}) {
        labels.push_back(std::string("B-") + type);
        labels.push_back(std::string("I-") + type);
    }
    nlohmann::json json{{"format", "ai_cpp_ner_bundle_v1"},
                        {"labels", labels},
                        {"vocab", {"<PAD>", "<UNK>"}},
                        {"config",
                         {{"encoder", "bidirectional_window"},
                          {"embedding_dim", 1},
                          {"hidden_dim", 1},
                          {"window", 1}}},
                        {"weights",
                         {{"embedding", {0.0, 0.0}},
                          {"hidden_weight", {0.0, 0.0}},
                          {"hidden_bias", {0.0}},
                          {"output_weight", std::vector<float>(17)},
                          {"output_bias", std::vector<float>(17)}}}};
    std::ofstream(path) << json.dump();
    ModelEntityExtractor extractor(path.string());
    CHECK(extractor.extract("未知文字").empty());
    CHECK(extractor.parameter_bytes() == (2 + 2 + 1 + 17 + 17) * sizeof(float));
}
} // namespace

int main() {
    try {
        rule_checks();
        utf8_checks();
        const auto path =
            std::filesystem::temp_directory_path() / "ai_cpp_ner_test.json";
        model_bundle_check(path);
        std::filesystem::remove(path);
        std::cout << "ner_check: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
