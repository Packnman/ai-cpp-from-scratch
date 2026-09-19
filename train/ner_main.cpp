#include "ai/ner/entity.h"
#include "ai/ner/extractor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <stdexcept>
#include <sys/resource.h>
#include <unordered_map>

using namespace ai::ner;
namespace {
constexpr std::size_t class_count = 17;
const std::vector<std::string> types{
    "人名", "法人名", "政治的組織名", "その他の組織名",
    "地名", "施設名", "製品名",       "イベント名"};

struct Span {
        std::size_t start{}, end{};
        std::string type;
};
struct Example {
        std::string text;
        std::vector<Span> spans;
};
struct Options {
        std::string command, train, validation, data, output, bundle, text;
        std::size_t epochs{3}, embedding_dim{16}, hidden_dim{32}, window{2};
        float learning_rate{0.03f};
        std::uint64_t seed{42};
};

Options options(int argc, char **argv) {
    if (argc < 2)
        throw std::invalid_argument("command required: train|evaluate|infer");
    Options out;
    out.command = argv[1];
    std::map<std::string, std::string> values;
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 == argc || !std::string(argv[i]).starts_with("--"))
            throw std::invalid_argument("options require --name value pairs");
        values[argv[i]] = argv[i + 1];
    }
    auto get = [&](const char *key, std::string fallback = {}) {
        auto it = values.find(key);
        return it == values.end() ? fallback : it->second;
    };
    out.train = get("--train");
    out.validation = get("--validation");
    out.data = get("--data");
    out.output = get("--output");
    out.bundle = get("--bundle");
    out.text = get("--text");
    out.epochs = std::stoull(get("--epochs", "3"));
    out.embedding_dim = std::stoull(get("--embedding-dim", "16"));
    out.hidden_dim = std::stoull(get("--hidden-dim", "32"));
    out.window = std::stoull(get("--window", "2"));
    out.learning_rate = std::stof(get("--learning-rate", "0.03"));
    out.seed = std::stoull(get("--seed", "42"));
    return out;
}

std::vector<Example> load_examples(const std::string &path) {
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open dataset: " + path);
    std::vector<Example> result;
    std::string line;
    for (std::size_t line_number = 1; std::getline(stream, line);
         ++line_number) {
        if (line.empty())
            continue;
        auto json = nlohmann::json::parse(line);
        Example example{json.at("text").get<std::string>(), {}};
        const auto cps = decode_utf8(example.text);
        for (const auto &entity : json.at("entities")) {
            const auto span =
                entity.at("codepoint_span").get<std::vector<std::size_t>>();
            if (span.size() != 2 || span[0] >= span[1] || span[1] > cps.size())
                throw std::runtime_error("invalid codepoint span at line " +
                                         std::to_string(line_number));
            example.spans.push_back(
                {span[0], span[1], entity.at("type").get<std::string>()});
        }
        result.push_back(std::move(example));
    }
    return result;
}

std::string encode(char32_t cp) {
    const Utf8CodePoint unit{cp, 0, 0};
    std::string s;
    if (cp < 0x80)
        s.push_back(static_cast<char>(cp));
    else if (cp < 0x800) {
        s += static_cast<char>(0xc0 | cp >> 6);
        s += static_cast<char>(0x80 | (cp & 63));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xe0 | cp >> 12);
        s += static_cast<char>(0x80 | ((cp >> 6) & 63));
        s += static_cast<char>(0x80 | (cp & 63));
    } else {
        s += static_cast<char>(0xf0 | cp >> 18);
        s += static_cast<char>(0x80 | ((cp >> 12) & 63));
        s += static_cast<char>(0x80 | ((cp >> 6) & 63));
        s += static_cast<char>(0x80 | (cp & 63));
    }
    (void)unit;
    return s;
}

std::vector<std::string> labels() {
    std::vector<std::string> result{"O"};
    for (const auto &type : types) {
        result.push_back("B-" + type);
        result.push_back("I-" + type);
    }
    return result;
}

struct Model {
        std::size_t embedding_dim, hidden_dim, window;
        std::vector<std::string> vocab, label_names{labels()};
        std::unordered_map<std::string, std::size_t> vocab_index, label_index;
        std::vector<float> embedding, hidden_weight, hidden_bias, output_weight,
            output_bias;
        std::vector<std::pair<std::string, std::string>> lexicon;

        Model(const std::vector<Example> &train, const Options &option)
            : embedding_dim(option.embedding_dim),
              hidden_dim(option.hidden_dim), window(option.window) {
            std::set<std::string> chars;
            for (const auto &example : train)
                for (const auto &cp : decode_utf8(example.text))
                    chars.insert(encode(cp.value));
            vocab = {"<PAD>", "<UNK>"};
            vocab.insert(vocab.end(), chars.begin(), chars.end());
            for (std::size_t i = 0; i < vocab.size(); ++i)
                vocab_index[vocab[i]] = i;
            for (std::size_t i = 0; i < label_names.size(); ++i)
                label_index[label_names[i]] = i;
            std::map<std::string, std::map<std::string, std::size_t>>
                lexicon_counts;
            for (const auto &example : train) {
                const auto cps = decode_utf8(example.text);
                for (const auto &span : example.spans) {
                    const auto start = cps[span.start].byte_start;
                    const auto end = cps[span.end - 1].byte_end;
                    ++lexicon_counts[example.text.substr(start, end - start)]
                                    [span.type];
                }
            }
            for (const auto &[surface, counts] : lexicon_counts) {
                const auto best =
                    std::max_element(counts.begin(), counts.end(),
                                     [](const auto &a, const auto &b) {
                                         return a.second < b.second;
                                     });
                lexicon.emplace_back(surface, best->first);
            }
            std::mt19937_64 random(option.seed);
            std::normal_distribution<float> normal(0, 0.05f);
            embedding.resize(vocab.size() * embedding_dim);
            hidden_weight.resize(hidden_dim * embedding_dim * 2);
            hidden_bias.assign(hidden_dim, 0);
            output_weight.resize(class_count * hidden_dim);
            output_bias.assign(class_count, 0);
            for (auto *vector : {&embedding, &hidden_weight, &output_weight})
                for (auto &value : *vector)
                    value = normal(random);
        }

        std::vector<std::size_t> ids(const Example &example) const {
            std::vector<std::size_t> result;
            for (const auto &cp : decode_utf8(example.text)) {
                auto it = vocab_index.find(encode(cp.value));
                result.push_back(it == vocab_index.end() ? 1 : it->second);
            }
            return result;
        }
        std::vector<std::size_t> gold(const Example &example) const {
            std::vector<std::size_t> result(decode_utf8(example.text).size(),
                                            0);
            for (const auto &span : example.spans) {
                auto b = label_index.at("B-" + span.type),
                     in = label_index.at("I-" + span.type);
                result[span.start] = b;
                for (std::size_t i = span.start + 1; i < span.end; ++i)
                    result[i] = in;
            }
            return result;
        }
        std::vector<float> context(const std::vector<std::size_t> &input,
                                   std::size_t pos) const {
            std::vector<float> result(embedding_dim * 2);
            const auto first = pos > window ? pos - window : 0;
            std::size_t lc = 0, rc = 0;
            for (std::size_t i = first; i <= pos; ++i, ++lc)
                for (std::size_t d = 0; d < embedding_dim; ++d)
                    result[d] += embedding[input[i] * embedding_dim + d];
            for (std::size_t i = pos;
                 i < std::min(input.size(), pos + window + 1); ++i, ++rc)
                for (std::size_t d = 0; d < embedding_dim; ++d)
                    result[embedding_dim + d] +=
                        embedding[input[i] * embedding_dim + d];
            for (std::size_t d = 0; d < embedding_dim; ++d) {
                result[d] /= lc;
                result[embedding_dim + d] /= rc;
            }
            return result;
        }
        double train_one(const Example &example, float rate) {
            auto input = ids(example), target = gold(example);
            double loss = 0;
            for (std::size_t pos = 0; pos < input.size(); ++pos) {
                auto ctx = context(input, pos);
                std::vector<float> hidden(hidden_dim), logits(class_count);
                for (std::size_t h = 0; h < hidden_dim; ++h) {
                    float z = hidden_bias[h];
                    for (std::size_t d = 0; d < ctx.size(); ++d)
                        z += hidden_weight[h * ctx.size() + d] * ctx[d];
                    hidden[h] = std::tanh(z);
                }
                float maximum = -1e30f;
                for (std::size_t l = 0; l < class_count; ++l) {
                    logits[l] = output_bias[l];
                    for (std::size_t h = 0; h < hidden_dim; ++h)
                        logits[l] +=
                            output_weight[l * hidden_dim + h] * hidden[h];
                    maximum = std::max(maximum, logits[l]);
                }
                float total = 0;
                for (auto &value : logits) {
                    value = std::exp(value - maximum);
                    total += value;
                }
                for (auto &value : logits)
                    value /= total;
                loss -= std::log(std::max(logits[target[pos]], 1e-12f));
                logits[target[pos]] -= 1;
                std::vector<float> dh(hidden_dim), dc(ctx.size());
                for (std::size_t l = 0; l < class_count; ++l)
                    for (std::size_t h = 0; h < hidden_dim; ++h)
                        dh[h] += output_weight[l * hidden_dim + h] * logits[l];
                for (std::size_t l = 0; l < class_count; ++l) {
                    for (std::size_t h = 0; h < hidden_dim; ++h)
                        output_weight[l * hidden_dim + h] -=
                            rate * logits[l] * hidden[h];
                    output_bias[l] -= rate * logits[l];
                }
                for (std::size_t h = 0; h < hidden_dim; ++h)
                    dh[h] *= 1 - hidden[h] * hidden[h];
                for (std::size_t h = 0; h < hidden_dim; ++h)
                    for (std::size_t d = 0; d < ctx.size(); ++d)
                        dc[d] += hidden_weight[h * ctx.size() + d] * dh[h];
                for (std::size_t h = 0; h < hidden_dim; ++h) {
                    for (std::size_t d = 0; d < ctx.size(); ++d)
                        hidden_weight[h * ctx.size() + d] -=
                            rate * dh[h] * ctx[d];
                    hidden_bias[h] -= rate * dh[h];
                }
                const auto first = pos > window ? pos - window : 0;
                const auto lc = pos - first + 1;
                const auto limit = std::min(input.size(), pos + window + 1);
                const auto rc = limit - pos;
                for (std::size_t i = first; i <= pos; ++i)
                    for (std::size_t d = 0; d < embedding_dim; ++d)
                        embedding[input[i] * embedding_dim + d] -=
                            rate * dc[d] / lc;
                for (std::size_t i = pos; i < limit; ++i)
                    for (std::size_t d = 0; d < embedding_dim; ++d)
                        embedding[input[i] * embedding_dim + d] -=
                            rate * dc[embedding_dim + d] / rc;
            }
            return loss;
        }
        void save(const std::string &path, const Options &option) const {
            nlohmann::json fingerprint = nlohmann::json::object();
            const auto manifest_path =
                std::filesystem::path(option.train).parent_path() /
                "manifest.json";
            std::ifstream manifest_stream(manifest_path);
            if (manifest_stream) {
                nlohmann::json manifest;
                manifest_stream >> manifest;
                fingerprint = {{"revision", manifest.value("revision", "")},
                               {"source_sha256",
                                manifest.value("actual_source_sha256", "")},
                               {"split_seed", manifest.value("seed", 0)}};
            }
            nlohmann::json lexicon_json = nlohmann::json::array();
            for (const auto &[surface, type] : lexicon)
                lexicon_json.push_back({{"surface", surface}, {"type", type}});
            nlohmann::json root{{"format", "ai_cpp_ner_bundle_v1"},
                                {"labels", label_names},
                                {"vocab", vocab},
                                {"config",
                                 {{"encoder", "bidirectional_window"},
                                  {"embedding_dim", embedding_dim},
                                  {"hidden_dim", hidden_dim},
                                  {"window", window},
                                  {"epochs", option.epochs},
                                  {"seed", option.seed}}},
                                {"data_fingerprint", fingerprint},
                                {"lexicon", lexicon_json},
                                {"score_semantics", "uncalibrated_logit"},
                                {"weights",
                                 {{"embedding", embedding},
                                  {"hidden_weight", hidden_weight},
                                  {"hidden_bias", hidden_bias},
                                  {"output_weight", output_weight},
                                  {"output_bias", output_bias}}}};
            std::ofstream stream(path);
            if (!stream)
                throw std::runtime_error("cannot write bundle: " + path);
            stream << root.dump() << '\n';
        }
};

struct Counts {
        std::size_t predicted{}, gold{}, correct{};
};
void evaluate(const std::string &bundle, const std::string &data) {
    ModelEntityExtractor extractor(bundle);
    auto examples = load_examples(data);
    std::map<std::string, Counts> by_type;
    Counts all;
    const auto started = std::chrono::steady_clock::now();
    std::size_t codepoints = 0;
    for (const auto &example : examples) {
        auto predicted = extractor.extract(example.text);
        codepoints += decode_utf8(example.text).size();
        std::set<std::tuple<std::size_t, std::size_t, std::string>> p, g;
        for (const auto &mention : predicted)
            p.emplace(mention.start, mention.end, to_string(mention.type));
        const auto cps = decode_utf8(example.text);
        for (const auto &span : example.spans)
            g.emplace(cps[span.start].byte_start, cps[span.end - 1].byte_end,
                      span.type);
        all.predicted += p.size();
        all.gold += g.size();
        for (const auto &item : p) {
            by_type[std::get<2>(item)].predicted++;
            if (g.contains(item)) {
                all.correct++;
                by_type[std::get<2>(item)].correct++;
            }
        }
        for (const auto &item : g)
            by_type[std::get<2>(item)].gold++;
    }
    auto metrics = [](Counts c) {
        double precision = c.predicted ? double(c.correct) / c.predicted : 0,
               recall = c.gold ? double(c.correct) / c.gold : 0;
        return nlohmann::json{
            {"precision", precision},
            {"recall", recall},
            {"f1", precision + recall
                       ? 2 * precision * recall / (precision + recall)
                       : 0},
            {"correct", c.correct},
            {"predicted", c.predicted},
            {"gold", c.gold}};
    };
    nlohmann::json report{
        {"exact_entity", metrics(all)},
        {"by_type", nlohmann::json::object()},
        {"model_parameter_bytes", extractor.parameter_bytes()},
        {"examples", examples.size()}};
    report["bundle_bytes"] = std::filesystem::file_size(bundle);
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
        report["process_max_rss_kib"] = usage.ru_maxrss;
    for (const auto &[type, counts] : by_type)
        report["by_type"][type] = metrics(counts);
    const auto seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    report["runtime"] = {
        {"seconds", seconds},
        {"codepoints", codepoints},
        {"codepoints_per_second", seconds ? codepoints / seconds : 0}};
    std::cout << report.dump(2) << '\n';
}
} // namespace

int main(int argc, char **argv) {
    try {
        auto option = options(argc, argv);
        if (option.command == "train") {
            if (option.train.empty() || option.output.empty())
                throw std::invalid_argument(
                    "train requires --train and --output");
            auto train = load_examples(option.train);
            if (train.empty())
                throw std::runtime_error("empty training set");
            Model model(train, option);
            std::mt19937_64 random(option.seed);
            for (std::size_t epoch = 0; epoch < option.epochs; ++epoch) {
                std::shuffle(train.begin(), train.end(), random);
                double loss = 0;
                std::size_t chars = 0;
                for (const auto &example : train) {
                    loss += model.train_one(example, option.learning_rate);
                    chars += decode_utf8(example.text).size();
                }
                std::cerr << "epoch=" << (epoch + 1) << " loss=" << loss / chars
                          << " chars=" << chars << '\n';
            }
            model.save(option.output, option);
            ModelEntityExtractor reload(option.output);
            (void)reload.extract(train.front().text);
            std::cerr << "saved_and_reloaded=" << option.output
                      << " parameter_bytes=" << reload.parameter_bytes()
                      << '\n';
            if (!option.validation.empty())
                evaluate(option.output, option.validation);
        } else if (option.command == "evaluate") {
            if (option.bundle.empty() || option.data.empty())
                throw std::invalid_argument(
                    "evaluate requires --bundle and --data");
            evaluate(option.bundle, option.data);
        } else if (option.command == "infer") {
            if (option.bundle.empty())
                throw std::invalid_argument(
                    "infer requires --bundle and --text");
            ModelEntityExtractor extractor(option.bundle);
            nlohmann::json result = nlohmann::json::array();
            for (const auto &m : extractor.extract(option.text, "cli"))
                result.push_back({{"type", to_string(m.type)},
                                  {"start", m.start},
                                  {"end", m.end},
                                  {"surface", m.surface},
                                  {"source", to_string(m.source)},
                                  {"score", m.score}});
            std::cout << result.dump(2) << '\n';
        } else
            throw std::invalid_argument("unknown command: " + option.command);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ner_cli: " << error.what() << '\n';
        return 1;
    }
}
