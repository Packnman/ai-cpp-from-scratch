#include "ai/ner/extractor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

namespace ai::ner {
namespace {
std::string encode(char32_t cp) {
    std::string out;
    if (cp <= 0x7f)
        out.push_back(static_cast<char>(cp));
    else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
    return out;
}
template <class T>
std::vector<T> vector_field(const nlohmann::json &json, const char *name) {
    if (!json.contains(name) || !json[name].is_array())
        throw std::runtime_error(std::string("bundle field is not an array: ") +
                                 name);
    return json[name].get<std::vector<T>>();
}
} // namespace

struct ModelEntityExtractor::Impl {
        std::size_t embedding_dim{}, hidden_dim{}, window{};
        std::vector<std::string> labels;
        std::unordered_map<std::string, std::size_t> vocab;
        std::vector<float> embedding, hidden_weight, hidden_bias, output_weight,
            output_bias;
        std::vector<std::pair<std::string, EntityType>> lexicon;

        explicit Impl(const std::string &path) {
            std::ifstream stream(path);
            if (!stream)
                throw std::runtime_error("cannot open NER bundle: " + path);
            nlohmann::json root;
            stream >> root;
            if (root.value("format", "") != "ai_cpp_ner_bundle_v1")
                throw std::runtime_error("unsupported NER bundle format");
            const auto &config = root.at("config");
            if (config.value("encoder", "") != "bidirectional_window")
                throw std::runtime_error("unsupported NER encoder");
            embedding_dim = config.at("embedding_dim").get<std::size_t>();
            hidden_dim = config.at("hidden_dim").get<std::size_t>();
            window = config.at("window").get<std::size_t>();
            labels = vector_field<std::string>(root, "labels");
            const auto chars = vector_field<std::string>(root, "vocab");
            for (std::size_t i = 0; i < chars.size(); ++i)
                vocab.emplace(chars[i], i);
            const auto &weights = root.at("weights");
            embedding = vector_field<float>(weights, "embedding");
            hidden_weight = vector_field<float>(weights, "hidden_weight");
            hidden_bias = vector_field<float>(weights, "hidden_bias");
            output_weight = vector_field<float>(weights, "output_weight");
            output_bias = vector_field<float>(weights, "output_bias");
            if (root.contains("lexicon"))
                for (const auto &item : root.at("lexicon")) {
                    const auto type = entity_type_from_string(
                        item.at("type").get<std::string>());
                    if (!type)
                        throw std::runtime_error("unknown lexicon entity type");
                    lexicon.emplace_back(item.at("surface").get<std::string>(),
                                         *type);
                }
            std::sort(lexicon.begin(), lexicon.end(),
                      [](const auto &a, const auto &b) {
                          return a.first.size() > b.first.size();
                      });
            if (chars.size() < 2 || labels.size() != 17 || embedding_dim == 0 ||
                hidden_dim == 0 ||
                embedding.size() != chars.size() * embedding_dim ||
                hidden_weight.size() != hidden_dim * embedding_dim * 2 ||
                hidden_bias.size() != hidden_dim ||
                output_weight.size() != labels.size() * hidden_dim ||
                output_bias.size() != labels.size())
                throw std::runtime_error("inconsistent NER bundle dimensions");
        }

        std::vector<std::pair<std::size_t, float>>
        predict(const std::vector<Utf8CodePoint> &cps) const {
            std::vector<std::size_t> ids;
            ids.reserve(cps.size());
            for (const auto &cp : cps) {
                const auto it = vocab.find(encode(cp.value));
                ids.push_back(it == vocab.end() ? 1 : it->second);
            }
            std::vector<std::pair<std::size_t, float>> result;
            for (std::size_t pos = 0; pos < ids.size(); ++pos) {
                std::vector<float> context(embedding_dim * 2, 0.0f);
                const auto left = pos > window ? pos - window : 0;
                std::size_t left_count = 0, right_count = 0;
                for (std::size_t i = left; i <= pos; ++i) {
                    for (std::size_t d = 0; d < embedding_dim; ++d)
                        context[d] += embedding[ids[i] * embedding_dim + d];
                    ++left_count;
                }
                const auto right = std::min(ids.size(), pos + window + 1);
                for (std::size_t i = pos; i < right; ++i) {
                    for (std::size_t d = 0; d < embedding_dim; ++d)
                        context[embedding_dim + d] +=
                            embedding[ids[i] * embedding_dim + d];
                    ++right_count;
                }
                for (std::size_t d = 0; d < embedding_dim; ++d) {
                    context[d] /= static_cast<float>(left_count);
                    context[embedding_dim + d] /=
                        static_cast<float>(right_count);
                }
                std::vector<float> hidden(hidden_dim);
                for (std::size_t h = 0; h < hidden_dim; ++h) {
                    float value = hidden_bias[h];
                    for (std::size_t d = 0; d < context.size(); ++d)
                        value +=
                            hidden_weight[h * context.size() + d] * context[d];
                    hidden[h] = std::tanh(value);
                }
                std::size_t best = 0;
                float best_logit = -std::numeric_limits<float>::infinity();
                for (std::size_t label = 0; label < labels.size(); ++label) {
                    float logit = output_bias[label];
                    for (std::size_t h = 0; h < hidden_dim; ++h)
                        logit +=
                            output_weight[label * hidden_dim + h] * hidden[h];
                    if (logit > best_logit) {
                        best = label;
                        best_logit = logit;
                    }
                }
                result.emplace_back(best, best_logit);
            }
            return result;
        }
};

ModelEntityExtractor::ModelEntityExtractor(const std::string &path)
    : _impl(std::make_unique<Impl>(path)) {}
ModelEntityExtractor::~ModelEntityExtractor() = default;
ModelEntityExtractor::ModelEntityExtractor(ModelEntityExtractor &&) noexcept =
    default;
ModelEntityExtractor &
ModelEntityExtractor::operator=(ModelEntityExtractor &&) noexcept = default;

std::vector<EntityMention>
ModelEntityExtractor::extract(std::string_view text,
                              std::string_view utterance_id) const {
    const auto cps = decode_utf8(text);
    const auto predictions = _impl->predict(cps);
    std::vector<EntityMention> result;
    std::optional<EntityMention> active;
    const auto flush = [&] {
        if (active) {
            active->surface = std::string(
                text.substr(active->start, active->end - active->start));
            result.push_back(std::move(*active));
            active.reset();
        }
    };
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        const auto &[label_id, logit] = predictions[i];
        const auto &label = _impl->labels[label_id];
        if (label == "O") {
            flush();
            continue;
        }
        if (label.size() < 3 || label[1] != '-')
            throw std::runtime_error("invalid BIO label in NER bundle");
        const auto type =
            entity_type_from_string(std::string_view(label).substr(2));
        if (!type ||
            static_cast<int>(*type) > static_cast<int>(EntityType::Event))
            throw std::runtime_error("unknown model entity label");
        const bool begin = label[0] == 'B';
        // Explicit repair rule: an I tag without a compatible predecessor is B.
        if (begin || !active || active->type != *type) {
            flush();
            active = EntityMention{*type,
                                   cps[i].byte_start,
                                   cps[i].byte_end,
                                   {},
                                   std::nullopt,
                                   EntitySource::Model,
                                   std::string(utterance_id),
                                   logit};
        } else {
            active->end = cps[i].byte_end;
            active->score = std::min(*active->score, logit);
        }
    }
    flush();
    std::vector<EntityMention> lexical;
    for (const auto &[surface, type] : _impl->lexicon) {
        std::size_t start = 0;
        while ((start = text.find(surface, start)) != std::string_view::npos) {
            EntityMention mention{type,
                                  start,
                                  start + surface.size(),
                                  surface,
                                  std::nullopt,
                                  EntitySource::Model,
                                  std::string(utterance_id),
                                  std::nullopt};
            if (std::none_of(lexical.begin(), lexical.end(),
                             [&](const auto &old) {
                                 return old.start < mention.end &&
                                        mention.start < old.end;
                             }))
                lexical.push_back(std::move(mention));
            start += surface.size();
        }
    }
    result.erase(std::remove_if(result.begin(), result.end(),
                                [&](const auto &neural) {
                                    return std::any_of(
                                        lexical.begin(), lexical.end(),
                                        [&](const auto &known) {
                                            return known.start < neural.end &&
                                                   neural.start < known.end;
                                        });
                                }),
                 result.end());
    result.insert(result.end(), lexical.begin(), lexical.end());
    std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
        return a.start != b.start ? a.start < b.start : a.end < b.end;
    });
    return result;
}

std::size_t ModelEntityExtractor::parameter_bytes() const noexcept {
    return sizeof(float) *
           (_impl->embedding.size() + _impl->hidden_weight.size() +
            _impl->hidden_bias.size() + _impl->output_weight.size() +
            _impl->output_bias.size());
}
} // namespace ai::ner
