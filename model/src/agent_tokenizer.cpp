#include "ai/model/agent_tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sentencepiece_model.pb.h>
#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <tuple>
#include <utility>

namespace ai::model {
namespace sentencepiece_detail {
class Holder {
    public:
        sentencepiece::SentencePieceProcessor processor;
};
} // namespace sentencepiece_detail
namespace {
constexpr std::array<const char *, 13> specials = {
    "<pad>",          "<unk>",   "<s>",        "</s>",        "<CHAT>",
    "<PARSE>",        "<PLAN>",  "<EVALUATE>", "<SUMMARIZE>", "<MEMORY_WRITE>",
    "<MEMORY_QUERY>", "<FINAL>", "<TOOL>"};

void check(const sentencepiece::util::Status &status) {
    if (!status.ok())
        throw std::runtime_error("SentencePiece: " + status.ToString());
}
std::string byte_piece(unsigned char byte) {
    char piece[7];
    std::snprintf(piece, sizeof(piece), "<0x%02X>", byte);
    return piece;
}
std::size_t literal_length(std::string_view text, std::size_t offset) {
    const auto byte = static_cast<unsigned char>(text[offset]);
    if (byte <= 0x20 || byte == 0x7f)
        return 1;
    return text.substr(offset, 3) == "▁" ? 3 : 0;
}
template <class Text, class Literal>
void spans(std::string_view text, Text normal, Literal literal) {
    std::size_t start = 0;
    for (std::size_t index = 0; index < text.size();) {
        const auto length = literal_length(text, index);
        if (!length) {
            ++index;
            continue;
        }
        if (index > start)
            normal(text.substr(start, index - start));
        literal(text.substr(index, length));
        index += length;
        start = index;
    }
    if (start < text.size())
        normal(text.substr(start));
}
std::uint64_t hash_bytes(std::string_view bytes) {
    std::uint64_t value = 1469598103934665603ULL;
    for (unsigned char byte : bytes) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    return value;
}
class JsonlSentenceIterator final : public sentencepiece::SentenceIterator {
    public:
        explicit JsonlSentenceIterator(const std::filesystem::path &path)
            : _input(path) {
            if (!_input)
                _status = {sentencepiece::util::StatusCode::kNotFound,
                           path.string()};
            next_value();
        }
        bool done() const override {
            return !_status.ok() || (_value.empty() && _finished);
        }
        void Next() override { next_value(); }
        const std::string &value() const override { return _value; }
        sentencepiece::util::Status status() const override { return _status; }

    private:
        void next_value() {
            _value.clear();
            if (!_status.ok())
                return;
            if (_finish_after_value) {
                _finished = true;
                return;
            }
            if (!_pending.empty()) {
                _value = std::move(_pending.front());
                _pending.pop_front();
            } else {
                std::string line;
                while (std::getline(_input, line)) {
                    try {
                        const auto json = nlohmann::json::parse(line);
                        const auto add_chunks = [&](const std::string &text) {
                            for (std::size_t begin = 0; begin < text.size();) {
                                auto end =
                                    std::min(text.size(),
                                             begin + _maximum_sentence_bytes);
                                while (end < text.size() &&
                                       (static_cast<unsigned char>(text[end]) &
                                        0xc0U) == 0x80U)
                                    --end;
                                if (end == begin)
                                    end = std::min(text.size(),
                                                   begin +
                                                       _maximum_sentence_bytes);
                                _pending.push_back(
                                    text.substr(begin, end - begin));
                                begin = end;
                            }
                        };
                        if (json.contains("text"))
                            add_chunks(json.at("text").get<std::string>());
                        if (json.contains("utterances"))
                            for (const auto &utterance : json.at("utterances"))
                                add_chunks(
                                    utterance.at("text").get<std::string>());
                    } catch (const std::exception &error) {
                        _status = {
                            sentencepiece::util::StatusCode::kInvalidArgument,
                            error.what()};
                        return;
                    }
                    if (!_pending.empty()) {
                        _value = std::move(_pending.front());
                        _pending.pop_front();
                        break;
                    }
                }
                if (_value.empty()) {
                    _finished = true;
                    if (!_input.eof())
                        _status = {sentencepiece::util::StatusCode::kDataLoss,
                                   "JSONL read failed"};
                    return;
                }
            }
            const auto remaining = _maximum_training_bytes - _selected_bytes;
            if (_value.size() > remaining) {
                auto end = remaining;
                while (end > 0 && end < _value.size() &&
                       (static_cast<unsigned char>(_value[end]) & 0xc0U) ==
                           0x80U)
                    --end;
                _value.resize(end);
                _pending.clear();
                _finish_after_value = true;
            }
            _selected_bytes += _value.size();
            if (_selected_bytes >= _maximum_training_bytes)
                _finish_after_value = true;
        }
        std::ifstream _input;
        std::deque<std::string> _pending;
        std::string _value;
        sentencepiece::util::Status _status;
        std::size_t _selected_bytes = 0;
        bool _finish_after_value = false;
        bool _finished = false;
        static constexpr std::size_t _maximum_training_bytes =
            16U * 1024U * 1024U;
        static constexpr std::size_t _maximum_sentence_bytes = 64U * 1024U;
};
class VectorSentenceIterator final : public sentencepiece::SentenceIterator {
public:
    explicit VectorSentenceIterator(std::vector<std::string> values)
        : _values(std::move(values)) {}
    bool done() const override { return _index >= _values.size(); }
    void Next() override { if (!done()) ++_index; }
    const std::string &value() const override { return _values.at(_index); }
    sentencepiece::util::Status status() const override { return {}; }
private:
    std::vector<std::string> _values;
    std::size_t _index = 0;
};

std::uint64_t mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}
std::vector<std::string> texts(const nlohmann::json &json) {
    std::vector<std::string> result;
    if (json.contains("text"))
        result.push_back(json.at("text").get<std::string>());
    if (json.contains("utterances"))
        for (const auto &utterance : json.at("utterances"))
            result.push_back(utterance.at("text").get<std::string>());
    return result;
}
struct Candidate {
    std::uint64_t priority = 0;
    std::size_t index = 0;
    std::size_t bytes = 0;
};
struct SourceSample {
    std::vector<std::string> sentences;
    std::size_t scanned_documents = 0;
    std::size_t selected_documents = 0;
    std::size_t selected_bytes = 0;
    std::size_t input_text_bytes = 0;
    std::string fingerprint;
};
std::size_t utf8_prefix(std::string_view text, std::size_t maximum) {
    auto end = std::min(text.size(), maximum);
    while (end && end < text.size() &&
           (static_cast<unsigned char>(text[end]) & 0xc0U) == 0x80U)
        --end;
    return end;
}
SourceSample sample_source(const std::filesystem::path &path,
                           std::size_t requested_bytes, std::uint64_t seed) {
    std::ifstream first(path);
    if (!first)
        throw std::runtime_error("cannot read tokenizer train split: " +
                                 path.string());
    SourceSample result;
    std::vector<Candidate> candidates;
    std::uint64_t hash = 1469598103934665603ULL;
    std::string line;
    while (std::getline(first, line)) {
        for (const unsigned char byte : line) { hash ^= byte; hash *= 1099511628211ULL; }
        hash ^= static_cast<unsigned char>('\n'); hash *= 1099511628211ULL;
        const auto json = nlohmann::json::parse(line);
        std::size_t bytes = 0;
        for (const auto &text : texts(json)) bytes += text.size();
        result.input_text_bytes += bytes;
        candidates.push_back({mix(seed ^ result.scanned_documents),
                              result.scanned_documents, bytes});
        ++result.scanned_documents;
    }
    if (!first.eof() || candidates.empty())
        throw std::runtime_error("invalid or empty tokenizer train split: " +
                                 path.string());
    std::ostringstream fingerprint;
    fingerprint << std::hex << std::setw(16) << std::setfill('0') << hash;
    result.fingerprint = fingerprint.str();
    const auto target = std::min(requested_bytes, result.input_text_bytes);
    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
        return std::tie(a.priority, a.index) < std::tie(b.priority, b.index);
    });
    std::vector<bool> selected(candidates.size());
    std::size_t estimated = 0;
    for (const auto &candidate : candidates) {
        if (estimated >= target) break;
        selected[candidate.index] = true;
        estimated += candidate.bytes;
        ++result.selected_documents;
    }
    std::ifstream second(path);
    std::size_t index = 0;
    while (result.selected_bytes < target && std::getline(second, line)) {
        if (!selected.at(index++)) continue;
        for (const auto &text : texts(nlohmann::json::parse(line))) {
            for (std::size_t begin = 0; begin < text.size() &&
                 result.selected_bytes < target;) {
                const auto room = target - result.selected_bytes;
                const auto length = utf8_prefix(
                    std::string_view(text).substr(begin),
                    std::min<std::size_t>(64U * 1024U, room));
                if (!length) break;
                result.sentences.push_back(text.substr(begin, length));
                result.selected_bytes += length;
                begin += length;
            }
        }
    }
    return result;
}
std::string train_serialized(sentencepiece::SentenceIterator *sentences,
                             std::size_t vocabulary_size) {
    if (vocabulary_size < AgentTokenizer::special_count + 256)
        throw std::invalid_argument("vocabulary must include byte fallback pieces");
    const std::unordered_map<std::string, std::string> options = {
        {"model_type", "bpe"}, {"vocab_size", std::to_string(vocabulary_size)},
        {"byte_fallback", "true"}, {"character_coverage", "0.9995"},
        {"normalization_rule_name", "identity"}, {"add_dummy_prefix", "false"},
        {"remove_extra_whitespaces", "false"}, {"hard_vocab_limit", "false"},
        {"pad_id", "0"}, {"unk_id", "1"}, {"bos_id", "2"}, {"eos_id", "3"},
        {"control_symbols", "<CHAT>,<PARSE>,<PLAN>,<EVALUATE>,<SUMMARIZE>,<MEMORY_WRITE>,<MEMORY_QUERY>,<FINAL>,<TOOL>"},
        {"num_threads", "1"}, {"shuffle_input_sentence", "false"},
        {"max_sentence_length", "1048576"}, {"minloglevel", "1"}};
    std::string serialized;
    check(sentencepiece::SentencePieceTrainer::Train(options, sentences,
                                                     &serialized));
    return serialized;
}

AgentTokenizer from_serialized(std::string serialized,
                               std::string training_metadata = {}) {
    auto holder = std::make_shared<sentencepiece_detail::Holder>();
    check(holder->processor.LoadFromSerializedProto(serialized));
    const auto &model = holder->processor.model_proto();
    const auto &normalizer = model.normalizer_spec();
    if (model.trainer_spec().model_type() != sentencepiece::TrainerSpec::BPE ||
        !model.trainer_spec().byte_fallback() ||
        normalizer.name() != "identity" ||
        !normalizer.precompiled_charsmap().empty() ||
        normalizer.add_dummy_prefix() ||
        normalizer.remove_extra_whitespaces() ||
        !normalizer.escape_whitespaces() || model.has_denormalizer_spec())
        throw std::invalid_argument(
            "unsupported tokenizer normalization or model type");
    if (holder->processor.GetPieceSize() < AgentTokenizer::special_count + 256)
        throw std::invalid_argument("incomplete byte-fallback tokenizer");
    for (int id = 0; id < AgentTokenizer::special_count; ++id)
        if (holder->processor.IdToPiece(id) != specials[id] ||
            (id == AgentTokenizer::unk_id ? !holder->processor.IsUnknown(id)
                                          : !holder->processor.IsControl(id)))
            throw std::invalid_argument("tokenizer special ID mismatch");
    for (int byte = 0; byte < 256; ++byte)
        if (!holder->processor.IsByte(
                holder->processor.PieceToId(byte_piece(byte))))
            throw std::invalid_argument(
                "tokenizer byte fallback is incomplete");
    return AgentTokenizer(std::move(holder), std::move(serialized),
                          std::move(training_metadata));
}
} // namespace

const char *mode_token(AgentMode mode) {
    const auto id = static_cast<std::size_t>(mode);
    if (id < 4 || id >= specials.size())
        throw std::invalid_argument("invalid agent mode");
    return specials[id];
}

AgentTokenizer::AgentTokenizer(
    std::shared_ptr<const sentencepiece_detail::Holder> holder,
    std::string serialized, std::string training_metadata)
    : _holder(std::move(holder)), _serialized(std::move(serialized)),
      _training_metadata(std::move(training_metadata)) {}

AgentTokenizer AgentTokenizer::train(const std::filesystem::path &train_jsonl,
                                     std::size_t vocabulary_size) {
    JsonlSentenceIterator sentences(train_jsonl);
    if (!sentences.status().ok())
        throw std::runtime_error("cannot read tokenizer train split: " +
                                 train_jsonl.string());
    return from_serialized(train_serialized(&sentences, vocabulary_size));
}

AgentTokenizer AgentTokenizer::train_balanced(
    const std::filesystem::path &jawiki_train,
    const std::filesystem::path &conversation_train,
    const BalancedTokenizerConfig &config) {
    auto jawiki = sample_source(jawiki_train, config.jawiki_bytes, config.seed);
    auto conversation = sample_source(conversation_train,
                                      config.conversation_bytes,
                                      config.seed ^ 0x434f4e5645525345ULL);
    std::vector<std::string> combined;
    combined.reserve(jawiki.sentences.size() + conversation.sentences.size());
    std::move(jawiki.sentences.begin(), jawiki.sentences.end(),
              std::back_inserter(combined));
    std::move(conversation.sentences.begin(), conversation.sentences.end(),
              std::back_inserter(combined));
    VectorSentenceIterator sentences(std::move(combined));
    auto serialized = train_serialized(&sentences, config.vocabulary_size);
    const auto tokenizer_fingerprint = [&] {
        std::ostringstream value;
        value << std::hex << std::setw(16) << std::setfill('0')
              << hash_bytes(serialized);
        return value.str();
    }();
    const auto source_json = [](const std::filesystem::path &path,
                                const SourceSample &sample,
                                std::size_t requested) {
        return nlohmann::json{{"path", path.string()},
                              {"fingerprint", sample.fingerprint},
                              {"scanned_documents", sample.scanned_documents},
                              {"input_text_bytes", sample.input_text_bytes},
                              {"requested_bytes", requested},
                              {"selected_documents", sample.selected_documents},
                              {"selected_bytes", sample.selected_bytes}};
    };
    const nlohmann::json metadata = {
        {"format", "ai_cpp_balanced_tokenizer_training"}, {"version", 1},
        {"tokenizer_fingerprint", tokenizer_fingerprint},
        {"seed", config.seed}, {"vocabulary_size", config.vocabulary_size},
        {"sampling", "deterministic_uniform_document_priority"},
        {"sources", {{"jawiki", source_json(jawiki_train, jawiki, config.jawiki_bytes)},
                     {"conversation", source_json(conversation_train, conversation,
                                                   config.conversation_bytes)}}}};
    return from_serialized(std::move(serialized), metadata.dump(2));
}

std::vector<std::int32_t> AgentTokenizer::encode(std::string_view text) const {
    std::vector<std::int32_t> result;
    spans(
        text,
        [&](std::string_view part) {
            std::vector<int> ids;
            check(_holder->processor.Encode(part, &ids));
            result.insert(result.end(), ids.begin(), ids.end());
        },
        [&](std::string_view part) {
            for (const unsigned char byte : part)
                result.push_back(
                    _holder->processor.PieceToId(byte_piece(byte)));
        });
    return result;
}
std::string AgentTokenizer::decode(const std::vector<std::int32_t> &ids) const {
    for (const auto id : ids)
        if (id < 0 || id >= _holder->processor.GetPieceSize())
            throw std::out_of_range("token ID outside vocabulary");
    std::string result;
    check(_holder->processor.Decode(std::vector<int>(ids.begin(), ids.end()),
                                    &result));
    return result;
}
std::size_t AgentTokenizer::vocabulary_size() const noexcept {
    return static_cast<std::size_t>(_holder->processor.GetPieceSize());
}
std::size_t AgentTokenizer::byte_fallback_count(
    const std::vector<std::int32_t> &ids) const {
    return static_cast<std::size_t>(std::count_if(
        ids.begin(), ids.end(), [&](std::int32_t id) {
            return id >= 0 && id < _holder->processor.GetPieceSize() &&
                   _holder->processor.IsByte(id);
        }));
}
void AgentTokenizer::save(const std::filesystem::path &path) const {
    const auto temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.write(_serialized.data(),
                 static_cast<std::streamsize>(_serialized.size()));
    if (!output)
        throw std::runtime_error("cannot write tokenizer");
    output.close();
    std::filesystem::rename(temporary, path);
    if (!_training_metadata.empty()) {
        const auto metadata_path = path.string() + ".metadata.json";
        const auto metadata_temporary = metadata_path + ".tmp";
        std::ofstream metadata(metadata_temporary, std::ios::trunc);
        metadata << _training_metadata << '\n';
        if (!metadata) throw std::runtime_error("cannot write tokenizer metadata");
        metadata.close();
        std::filesystem::rename(metadata_temporary, metadata_path);
    }
}
AgentTokenizer AgentTokenizer::load(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read tokenizer: " + path.string());
    auto serialized = std::string(std::istreambuf_iterator<char>(input), {});
    std::string metadata;
    std::ifstream metadata_input(path.string() + ".metadata.json");
    if (metadata_input) {
        metadata.assign(std::istreambuf_iterator<char>(metadata_input), {});
        const auto parsed = nlohmann::json::parse(metadata);
        std::ostringstream expected;
        expected << std::hex << std::setw(16) << std::setfill('0')
                 << hash_bytes(serialized);
        if (parsed.value("tokenizer_fingerprint", "") != expected.str())
            throw std::runtime_error("tokenizer metadata fingerprint mismatch");
    }
    return from_serialized(std::move(serialized), std::move(metadata));
}
std::string AgentTokenizer::fingerprint() const {
    std::ostringstream result;
    result << std::hex << std::setw(16) << std::setfill('0')
           << hash_bytes(_serialized);
    return result.str();
}

TokenizerEfficiencyMetrics evaluate_tokenizer(
    const AgentTokenizer &tokenizer, const std::filesystem::path &jsonl,
    std::size_t context) {
    if (!context) throw std::invalid_argument("tokenizer evaluation context must be positive");
    std::ifstream input(jsonl);
    if (!input) throw std::runtime_error("cannot read tokenizer evaluation split: " + jsonl.string());
    TokenizerEfficiencyMetrics result;
    std::string line;
    while (std::getline(input, line)) {
        const auto json = nlohmann::json::parse(line);
        std::size_t document_tokens = 0;
        for (const auto &text : texts(json)) {
            const auto ids = tokenizer.encode(text);
            result.input_bytes += text.size();
            result.characters += static_cast<std::size_t>(std::count_if(
                text.begin(), text.end(), [](unsigned char byte) {
                    return (byte & 0xc0U) != 0x80U;
                }));
            result.tokens += ids.size();
            document_tokens += ids.size();
            result.byte_fallback_tokens += tokenizer.byte_fallback_count(ids);
        }
        ++result.documents;
        if (document_tokens > context) ++result.over_context;
    }
    if (!input.eof() || !result.documents)
        throw std::runtime_error("invalid or empty tokenizer evaluation split");
    return result;
}
} // namespace ai::model
