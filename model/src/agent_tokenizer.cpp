#include "ai/model/agent_tokenizer.h"

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
AgentTokenizer from_serialized(std::string serialized) {
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
    return AgentTokenizer(std::move(holder), std::move(serialized));
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
    std::string serialized)
    : _holder(std::move(holder)), _serialized(std::move(serialized)) {}

AgentTokenizer AgentTokenizer::train(const std::filesystem::path &train_jsonl,
                                     std::size_t vocabulary_size) {
    if (vocabulary_size < special_count + 256)
        throw std::invalid_argument(
            "vocabulary must include byte fallback pieces");
    JsonlSentenceIterator sentences(train_jsonl);
    if (!sentences.status().ok())
        throw std::runtime_error("cannot read tokenizer train split: " +
                                 train_jsonl.string());
    const std::unordered_map<std::string, std::string> options = {
        {"model_type", "bpe"},
        {"vocab_size", std::to_string(vocabulary_size)},
        {"byte_fallback", "true"},
        {"character_coverage", "0.9995"},
        {"normalization_rule_name", "identity"},
        {"add_dummy_prefix", "false"},
        {"remove_extra_whitespaces", "false"},
        {"hard_vocab_limit", "false"},
        {"pad_id", "0"},
        {"unk_id", "1"},
        {"bos_id", "2"},
        {"eos_id", "3"},
        {"control_symbols", "<CHAT>,<PARSE>,<PLAN>,<EVALUATE>,<SUMMARIZE>,<"
                            "MEMORY_WRITE>,<MEMORY_QUERY>,<FINAL>,<TOOL>"},
        {"num_threads", "1"},
        {"shuffle_input_sentence", "false"},
        {"max_sentence_length", "1048576"},
        {"minloglevel", "1"}};
    std::string serialized;
    check(sentencepiece::SentencePieceTrainer::Train(options, &sentences,
                                                     &serialized));
    return from_serialized(std::move(serialized));
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
void AgentTokenizer::save(const std::filesystem::path &path) const {
    const auto temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.write(_serialized.data(),
                 static_cast<std::streamsize>(_serialized.size()));
    if (!output)
        throw std::runtime_error("cannot write tokenizer");
    output.close();
    std::filesystem::rename(temporary, path);
}
AgentTokenizer AgentTokenizer::load(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read tokenizer: " + path.string());
    return from_serialized(
        std::string(std::istreambuf_iterator<char>(input), {}));
}
std::string AgentTokenizer::fingerprint() const {
    std::ostringstream result;
    result << std::hex << std::setw(16) << std::setfill('0')
           << hash_bytes(_serialized);
    return result.str();
}
} // namespace ai::model
