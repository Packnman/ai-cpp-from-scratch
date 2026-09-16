#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ai::model {
namespace sentencepiece_detail {
class Holder;
}

enum class AgentMode : std::int32_t {
    Chat = 4,
    Parse,
    Plan,
    Evaluate,
    Summarize,
    MemoryWrite,
    MemoryQuery,
    Final,
    Tool
};

struct BalancedTokenizerConfig {
    std::size_t vocabulary_size = 8192;
    std::size_t jawiki_bytes = 224ULL * 1024ULL * 1024ULL;
    std::size_t conversation_bytes = 32ULL * 1024ULL * 1024ULL;
    std::uint64_t seed = 42;
};

struct TokenizerEfficiencyMetrics {
    std::size_t documents = 0;
    std::size_t input_bytes = 0;
    std::size_t characters = 0;
    std::size_t tokens = 0;
    std::size_t byte_fallback_tokens = 0;
    std::size_t over_context = 0;
};

class AgentTokenizer {
    public:
        static constexpr std::int32_t pad_id = 0; // パディングトークンのID
        static constexpr std::int32_t unk_id = 1; // 未知語トークンのID
        static constexpr std::int32_t bos_id = 2; // 文頭トークンのID
        static constexpr std::int32_t eos_id = 3; // 文末トークンのID
        static constexpr std::int32_t special_count = 13; // 予約済み特殊トークンの総数

        AgentTokenizer(
            std::shared_ptr<const sentencepiece_detail::Holder> holder,
            std::string serialized, std::string training_metadata = {});

        static AgentTokenizer train(
            const std::filesystem::path &train_jsonl,
            std::size_t vocabulary_size = 8192
        );
        static AgentTokenizer train_balanced(
            const std::filesystem::path &jawiki_train,
            const std::filesystem::path &conversation_train,
            const BalancedTokenizerConfig &config = {});
        static AgentTokenizer load(const std::filesystem::path &path);
        void save(const std::filesystem::path &path) const;

        std::vector<std::int32_t> encode(std::string_view text) const;
        std::string decode(const std::vector<std::int32_t> &ids) const;

        std::size_t count(std::string_view text) const {return encode(text).size();}
        std::size_t vocabulary_size() const noexcept;
        std::size_t byte_fallback_count(
            const std::vector<std::int32_t> &ids) const;
        std::int32_t mode_id(AgentMode mode) const noexcept {return static_cast<std::int32_t>(mode);}
        std::string fingerprint() const;
        const std::string &training_metadata() const noexcept {
            return _training_metadata;
        }

    private:
        std::shared_ptr<const sentencepiece_detail::Holder> _holder; // SentencePieceモデルの所有者
        std::string _serialized; // シリアライズ済みモデルデータ
        std::string _training_metadata; // 抽出元と再現条件を記録するJSON
};

TokenizerEfficiencyMetrics evaluate_tokenizer(
    const AgentTokenizer &tokenizer,
    const std::filesystem::path &jsonl,
    std::size_t context = 1024);

const char *mode_token(AgentMode mode);

} // namespace ai::model
