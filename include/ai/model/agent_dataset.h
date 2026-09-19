#pragma once

#include "ai/model/agent_tokenizer.h"

#include <filesystem>
#include <string>
#include <vector>

namespace ai::model {

struct AgentWindow {
        std::vector<std::int32_t> inputs; // モデルへ入力するトークンID列
        std::vector<std::int32_t> targets; // 各入力位置の正解トークンID列
        AgentMode mode = AgentMode::Chat; // 学習対象のエージェント動作モード
        std::string document_id; // 元文書を識別するID
        std::size_t valid_tokens = 0; // パディングを除く学習対象トークン数
};

struct SftSourceProvenance {
        std::string name, revision, converter_version;
        bool operator==(const SftSourceProvenance &) const = default;
};

class AgentDataset {
    public:
        static AgentDataset jawiki(const std::filesystem::path &train_jsonl,
                                   const AgentTokenizer &tokenizer,
                                   std::size_t context,
                                   std::size_t token_limit = 0);
        static AgentDataset sft(const std::filesystem::path &train_jsonl,
                                const AgentTokenizer &tokenizer,
                                std::size_t context);

        const std::vector<AgentWindow> &windows() const noexcept {
            return _windows;
        }
        std::vector<std::size_t> order(std::uint64_t seed,
                                       std::size_t epoch) const;
        double padding_ratio(std::size_t batch_size) const;
        std::string fingerprint() const;
        const std::string &sft_profile() const noexcept { return _sft_profile; }
        const std::vector<std::string> &trained_modes() const noexcept {
            return _trained_modes;
        }
        std::uint64_t generation_seed() const noexcept {
            return _generation_seed;
        }
        const std::string &split() const noexcept { return _split; }
        std::size_t input_bytes() const noexcept { return _input_bytes; }
        std::size_t excluded_too_long() const noexcept {
            return _excluded_too_long;
        }

        const std::vector<SftSourceProvenance> &
        source_provenance() const noexcept {
            return _source_provenance;
        }

    private:
        std::vector<AgentWindow> _windows; // 学習用に分割したトークン窓
        std::string _sft_profile,
            _split; // SFT生成プロファイルとデータセット分割名
        std::vector<std::string> _trained_modes; // コーパスに含まれる学習モード
        std::uint64_t _generation_seed =
            0; // コーパス生成時の乱数シード（未記録時は0）
        std::size_t _input_bytes = 0; // loss対象の元UTF-8入力byte数
        std::size_t _excluded_too_long = 0; // context超過で明示除外した件数
        std::vector<SftSourceProvenance> _source_provenance;
};

void generate_sft_corpus(const std::filesystem::path &conversation_train,
                         const std::filesystem::path &output,
                         std::uint64_t seed = 42,
                         std::string_view profile = "agent",
                         std::string_view split = "train");

} // namespace ai::model
