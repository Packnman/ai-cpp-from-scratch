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

class AgentTokenizer {
    public:
        static constexpr std::int32_t pad_id = 0;
        static constexpr std::int32_t unk_id = 1;
        static constexpr std::int32_t bos_id = 2;
        static constexpr std::int32_t eos_id = 3;
        static constexpr std::int32_t special_count = 13;

        AgentTokenizer(
            std::shared_ptr<const sentencepiece_detail::Holder> holder,
            std::string serialized);
        static AgentTokenizer train(const std::filesystem::path &train_jsonl,
                                    std::size_t vocabulary_size = 8192);
        static AgentTokenizer load(const std::filesystem::path &path);
        void save(const std::filesystem::path &path) const;

        std::vector<std::int32_t> encode(std::string_view text) const;
        std::string decode(const std::vector<std::int32_t> &ids) const;
        std::size_t count(std::string_view text) const {
            return encode(text).size();
        }
        std::size_t vocabulary_size() const noexcept;
        std::int32_t mode_id(AgentMode mode) const noexcept {
            return static_cast<std::int32_t>(mode);
        }
        std::string fingerprint() const;

    private:
        std::shared_ptr<const sentencepiece_detail::Holder> _holder;
        std::string _serialized;
};

const char *mode_token(AgentMode mode);

} // namespace ai::model
