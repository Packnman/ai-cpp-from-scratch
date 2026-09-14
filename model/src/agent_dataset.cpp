#include "ai/model/agent_dataset.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace ai::model {
namespace {
AgentMode parse_mode(std::string_view name) {
    static const std::vector<std::pair<std::string_view, AgentMode>> modes = {
        {"CHAT", AgentMode::Chat},
        {"PARSE", AgentMode::Parse},
        {"PLAN", AgentMode::Plan},
        {"EVALUATE", AgentMode::Evaluate},
        {"SUMMARIZE", AgentMode::Summarize},
        {"MEMORY_WRITE", AgentMode::MemoryWrite},
        {"MEMORY_QUERY", AgentMode::MemoryQuery},
        {"FINAL", AgentMode::Final},
        {"TOOL", AgentMode::Tool}};
    for (const auto &[text, mode] : modes)
        if (text == name)
            return mode;
    throw std::invalid_argument("unknown SFT mode: " + std::string(name));
}
} // namespace

AgentDataset AgentDataset::jawiki(const std::filesystem::path &train_jsonl,
                                  const AgentTokenizer &tokenizer,
                                  std::size_t context,
                                  std::size_t token_limit) {
    if (context == 0)
        throw std::invalid_argument("context must be positive");
    std::ifstream input(train_jsonl);
    if (!input)
        throw std::runtime_error("cannot read Jawiki train split");
    AgentDataset dataset;
    std::size_t accepted = 0;
    std::string line;
    while (std::getline(input, line) &&
           (token_limit == 0 || accepted < token_limit)) {
        const auto json = nlohmann::json::parse(line);
        const auto id = json.at("id").dump();
        auto ids = tokenizer.encode(json.at("text").get<std::string>());
        ids.insert(ids.begin(), AgentTokenizer::bos_id);
        ids.push_back(AgentTokenizer::eos_id);
        for (std::size_t begin = 0; begin + 1 < ids.size();) {
            const auto end = std::min(ids.size(), begin + context + 1);
            AgentWindow window;
            window.document_id = id;
            window.inputs.assign(ids.begin() + begin, ids.begin() + end - 1);
            window.targets.assign(ids.begin() + begin + 1, ids.begin() + end);
            window.valid_tokens = window.targets.size();
            if (token_limit && accepted + window.valid_tokens > token_limit) {
                const auto keep = token_limit - accepted;
                window.inputs.resize(keep);
                window.targets.resize(keep);
                window.valid_tokens = keep;
            }
            if (window.valid_tokens)
                dataset._windows.push_back(std::move(window));
            accepted += dataset._windows.back().valid_tokens;
            begin = end - 1;
            if (token_limit && accepted >= token_limit)
                break;
        }
    }
    if (dataset._windows.empty())
        throw std::invalid_argument("Jawiki split has no trainable tokens");
    return dataset;
}

AgentDataset AgentDataset::sft(const std::filesystem::path &train_jsonl,
                               const AgentTokenizer &tokenizer,
                               std::size_t context) {
    if (context == 0)
        throw std::invalid_argument("context must be positive");
    std::ifstream input(train_jsonl);
    if (!input)
        throw std::runtime_error("cannot read SFT train split");
    AgentDataset dataset;
    std::string line;
    while (std::getline(input, line)) {
        const auto json = nlohmann::json::parse(line);
        const auto mode = parse_mode(json.at("mode").get<std::string>());
        auto prompt = tokenizer.encode(json.at("prompt").get<std::string>());
        auto output = tokenizer.encode(json.at("output").get<std::string>());
        std::vector<std::int32_t> ids = {AgentTokenizer::bos_id,
                                         tokenizer.mode_id(mode)};
        ids.insert(ids.end(), prompt.begin(), prompt.end());
        const auto response_begin = ids.size();
        ids.insert(ids.end(), output.begin(), output.end());
        ids.push_back(AgentTokenizer::eos_id);
        if (ids.size() > context + 1)
            throw std::length_error("SFT example exceeds context; "
                                    "prompt/answer truncation is forbidden");
        AgentWindow window;
        window.mode = mode;
        window.document_id =
            json.value("id", std::to_string(dataset._windows.size()));
        window.inputs.assign(ids.begin(), ids.end() - 1);
        window.targets.assign(ids.size() - 1, AgentTokenizer::pad_id);
        for (std::size_t next = response_begin; next < ids.size(); ++next)
            window.targets[next - 1] = ids[next];
        window.valid_tokens = ids.size() - response_begin;
        dataset._windows.push_back(std::move(window));
    }
    if (dataset._windows.empty())
        throw std::invalid_argument("SFT split is empty");
    std::array<bool, 9> covered{};
    for (const auto &window : dataset._windows)
        covered[static_cast<std::size_t>(window.mode) - 4] = true;
    if (!std::all_of(covered.begin(), covered.end(),
                     [](bool value) { return value; }))
        throw std::invalid_argument("SFT split must cover all nine modes");
    return dataset;
}

std::vector<std::size_t> AgentDataset::order(std::uint64_t seed,
                                             std::size_t epoch) const {
    std::array<std::vector<std::size_t>, 9> by_mode;
    for (std::size_t index = 0; index < _windows.size(); ++index)
        by_mode[static_cast<std::size_t>(_windows[index].mode) - 4].push_back(
            index);
    const bool is_sft =
        std::all_of(by_mode.begin(), by_mode.end(),
                    [](const auto &items) { return !items.empty(); });
    std::mt19937 random(static_cast<std::mt19937::result_type>(seed + epoch));
    if (is_sft) {
        static constexpr std::array<std::size_t, 9> weights = {
            20, 15, 15, 10, 10, 10, 5, 10, 5};
        std::vector<std::size_t> result;
        result.reserve(100);
        for (std::size_t mode = 0; mode < weights.size(); ++mode) {
            std::shuffle(by_mode[mode].begin(), by_mode[mode].end(), random);
            for (std::size_t count = 0; count < weights[mode]; ++count)
                result.push_back(by_mode[mode][count % by_mode[mode].size()]);
        }
        std::shuffle(result.begin(), result.end(), random);
        return result;
    }
    std::vector<std::size_t> result(_windows.size());
    std::iota(result.begin(), result.end(), 0);
    std::shuffle(result.begin(), result.end(), random);
    return result;
}

double AgentDataset::padding_ratio(std::size_t batch_size) const {
    if (batch_size == 0)
        throw std::invalid_argument("batch size must be positive");
    std::size_t padded = 0, total = 0;
    for (std::size_t start = 0; start < _windows.size(); start += batch_size) {
        const auto end = std::min(_windows.size(), start + batch_size);
        std::size_t longest = 0, used = 0;
        for (std::size_t i = start; i < end; ++i) {
            longest = std::max(longest, _windows[i].inputs.size());
            used += _windows[i].inputs.size();
        }
        total += longest * (end - start);
        padded += longest * (end - start) - used;
    }
    return total ? static_cast<double>(padded) / total : 0.0;
}

std::string AgentDataset::fingerprint() const {
    std::uint64_t hash = 1469598103934665603ULL;
    auto add = [&](std::uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (8 * byte)) & 0xffU;
            hash *= 1099511628211ULL;
        }
    };
    for (const auto &window : _windows) {
        add(static_cast<std::uint64_t>(window.mode));
        for (const auto id : window.inputs)
            add(static_cast<std::uint32_t>(id));
        add(0xffffffffffffffffULL);
        for (const auto id : window.targets)
            add(static_cast<std::uint32_t>(id));
    }
    std::ostringstream result;
    result << std::hex << hash;
    return result.str();
}

void generate_sft_corpus(const std::filesystem::path &conversation_train,
                         const std::filesystem::path &output,
                         std::uint64_t seed) {
    std::ifstream conversations(conversation_train);
    if (!conversations)
        throw std::runtime_error("cannot read conversation train split");
    std::ofstream corpus(output, std::ios::trunc);
    if (!corpus)
        throw std::runtime_error("cannot write SFT corpus");
    std::string line;
    std::size_t id = 0;
    while (std::getline(conversations, line)) {
        const auto json = nlohmann::json::parse(line);
        const auto &utterances = json.at("utterances");
        for (std::size_t i = 1; i < utterances.size(); ++i)
            if (utterances[i - 1].at("speaker") == 0 &&
                utterances[i].at("speaker") == 1)
                corpus << nlohmann::json{{"id", "chat-" + std::to_string(id++)},
                                         {"mode", "CHAT"},
                                         {"prompt",
                                          utterances[i - 1].at("text")},
                                         {"output", utterances[i].at("text")}}
                              .dump()
                       << '\n';
    }
    std::mt19937 random(static_cast<std::mt19937::result_type>(seed));
    const auto nonce = std::to_string(random());
    const std::vector<nlohmann::json> structured = {
        {{"mode", "PARSE"},
         {"prompt", "赤い箱を安全に運んで"},
         {"output",
          R"({"raw":"赤い箱を安全に運んで","intent":"robot","goal":"箱を運ぶ","constraints":[{"text":"安全","critical":true}]})"}},
        {{"mode", "PLAN"},
         {"prompt", "資料を読んで要約する"},
         {"output",
          R"({"goal":"要約","tasks":[{"id":"read","type":"tool","operation":"file.read","arguments":{"path":"資料.txt"},"depends_on":[]},{"id":"summary","type":"reasoning","operation":"summarize","arguments":{},"depends_on":["read"]}]})"}},
        {{"mode", "EVALUATE"},
         {"prompt", "tool succeeded"},
         {"output", R"({"status":"success","reason":"結果を取得した"})"}},
        {{"mode", "SUMMARIZE"},
         {"prompt", "会話を圧縮"},
         {"output", "設計方針と未解決事項を保持する。"}},
        {{"mode", "MEMORY_WRITE"},
         {"prompt", "C++20を使う"},
         {"output",
          R"({"memories":[{"type":"project","content":"C++20を使う","importance":1.0,"confidence":1.0}]})"}},
        {{"mode", "MEMORY_QUERY"},
         {"prompt", "言語規格"},
         {"output", R"({"query":"C++ 言語規格","limit":8})"}},
        {{"mode", "FINAL"},
         {"prompt", "完了"},
         {"output", "処理が完了しました。"}},
        {{"mode", "TOOL"},
         {"prompt", "2+3を計算"},
         {"output",
          R"({"operation":"calculator.calculate","arguments":{"expression":"2+3"}})"}}};
    for (const auto &example : structured) {
        auto value = example;
        value["id"] = "generated-" + nonce + "-" + std::to_string(id++);
        corpus << value.dump() << '\n';
    }
}
} // namespace ai::model
