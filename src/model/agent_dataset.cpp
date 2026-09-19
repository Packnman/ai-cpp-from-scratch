#include "ai/model/agent_dataset.h"
#include "ai/agent/context_builder.h"

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
        const auto text = json.at("text").get<std::string>();
        auto ids = tokenizer.encode(text);
        const auto content_tokens = ids.size();
        const auto accepted_before_document = accepted;
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
        const auto accepted_targets = accepted - accepted_before_document;
        const auto accepted_content =
            std::min(content_tokens, accepted_targets);
        dataset._input_bytes +=
            content_tokens ? text.size() * accepted_content / content_tokens
                           : 0;
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
        const auto profile = json.value("sft_profile", "legacy");
        const auto split = json.value("split", "");
        if (dataset._sft_profile.empty()) {
            dataset._sft_profile = profile;
            dataset._split = split;
            dataset._generation_seed = json.value("generation_seed", 0ULL);
        } else if (dataset._sft_profile != profile || dataset._split != split)
            throw std::invalid_argument("mixed SFT profile or split");
        const SftSourceProvenance provenance{
            json.value("source_name", "unrecorded"),
            json.value("source_revision", "unrecorded"),
            json.value("converter_version", "unrecorded")};
        if (std::find(dataset._source_provenance.begin(),
                      dataset._source_provenance.end(),
                      provenance) == dataset._source_provenance.end())
            dataset._source_provenance.push_back(provenance);
        auto prompt = tokenizer.encode(json.at("prompt").get<std::string>());
        const auto output_text = json.at("output").get<std::string>();
        auto output = tokenizer.encode(output_text);
        std::vector<std::int32_t> ids = {AgentTokenizer::bos_id,
                                         tokenizer.mode_id(mode)};
        ids.insert(ids.end(), prompt.begin(), prompt.end());
        const auto response_begin = ids.size();
        ids.insert(ids.end(), output.begin(), output.end());
        ids.push_back(AgentTokenizer::eos_id);
        if (ids.size() > context + 1) {
            ++dataset._excluded_too_long;
            continue;
        }
        dataset._input_bytes += output_text.size();
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
    static const std::array<const char *, 9> names = {
        "CHAT",         "PARSE",        "PLAN",  "EVALUATE", "SUMMARIZE",
        "MEMORY_WRITE", "MEMORY_QUERY", "FINAL", "TOOL"};
    for (const auto &window : dataset._windows)
        covered[static_cast<std::size_t>(window.mode) - 4] = true;
    for (std::size_t i = 0; i < covered.size(); ++i)
        if (covered[i])
            dataset._trained_modes.push_back(names[i]);
    if (dataset._sft_profile == "chat" &&
        (dataset._trained_modes != std::vector<std::string>{"CHAT"}))
        throw std::invalid_argument("chat SFT must contain CHAT only");
    const std::array<bool, 9> agent_required = {true, true,  true, true, true,
                                                true, false, true, false};
    if (dataset._sft_profile == "agent")
        for (std::size_t i = 0; i < covered.size(); ++i)
            if (covered[i] != agent_required[i])
                throw std::invalid_argument(
                    "agent SFT has missing or unused modes");
    if (dataset._sft_profile == "legacy" &&
        !std::all_of(covered.begin(), covered.end(), [](bool v) { return v; }))
        throw std::invalid_argument(
            "legacy SFT split must cover all nine modes");
    return dataset;
}

std::vector<std::size_t> AgentDataset::order(std::uint64_t seed,
                                             std::size_t epoch) const {
    std::array<std::vector<std::size_t>, 9> by_mode;
    for (std::size_t index = 0; index < _windows.size(); ++index)
        by_mode[static_cast<std::size_t>(_windows[index].mode) - 4].push_back(
            index);
    const bool is_sft = _sft_profile == "agent" || _sft_profile == "legacy";
    std::mt19937 random(static_cast<std::mt19937::result_type>(seed + epoch));
    if (is_sft) {
        static constexpr std::array<std::size_t, 9> agent_weights = {
            20, 15, 15, 10, 20, 10, 0, 10, 0};
        static constexpr std::array<std::size_t, 9> legacy_weights = {
            20, 15, 15, 10, 10, 10, 5, 10, 5};
        const auto &weights =
            _sft_profile == "agent" ? agent_weights : legacy_weights;
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
                         std::uint64_t seed, std::string_view profile,
                         std::string_view split) {
    if (profile != "chat" && profile != "agent")
        throw std::invalid_argument("--profile must be chat or agent");
    if (split != "train" && split != "validation")
        throw std::invalid_argument("--split must be train or validation");
    std::ifstream conversations(conversation_train);
    if (!conversations)
        throw std::runtime_error("cannot read conversation split");
    std::ofstream corpus(output, std::ios::trunc);
    if (!corpus)
        throw std::runtime_error("cannot write SFT corpus");
    std::size_t id = 0;
    const auto emit = [&](std::string_view mode, const std::string &prompt,
                          const std::string &answer, std::string example_id) {
        const bool chat = mode == "CHAT";
        corpus << nlohmann::json{{"id", example_id},
                                 {"mode", mode},
                                 {"prompt", prompt},
                                 {"output", answer},
                                 {"sft_profile", profile},
                                 {"split", split},
                                 {"generation_seed", seed},
                                 {"generator", "local-template-v3"},
                                 {"source_name",
                                  chat ? "conversation-input"
                                       : "deterministic-synthetic"},
                                 {"source_id", example_id},
                                 {"source_revision",
                                  chat ? "user-provided" : "template-v3"},
                                 {"converter_version", "summary-sft-v3"}}
                      .dump()
               << '\n';
    };
    std::string line;
    while (std::getline(conversations, line)) {
        const auto json = nlohmann::json::parse(line);
        const auto &utterances = json.at("utterances");
        for (std::size_t i = 1; i < utterances.size(); ++i) {
            if (utterances[i - 1].at("speaker") != 0 ||
                utterances[i].at("speaker") != 1)
                continue;
            const auto raw = utterances[i - 1].at("text").get<std::string>();
            ai::agent::ContextInput context;
            context.current_input = raw;
            context.goal = raw;
            context.current_task = "Natural Japanese conversation";
            emit("CHAT", ai::agent::build_model_prompt(context),
                 utterances[i].at("text"),
                 std::string(split) + "-chat-" + std::to_string(id++));
        }
    }
    if (profile == "chat")
        return;
    const std::size_t count = split == "train" ? 2000 : 200;
    const std::string tag = split == "train" ? "訓練対象" : "検証項目";
    const std::array<std::string, 4> nouns =
        split == "train" ? std::array<std::string, 4>{"設計書", "青い箱",
                                                      "計算式", "会議メモ"}
                         : std::array<std::string, 4>{"仕様票", "緑の容器",
                                                      "数式", "審査記録"};
    constexpr std::string_view parse_schema =
        R"({"required":["raw","intent","goal","constraints"]})";
    constexpr std::string_view plan_schema = R"({"required":["goal","tasks"]})";
    constexpr std::string_view eval_schema =
        R"({"required":["status","reason"]})";
    constexpr std::string_view memory_schema = R"({"required":["memories"]})";
    for (std::size_t i = 0; i < count; ++i) {
        const auto serial = std::string(split) + "-" + std::to_string(seed) +
                            "-" + std::to_string(i);
        const auto noun = nouns[i % nouns.size()];
        const auto intent = std::array<const char *, 5>{
            "calculate", "read", "recall", "remember", "chat"}[i % 5];
        const std::array<std::string, 4> train_phrases = {
            "を処理して", "を安全に確認して", "の対応を計画して", "を扱って"};
        const std::array<std::string, 4> validation_phrases = {
            "について調べてください", "の検証をお願いします", "を確認願います",
            "について報告してください"};
        const auto &phrases =
            split == "train" ? train_phrases : validation_phrases;
        std::string raw = noun + serial + phrases[i % phrases.size()];
        nlohmann::json arguments = nlohmann::json::object();
        if (std::string_view(intent) == "calculate")
            arguments = {{"expression", std::to_string(i + 2) + "+3"}};
        if (std::string_view(intent) == "read")
            arguments = {{"path", noun + serial + ".txt"}};
        if (std::string_view(intent) == "recall")
            arguments = {{"query", noun + serial}, {"limit", 8}};
        if (std::string_view(intent) == "remember")
            arguments = {{"type", "project"},
                         {"content", noun + serial + "を採用"}};
        nlohmann::json parsed = {
            {"raw", raw},
            {"intent", intent},
            {"goal", noun + "を処理する"},
            {"constraints",
             nlohmann::json::array({{{"text", "安全に処理する" + serial},
                                     {"critical", i % 2 == 0}}})},
            {"arguments", arguments}};
        auto pc = ai::agent::structured_prompt_input(
            ai::agent::ModelMode::Parse, raw, parse_schema);
        emit("PARSE", ai::agent::build_model_prompt(pc), parsed.dump(),
             "parse-" + serial);

        const auto operation = std::array<const char *, 4>{
            "calculator.calculate", "file.read", "memory.retrieve",
            "rule-based reasoning"}[i % 4];
        nlohmann::json task_args =
            i % 4 == 0
                ? nlohmann::json{{"expression", std::to_string(i + 1) + "*2"}}
            : i % 4 == 1 ? nlohmann::json{{"path", noun + serial + ".txt"}}
            : i % 4 == 2
                ? nlohmann::json{{"query", noun + serial}, {"limit", 8}}
                : nlohmann::json::object();
        nlohmann::json plan_payload = {{"input", raw}, {"memories", i % 3}};
        if (i % 5 == 0) {
            plan_payload["failed_result"] = tag + serial + "の空結果";
            plan_payload["reason"] = "別経路で再計画";
        }
        nlohmann::json planned = {
            {"goal", noun + "を完了する"},
            {"tasks", nlohmann::json::array(
                          {{{"id", "task-" + serial},
                            {"type", i % 4 == 3 ? "reasoning" : "tool"},
                            {"operation", operation},
                            {"arguments", task_args},
                            {"depends_on", nlohmann::json::array()}}})}};
        auto plc = ai::agent::structured_prompt_input(
            ai::agent::ModelMode::Plan, plan_payload.dump(), plan_schema);
        emit("PLAN", ai::agent::build_model_prompt(plc), planned.dump(),
             "plan-" + serial);

        const auto eval_status = std::array<const char *, 4>{
            "success", "retry", "replan", "failed"}[i % 4];
        nlohmann::json eval_payload = {
            {"task", "task-" + serial},
            {"status", static_cast<int>(i % 4)},
            {"value", i % 4 == 0 ? nlohmann::json{{"result", serial}}
                                 : nlohmann::json::object()},
            {"error", i % 4 == 0 ? "" : tag + serial + "の失敗"}};
        auto ec = ai::agent::structured_prompt_input(
            ai::agent::ModelMode::Evaluate, eval_payload.dump(), eval_schema);
        emit("EVALUATE", ai::agent::build_model_prompt(ec),
             nlohmann::json{{"status", eval_status},
                            {"reason", tag + serial + "を評価"}}
                 .dump(),
             "evaluate-" + serial);

        const std::array<std::string, 8> summary_cases = {
            noun + "は" + std::to_string(i + 1) + "個",
            noun + "は削除しない",
            noun + "は旧案ではなく新案へ訂正",
            noun + "は新案に決定",
            noun + "の期限は明日17時",
            noun + "の担当者は未解決？",
            noun + "の確認タスクは完了",
            noun + "の確認タスクは取消"};
        const auto summary_case = summary_cases[i % summary_cases.size()];
        const auto repeated = std::array<std::size_t, 3>{1, 3, 5}[i % 3];
        ai::agent::ContextInput summary;
        summary.current_input = "Summarize the conversation";
        summary.goal = "Preserve decisions, open questions, and current intent";
        summary.current_task = "Conversation summary";
        for (std::size_t update = 0; update < repeated; ++update)
            summary.recent.push_back(
                {tag + serial + "について相談: " + summary_case,
                 noun + "を確認"});
        summary.summary = noun + "の以前の方針";
        const auto decision = i % 8 == 2 || i % 8 == 3 ? summary_case : "なし";
        const auto constraint =
            i % 8 == 1 || i % 8 == 4 ? summary_case : "なし";
        const auto question = i % 8 == 5 ? summary_case : "なし";
        const auto task = i % 8 >= 6 ? summary_case : "なし";
        const auto correction = i % 8 == 2 ? summary_case : "なし";
        emit("SUMMARIZE", ai::agent::build_model_prompt(summary),
             "[FACTS]\n- " + summary_case + "\n[DECISIONS]\n- " + decision +
                 "\n[CONSTRAINTS]\n- " + constraint + "\n[OPEN_QUESTIONS]\n- " +
                 question + "\n[ACTIVE_TASKS]\n- " + task +
                 "\n[CORRECTIONS]\n- " + correction,
             "summarize-" + serial);

        nlohmann::json memory_payload = {
            {"input", raw}, {"response", noun + serial + "を記憶します"}};
        nlohmann::json memories =
            i % 3 == 0 ? nlohmann::json::array()
                       : nlohmann::json::array(
                             {{{"type", i % 2 ? "project" : "semantic"},
                               {"content", noun + serial + "を採用"},
                               {"importance", 0.8},
                               {"confidence", 0.9}}});
        auto mc = ai::agent::structured_prompt_input(
            ai::agent::ModelMode::MemoryWrite, memory_payload.dump(),
            memory_schema);
        emit("MEMORY_WRITE", ai::agent::build_model_prompt(mc),
             nlohmann::json{{"memories", memories}}.dump(), "memory-" + serial);

        ai::agent::ContextInput final;
        final.current_input = raw;
        final.goal = noun + "を処理する";
        final.current_task =
            "Produce the final response from: " +
            nlohmann::json{
                {"results", nlohmann::json::array({{{"value", serial}}})}}
                .dump();
        emit("FINAL", ai::agent::build_model_prompt(final),
             noun + serial + "の処理が完了しました。", "final-" + serial);
    }
}
} // namespace ai::model
