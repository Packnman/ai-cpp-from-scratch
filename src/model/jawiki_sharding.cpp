#include "ai/model/jawiki_sharding.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ai::model {
namespace {
class Hash {
    public:
        void add(std::string_view bytes) {
            for (const unsigned char byte : bytes) {
                value ^= byte;
                value *= 1099511628211ULL;
            }
        }
        std::string text() const {
            std::ostringstream out;
            out << std::hex << std::setw(16) << std::setfill('0') << value;
            return out.str();
        }

    private:
        std::uint64_t value = 1469598103934665603ULL;
};

std::string file_fingerprint(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read file for fingerprint: " +
                                 path.string());
    Hash hash;
    char buffer[1 << 16];
    while (input) {
        input.read(buffer, sizeof(buffer));
        hash.add(
            std::string_view(buffer, static_cast<std::size_t>(input.gcount())));
    }
    return hash.text();
}

std::string shard_name(std::size_t index) {
    std::ostringstream name;
    name << "shard-" << std::setw(5) << std::setfill('0') << index << ".jsonl";
    return name.str();
}

void validate_existing(const std::filesystem::path &source,
                       const std::filesystem::path &output,
                       std::size_t target_bytes) {
    std::ifstream input(output / "manifest.json");
    nlohmann::json manifest;
    if (!input || !(input >> manifest))
        throw std::runtime_error("shard output exists without a readable "
                                 "manifest; refusing to overwrite");
    const auto recorded_fingerprint =
        manifest.at("manifest_fingerprint").get<std::string>();
    auto fingerprint_payload = manifest;
    fingerprint_payload.erase("manifest_fingerprint");
    Hash manifest_hash;
    manifest_hash.add(fingerprint_payload.dump());
    if (manifest.value("format", "") != "ai_cpp_jawiki_shards" ||
        manifest.value("version", 0) != 1 ||
        recorded_fingerprint != manifest_hash.text() ||
        manifest.value("target_shard_bytes", 0ULL) != target_bytes ||
        manifest.at("source").at("path") !=
            std::filesystem::absolute(source).lexically_normal().string() ||
        manifest.at("source").at("bytes") !=
            std::filesystem::file_size(source) ||
        manifest.at("source").at("fingerprint") != file_fingerprint(source))
        throw std::runtime_error("existing shard manifest does not match the "
                                 "source; refusing to overwrite");
    std::size_t documents = 0;
    std::uintmax_t bytes = 0;
    for (std::size_t i = 0; i < manifest.at("shards").size(); ++i) {
        const auto &entry = manifest.at("shards").at(i);
        const auto relative =
            std::filesystem::path(entry.at("file").get<std::string>());
        const auto path = output / relative;
        if (entry.at("index") != i || relative.is_absolute() ||
            relative.has_parent_path() ||
            !std::filesystem::is_regular_file(path) ||
            std::filesystem::file_size(path) != entry.at("bytes") ||
            file_fingerprint(path) !=
                entry.at("fingerprint").get<std::string>())
            throw std::runtime_error(
                "existing shard is missing or modified; refusing to overwrite");
        documents += entry.at("documents").get<std::size_t>();
        bytes += entry.at("bytes").get<std::uintmax_t>();
    }
    if (documents != manifest.at("source").at("documents") ||
        bytes != manifest.at("source").at("bytes"))
        throw std::runtime_error("existing shard coverage is inconsistent");
    std::cout << manifest.dump() << '\n';
}
} // namespace

void shard_jawiki(const std::filesystem::path &source,
                  const std::filesystem::path &output,
                  std::size_t target_bytes) {
    if (!target_bytes)
        throw std::invalid_argument("shard byte size must be positive");
    if (!std::filesystem::is_regular_file(source))
        throw std::runtime_error("cannot read Jawiki source");
    if (std::filesystem::exists(output)) {
        if (std::filesystem::exists(output / "manifest.json")) {
            validate_existing(source, output, target_bytes);
            return;
        }
        if (!std::filesystem::is_directory(output) ||
            std::filesystem::directory_iterator(output) !=
                std::filesystem::directory_iterator())
            throw std::runtime_error(
                "shard output is not empty; refusing to overwrite");
    }
    std::filesystem::create_directories(output);
    std::ifstream input(source, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read Jawiki source");

    nlohmann::json shards = nlohmann::json::array();
    Hash source_hash;
    std::size_t total_documents = 0, shard_documents = 0, index = 0;
    std::uintmax_t total_bytes = 0, shard_bytes = 0;
    Hash shard_hash;
    std::ofstream shard;
    std::filesystem::path temporary;
    const auto open_shard = [&] {
        temporary = output / (shard_name(index) + ".tmp");
        shard.open(temporary, std::ios::binary | std::ios::trunc);
        if (!shard)
            throw std::runtime_error("cannot create Jawiki shard");
    };
    const auto close_shard = [&] {
        shard.close();
        if (!shard)
            throw std::runtime_error("cannot write Jawiki shard");
        const auto name = shard_name(index);
        std::filesystem::rename(temporary, output / name);
        shards.push_back({{"index", index},
                          {"file", name},
                          {"documents", shard_documents},
                          {"bytes", shard_bytes},
                          {"fingerprint", shard_hash.text()}});
        ++index;
        shard_documents = 0;
        shard_bytes = 0;
        shard_hash = Hash{};
    };

    std::string line;
    while (std::getline(input, line)) {
        const bool had_newline = !input.eof();
        if (line.empty())
            throw std::runtime_error(
                "Jawiki source contains an empty JSONL row");
        const auto parsed = nlohmann::json::parse(line);
        if (!parsed.is_object() || !parsed.contains("id") ||
            !parsed.contains("text") || !parsed.at("text").is_string())
            throw std::runtime_error("invalid Jawiki JSONL row");
        std::string record = line;
        if (had_newline)
            record.push_back('\n');
        if (shard.is_open() && shard_bytes &&
            shard_bytes + record.size() > target_bytes)
            close_shard();
        if (!shard.is_open())
            open_shard();
        shard.write(record.data(), static_cast<std::streamsize>(record.size()));
        source_hash.add(record);
        shard_hash.add(record);
        ++total_documents;
        ++shard_documents;
        total_bytes += record.size();
        shard_bytes += record.size();
    }
    if (shard.is_open())
        close_shard();
    if (!total_documents)
        throw std::invalid_argument("Jawiki source is empty");

    nlohmann::json manifest = {
        {"format", "ai_cpp_jawiki_shards"},
        {"version", 1},
        {"target_shard_bytes", target_bytes},
        {"source",
         {{"path",
           std::filesystem::absolute(source).lexically_normal().string()},
          {"documents", total_documents},
          {"bytes", total_bytes},
          {"fingerprint", source_hash.text()}}},
        {"shards", std::move(shards)}};
    Hash manifest_hash;
    manifest_hash.add(manifest.dump());
    manifest["manifest_fingerprint"] = manifest_hash.text();
    const auto temporary_manifest = output / "manifest.json.tmp";
    {
        std::ofstream out(temporary_manifest, std::ios::trunc);
        out << manifest.dump(2) << '\n';
        if (!out)
            throw std::runtime_error("cannot write shard manifest");
    }
    std::filesystem::rename(temporary_manifest, output / "manifest.json");
    std::cout << manifest.dump() << '\n';
}

} // namespace ai::model
