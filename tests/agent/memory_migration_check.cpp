#include "ai/agent/memory.h"
#include <filesystem>
#include <iostream>
#include <sqlite3.h>
#include <stdexcept>
using namespace ai::agent;
#define CHECK(x)                                                               \
    do {                                                                       \
        if (!(x))                                                              \
            throw std::runtime_error(std::string("check failed: ") + #x);      \
    } while (false)
int main() {
    try {
        auto dir =
            std::filesystem::temp_directory_path() / "ai_cpp_migration_check";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        auto future = dir / "future.sqlite";
        sqlite3 *raw = nullptr;
        CHECK(sqlite3_open(future.string().c_str(), &raw) == SQLITE_OK);
        CHECK(sqlite3_exec(raw, "PRAGMA user_version=2", nullptr, nullptr,
                           nullptr) == SQLITE_OK);
        sqlite3_close(raw);
        bool rejected = false;
        try {
            SqliteMemory m(future);
        } catch (...) {
            rejected = true;
        }
        CHECK(rejected);
        auto current = dir / "current.sqlite";
        {
            SqliteMemory m(current);
            m.store({MemoryType::Semantic, "共通検索", .6, .9});
            m.store({MemoryType::Project, "共通検索", .95, .8});
            auto rows = m.retrieve("共通検索");
            CHECK(rows.size() == 2);
            CHECK(rows[0].importance > .9);
            auto filtered = m.retrieve("共通検索", MemoryType::Semantic);
            CHECK(filtered.size() == 1 &&
                  filtered[0].type == MemoryType::Semantic);
        }
        CHECK(sqlite3_open(current.string().c_str(), &raw) == SQLITE_OK);
        sqlite3_stmt *s = nullptr;
        CHECK(sqlite3_prepare_v2(raw, "PRAGMA user_version", -1, &s, nullptr) ==
              SQLITE_OK);
        CHECK(sqlite3_step(s) == SQLITE_ROW && sqlite3_column_int(s, 0) == 1);
        sqlite3_finalize(s);
        sqlite3_close(raw);
        std::filesystem::remove_all(dir);
        std::cout << "memory_migration_check passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
