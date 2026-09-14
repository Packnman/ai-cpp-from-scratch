#include "ai/agent/memory.h"

#include <algorithm>
#include <sqlite3.h>
#include <stdexcept>

namespace ai::agent {
namespace {
void check(int rc, sqlite3 *db, std::string_view what) {
    if (rc != SQLITE_OK && rc != SQLITE_DONE && rc != SQLITE_ROW)
        throw std::runtime_error(std::string(what) + ": " + sqlite3_errmsg(db));
}
void exec(sqlite3 *db, const char *sql) {
    char *error = nullptr;
    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &error);
    if (rc != SQLITE_OK) {
        std::string msg = error ? error : sqlite3_errmsg(db);
        sqlite3_free(error);
        throw std::runtime_error(msg);
    }
}
std::size_t codepoints(std::string_view s) {
    std::size_t n = 0;
    for (unsigned char c : s)
        if ((c & 0xc0) != 0x80)
            ++n;
    return n;
}
MemoryRecord row(sqlite3_stmt *s) {
    MemoryRecord r;
    r.id = sqlite3_column_int64(s, 0);
    r.type = *memory_type_from_string(
        reinterpret_cast<const char *>(sqlite3_column_text(s, 1)));
    r.content = reinterpret_cast<const char *>(sqlite3_column_text(s, 2));
    r.importance = sqlite3_column_double(s, 3);
    r.confidence = sqlite3_column_double(s, 4);
    r.created_at = sqlite3_column_int64(s, 5);
    r.updated_at = sqlite3_column_int64(s, 6);
    r.last_accessed_at = sqlite3_column_int64(s, 7);
    return r;
}
} // namespace

SqliteMemory::SqliteMemory(const std::filesystem::path &path) {
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    int rc = sqlite3_open_v2(path.string().c_str(), &_db,
                             SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                                 SQLITE_OPEN_FULLMUTEX,
                             nullptr);
    if (rc != SQLITE_OK) {
        std::string msg = _db ? sqlite3_errmsg(_db) : "open failed";
        if (_db)
            sqlite3_close(_db);
        _db = nullptr;
        throw std::runtime_error(msg);
    }
    try {
        sqlite3_busy_timeout(_db, 5000);
        exec(_db, "PRAGMA journal_mode=WAL;");
        exec(_db, "PRAGMA foreign_keys=ON;");
        migrate();
    } catch (...) {
        sqlite3_close(_db);
        _db = nullptr;
        throw;
    }
}
SqliteMemory::~SqliteMemory() {
    if (_db)
        sqlite3_close(_db);
}
void SqliteMemory::migrate() {
    sqlite3_stmt *s = nullptr;
    check(sqlite3_prepare_v2(_db, "PRAGMA user_version", -1, &s, nullptr), _db,
          "read schema version");
    int rc = sqlite3_step(s);
    check(rc, _db, "read schema version");
    int version = sqlite3_column_int(s, 0);
    sqlite3_finalize(s);
    if (version > 1)
        throw std::runtime_error(
            "memory schema is newer than this application");
    if (version == 1)
        return;
    exec(_db, R"SQL(BEGIN IMMEDIATE;
CREATE TABLE memories(id INTEGER PRIMARY KEY, type TEXT NOT NULL CHECK(type IN('semantic','episodic','project')), content TEXT NOT NULL,
 importance REAL NOT NULL, confidence REAL NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, last_accessed_at INTEGER NOT NULL,
 UNIQUE(type,content));
CREATE VIRTUAL TABLE memories_fts USING fts5(content, content='memories', content_rowid='id', tokenize='trigram');
CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN INSERT INTO memories_fts(rowid,content) VALUES(new.id,new.content); END;
CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN INSERT INTO memories_fts(memories_fts,rowid,content) VALUES('delete',old.id,old.content); END;
CREATE TRIGGER memories_au AFTER UPDATE OF content ON memories BEGIN
 INSERT INTO memories_fts(memories_fts,rowid,content) VALUES('delete',old.id,old.content); INSERT INTO memories_fts(rowid,content) VALUES(new.id,new.content); END;
PRAGMA user_version=1; COMMIT;)SQL");
}
void SqliteMemory::store_unchecked(const MemoryCandidate &c) {
    if (c.content.empty() || c.content.size() > 16 * 1024 || c.importance < 0 ||
        c.importance > 1 || c.confidence < 0 || c.confidence > 1)
        throw std::invalid_argument("invalid memory candidate");
    sqlite3_stmt *s = nullptr;
    const char *sql =
        R"SQL(INSERT INTO memories(type,content,importance,confidence,created_at,updated_at,last_accessed_at)
VALUES(?1,?2,?3,?4,unixepoch(),unixepoch(),unixepoch()) ON CONFLICT(type,content) DO UPDATE SET importance=max(importance,excluded.importance),
confidence=max(confidence,excluded.confidence),updated_at=unixepoch())SQL";
    check(sqlite3_prepare_v2(_db, sql, -1, &s, nullptr), _db,
          "prepare memory upsert");
    auto type = to_string(c.type);
    sqlite3_bind_text(s, 1, type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(s, 2, c.content.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(s, 3, c.importance);
    sqlite3_bind_double(s, 4, c.confidence);
    int rc = sqlite3_step(s);
    sqlite3_finalize(s);
    check(rc, _db, "memory upsert");
}
void SqliteMemory::store(const MemoryCandidate &c) { store_batch({c}); }
void SqliteMemory::store_batch(const std::vector<MemoryCandidate> &cs) {
    exec(_db, "BEGIN IMMEDIATE;");
    try {
        for (const auto &c : cs)
            store_unchecked(c);
        exec(_db, "COMMIT;");
    } catch (...) {
        try {
            exec(_db, "ROLLBACK;");
        } catch (...) {
        }
        throw;
    }
}
std::vector<MemoryRecord> SqliteMemory::retrieve(std::string_view query,
                                                 std::optional<MemoryType> type,
                                                 std::size_t limit) {
    limit = std::min<std::size_t>(limit, 8);
    if (query.empty() || limit == 0)
        return {};
    sqlite3_stmt *s = nullptr;
    std::string sql;
    if (codepoints(query) < 3)
        sql = "SELECT "
              "id,type,content,importance,confidence,created_at,updated_at,"
              "last_accessed_at FROM memories WHERE (content=?1 OR "
              "instr(content,?1)>0) AND (?2 IS NULL OR type=?2) ORDER BY CASE "
              "WHEN content=?1 THEN 0 ELSE 1 END,importance DESC,confidence "
              "DESC,id DESC LIMIT ?3";
    else
        sql = "SELECT "
              "m.id,m.type,m.content,m.importance,m.confidence,m.created_at,m."
              "updated_at,m.last_accessed_at FROM memories_fts JOIN memories m "
              "ON m.id=memories_fts.rowid WHERE memories_fts MATCH ?1 AND (?2 "
              "IS NULL OR m.type=?2) ORDER BY bm25(memories_fts),m.importance "
              "DESC,m.confidence DESC,m.id DESC LIMIT ?3";
    check(sqlite3_prepare_v2(_db, sql.c_str(), -1, &s, nullptr), _db,
          "prepare memory query");
    std::string bound(query);
    if (codepoints(query) >= 3) {
        std::string quoted = "\"";
        for (char c : query) {
            if (c == '\"')
                quoted += "\"\"";
            else
                quoted += c;
        }
        quoted += '\"';
        bound = std::move(quoted);
    }
    sqlite3_bind_text(s, 1, bound.c_str(), -1, SQLITE_TRANSIENT);
    std::string ts;
    if (type) {
        ts = to_string(*type);
        sqlite3_bind_text(s, 2, ts.c_str(), -1, SQLITE_TRANSIENT);
    } else
        sqlite3_bind_null(s, 2);
    sqlite3_bind_int64(s, 3, limit);
    std::vector<MemoryRecord> out;
    for (;;) {
        int rc = sqlite3_step(s);
        if (rc == SQLITE_DONE)
            break;
        if (rc != SQLITE_ROW) {
            sqlite3_finalize(s);
            check(rc, _db, "memory query");
        }
        out.push_back(row(s));
    }
    sqlite3_finalize(s);
    if (!out.empty()) {
        exec(_db, "BEGIN IMMEDIATE;");
        try {
            sqlite3_stmt *u = nullptr;
            check(sqlite3_prepare_v2(_db,
                                     "UPDATE memories SET "
                                     "last_accessed_at=unixepoch() WHERE id=?1",
                                     -1, &u, nullptr),
                  _db, "prepare access update");
            for (const auto &r : out) {
                sqlite3_bind_int64(u, 1, r.id);
                check(sqlite3_step(u), _db, "access update");
                sqlite3_reset(u);
            }
            sqlite3_finalize(u);
            exec(_db, "COMMIT;");
        } catch (...) {
            try {
                exec(_db, "ROLLBACK;");
            } catch (...) {
            }
            throw;
        }
    }
    return out;
}
ToolResult MemoryRetrieveTool::execute(const nlohmann::json &a) {
    if (!a.contains("query") || !a["query"].is_string())
        return {.status = ToolStatus::PermanentError,
                .error = "query must be a string"};
    std::optional<MemoryType> type;
    if (a.contains("type")) {
        if (!a["type"].is_string() ||
            !(type = memory_type_from_string(a["type"].get<std::string>())))
            return {.status = ToolStatus::PermanentError,
                    .error = "invalid memory type"};
    }
    std::size_t limit = a.value("limit", 8U);
    try {
        auto rows =
            _memory->retrieve(a["query"].get<std::string>(), type, limit);
        nlohmann::json j = nlohmann::json::array();
        for (const auto &r : rows)
            j.push_back({{"id", r.id},
                         {"type", to_string(r.type)},
                         {"content", r.content},
                         {"importance", r.importance},
                         {"confidence", r.confidence}});
        return {.status = ToolStatus::Success, .value = {{"memories", j}}};
    } catch (const std::exception &e) {
        return {.status = ToolStatus::RetryableError, .error = e.what()};
    }
}
} // namespace ai::agent
