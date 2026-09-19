#include "brain/memory/MemoryBackend.hpp"
#include "sqlite3.h"
#include <sstream>
namespace ai::brain {
namespace {
long long ticks(TimePoint t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               t.time_since_epoch())
        .count();
}
TimePoint time_from(long long n) { return TimePoint{Duration{n}}; }
std::string text_of(const SemanticItem &s) {
    auto i = s.attributes.find("text");
    return i != s.attributes.end() &&
                   std::holds_alternative<std::string>(i->second)
               ? std::get<std::string>(i->second)
               : std::string{};
}
MemoryItem row(sqlite3_stmt *s) {
    MemoryItem i;
    i.id = sqlite3_column_int64(s, 0);
    i.type = static_cast<MemoryType>(sqlite3_column_int(s, 1));
    i.content.id = sqlite3_column_int64(s, 2);
    i.content.type = static_cast<SemanticType>(sqlite3_column_int(s, 3));
    i.content.confidence = static_cast<float>(sqlite3_column_double(s, 4));
    i.content.valid = sqlite3_column_int(s, 5) != 0;
    if (sqlite3_column_type(s, 6) != SQLITE_NULL)
        i.content.target = sqlite3_column_int64(s, 6);
    auto txt = reinterpret_cast<const char *>(sqlite3_column_text(s, 7));
    if (txt)
        i.content.attributes["text"] = std::string(txt);
    i.importance = static_cast<float>(sqlite3_column_double(s, 8));
    i.confidence = static_cast<float>(sqlite3_column_double(s, 9));
    i.createdAt = time_from(sqlite3_column_int64(s, 10));
    i.updatedAt = time_from(sqlite3_column_int64(s, 11));
    i.lastAccessed = time_from(sqlite3_column_int64(s, 12));
    i.accessCount = sqlite3_column_int64(s, 13);
    i.version = sqlite3_column_int(s, 14);
    return i;
}
} // namespace
struct SQLiteMemoryBackend::Impl {
        sqlite3 *db{};
};
SQLiteMemoryBackend::SQLiteMemoryBackend(const std::string &p)
    : _impl(std::make_unique<Impl>()) {
    if (sqlite3_open(p.c_str(), &_impl->db) != SQLITE_OK) {
        sqlite3_close(_impl->db);
        _impl->db = nullptr;
        return;
    }
    const char *sql =
        "CREATE TABLE IF NOT EXISTS brain_memory(id INTEGER PRIMARY KEY,type "
        "INTEGER NOT NULL,semantic_id INTEGER NOT NULL,semantic_type INTEGER "
        "NOT NULL,semantic_confidence REAL NOT NULL,semantic_valid INTEGER NOT "
        "NULL,target INTEGER,content_json TEXT NOT NULL,importance REAL NOT "
        "NULL,confidence REAL NOT NULL,created_at INTEGER NOT NULL,updated_at "
        "INTEGER NOT NULL,last_accessed INTEGER NOT NULL,access_count INTEGER "
        "NOT NULL,version INTEGER NOT NULL);CREATE TABLE IF NOT EXISTS "
        "brain_memory_tag(memory_id INTEGER NOT NULL,tag TEXT NOT NULL,PRIMARY "
        "KEY(memory_id,tag));PRAGMA user_version=1;";
    if (sqlite3_exec(_impl->db, sql, nullptr, nullptr, nullptr) != SQLITE_OK) {
        sqlite3_close(_impl->db);
        _impl->db = nullptr;
    }
}
SQLiteMemoryBackend::~SQLiteMemoryBackend() {
    if (_impl && _impl->db)
        sqlite3_close(_impl->db);
}
bool SQLiteMemoryBackend::available() const noexcept {
    return _impl && _impl->db;
}
MemoryId SQLiteMemoryBackend::store(MemoryItem i) {
    if (!available())
        return 0;
    const char *sql = "INSERT OR REPLACE INTO brain_memory "
                      "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";
    sqlite3_stmt *s{};
    if (sqlite3_prepare_v2(_impl->db, sql, -1, &s, nullptr) != SQLITE_OK)
        return 0;
    sqlite3_bind_int64(s, 1, i.id);
    sqlite3_bind_int(s, 2, int(i.type));
    sqlite3_bind_int64(s, 3, i.content.id);
    sqlite3_bind_int(s, 4, int(i.content.type));
    sqlite3_bind_double(s, 5, i.content.confidence);
    sqlite3_bind_int(s, 6, i.content.valid);
    if (i.content.target)
        sqlite3_bind_int64(s, 7, *i.content.target);
    else
        sqlite3_bind_null(s, 7);
    auto txt = text_of(i.content);
    sqlite3_bind_text(s, 8, txt.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(s, 9, i.importance);
    sqlite3_bind_double(s, 10, i.confidence);
    sqlite3_bind_int64(s, 11, ticks(i.createdAt));
    sqlite3_bind_int64(s, 12, ticks(i.updatedAt));
    sqlite3_bind_int64(s, 13, ticks(i.lastAccessed));
    sqlite3_bind_int64(s, 14, i.accessCount);
    sqlite3_bind_int(s, 15, i.version);
    bool ok = sqlite3_step(s) == SQLITE_DONE;
    sqlite3_finalize(s);
    if (!ok)
        return 0;
    sqlite3_exec(_impl->db, "BEGIN", nullptr, nullptr, nullptr);
    sqlite3_stmt *t{};
    sqlite3_prepare_v2(_impl->db,
                       "INSERT OR IGNORE INTO brain_memory_tag VALUES(?,?)", -1,
                       &t, nullptr);
    for (const auto &tag : i.tags) {
        sqlite3_reset(t);
        sqlite3_bind_int64(t, 1, i.id);
        sqlite3_bind_text(t, 2, tag.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(t);
    }
    sqlite3_finalize(t);
    sqlite3_exec(_impl->db, "COMMIT", nullptr, nullptr, nullptr);
    return i.id;
}
bool SQLiteMemoryBackend::update(const MemoryItem &i) { return store(i) != 0; }
bool SQLiteMemoryBackend::remove(MemoryId id) {
    if (!available())
        return false;
    sqlite3_stmt *s{};
    sqlite3_prepare_v2(_impl->db, "DELETE FROM brain_memory WHERE id=?", -1, &s,
                       nullptr);
    sqlite3_bind_int64(s, 1, id);
    bool ok = sqlite3_step(s) == SQLITE_DONE && sqlite3_changes(_impl->db) > 0;
    sqlite3_finalize(s);
    return ok;
}
std::optional<MemoryItem> SQLiteMemoryBackend::get(MemoryId id) {
    if (!available())
        return {};
    sqlite3_stmt *s{};
    sqlite3_prepare_v2(_impl->db, "SELECT * FROM brain_memory WHERE id=?", -1,
                       &s, nullptr);
    sqlite3_bind_int64(s, 1, id);
    std::optional<MemoryItem> out;
    if (sqlite3_step(s) == SQLITE_ROW)
        out = row(s);
    sqlite3_finalize(s);
    return out;
}
std::vector<MemoryItem> SQLiteMemoryBackend::search(const MemoryQuery &q) {
    std::vector<MemoryItem> out;
    if (!available() || !q.maxResults)
        return out;
    std::string sql = "SELECT DISTINCT m.* FROM brain_memory m";
    if (!q.tags.empty())
        sql += " JOIN brain_memory_tag t ON t.memory_id=m.id";
    sql += " WHERE 1=1";
    if (q.type)
        sql += " AND m.type=?";
    if (q.target)
        sql += " AND m.target=?";
    for (std::size_t n = 0; n < q.tags.size(); ++n)
        sql += " AND EXISTS(SELECT 1 FROM brain_memory_tag x WHERE "
               "x.memory_id=m.id AND x.tag=?)";
    sql += " ORDER BY (0.2*m.importance+0.15*m.confidence) DESC,m.updated_at "
           "DESC LIMIT ?";
    sqlite3_stmt *s{};
    if (sqlite3_prepare_v2(_impl->db, sql.c_str(), -1, &s, nullptr) !=
        SQLITE_OK)
        return out;
    int n = 1;
    if (q.type)
        sqlite3_bind_int(s, n++, int(*q.type));
    if (q.target)
        sqlite3_bind_int64(s, n++, *q.target);
    for (const auto &t : q.tags)
        sqlite3_bind_text(s, n++, t.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(s, n, q.maxResults);
    while (sqlite3_step(s) == SQLITE_ROW)
        out.push_back(row(s));
    sqlite3_finalize(s);
    return out;
}
} // namespace ai::brain
