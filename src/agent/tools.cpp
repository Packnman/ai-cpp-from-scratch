#include "ai/agent/tools.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ai::agent {
namespace {
class ExpressionParser {
    public:
        explicit ExpressionParser(std::string_view s) : s_(s) {}
        double run() {
            double v = expression();
            spaces();
            if (pos_ != s_.size())
                fail();
            if (!std::isfinite(v))
                fail();
            return v;
        }

    private:
        void spaces() {
            while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t'))
                ++pos_;
        }
        [[noreturn]] void fail() {
            throw std::invalid_argument("invalid arithmetic expression");
        }
        double expression() {
            double v = term();
            for (;;) {
                spaces();
                if (take('+'))
                    v += term();
                else if (take('-'))
                    v -= term();
                else
                    return v;
            }
        }
        double term() {
            double v = factor();
            for (;;) {
                spaces();
                if (take('*'))
                    v *= factor();
                else if (take('/')) {
                    double d = factor();
                    if (d == 0)
                        throw std::domain_error("division by zero");
                    v /= d;
                } else
                    return v;
            }
        }
        double factor() {
            spaces();
            if (take('+'))
                return factor();
            if (take('-'))
                return -factor();
            if (take('(')) {
                double v = expression();
                spaces();
                if (!take(')'))
                    fail();
                return v;
            }
            std::size_t start = pos_;
            bool dot = false;
            while (pos_ < s_.size() && ((s_[pos_] >= '0' && s_[pos_] <= '9') ||
                                        (!dot && s_[pos_] == '.'))) {
                dot |= s_[pos_] == '.';
                ++pos_;
            }
            if (start == pos_)
                fail();
            try {
                return std::stod(std::string(s_.substr(start, pos_ - start)));
            } catch (...) {
                fail();
            }
        }
        bool take(char c) {
            if (pos_ < s_.size() && s_[pos_] == c) {
                ++pos_;
                return true;
            }
            return false;
        }
        std::string_view s_;
        std::size_t pos_{};
};

bool valid_utf8(std::string_view s) {
    for (std::size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        std::size_t n = 0;
        std::uint32_t cp = 0;
        if (c < 0x80) {
            ++i;
            continue;
        }
        if ((c & 0xe0) == 0xc0) {
            n = 1;
            cp = c & 0x1f;
            if (cp < 2)
                return false;
        } else if ((c & 0xf0) == 0xe0) {
            n = 2;
            cp = c & 0xf;
        } else if ((c & 0xf8) == 0xf0) {
            n = 3;
            cp = c & 7;
        } else
            return false;
        if (i + n >= s.size())
            return false;
        for (std::size_t j = 1; j <= n; ++j) {
            unsigned char d = s[i + j];
            if ((d & 0xc0) != 0x80)
                return false;
            cp = (cp << 6) | (d & 0x3f);
        }
        if ((n == 2 && cp < 0x800) || (n == 3 && cp < 0x10000) ||
            cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
            return false;
        i += n + 1;
    }
    return true;
}
} // namespace

ToolResult CalculatorTool::execute(const nlohmann::json &args) {
    if (!args.contains("expression") || !args["expression"].is_string())
        return {.status = ToolStatus::PermanentError,
                .error = "expression must be a string"};
    try {
        return {.status = ToolStatus::Success,
                .value = {{"value", ExpressionParser(
                                        args["expression"].get<std::string>())
                                        .run()}}};
    } catch (const std::exception &e) {
        return {.status = ToolStatus::PermanentError, .error = e.what()};
    }
}

FileReadTool::FileReadTool(std::filesystem::path root)
    : root_(std::filesystem::weakly_canonical(std::move(root))) {
    if (!std::filesystem::is_directory(root_))
        throw std::invalid_argument("file root is not a directory");
}
ToolResult FileReadTool::execute(const nlohmann::json &args) {
    if (!args.contains("path") || !args["path"].is_string())
        return {.status = ToolStatus::PermanentError,
                .error = "path must be a string"};
    std::filesystem::path rel(args["path"].get<std::string>());
    if (rel.empty() || rel.is_absolute())
        return {.status = ToolStatus::PermanentError,
                .error = "path must be relative"};
    std::error_code ec;
    auto target = std::filesystem::weakly_canonical(root_ / rel, ec);
    if (ec)
        return {.status = ToolStatus::PermanentError,
                .error = "path cannot be resolved"};
    auto mismatch =
        std::mismatch(root_.begin(), root_.end(), target.begin(), target.end());
    if (mismatch.first != root_.end())
        return {.status = ToolStatus::PermanentError,
                .error = "path escapes configured root"};
    if (!std::filesystem::is_regular_file(target, ec) || ec)
        return {.status = ToolStatus::PermanentError,
                .error = "path is not a regular file"};
    auto size = std::filesystem::file_size(target, ec);
    if (ec || size > 1024 * 1024)
        return {.status = ToolStatus::PermanentError,
                .error = "file exceeds 1 MiB"};
    std::ifstream in(target, std::ios::binary);
    std::string content((std::istreambuf_iterator<char>(in)), {});
    if (!in.good() && !in.eof())
        return {.status = ToolStatus::RetryableError,
                .error = "file read failed"};
    if (!valid_utf8(content))
        return {.status = ToolStatus::PermanentError,
                .error = "file is not valid UTF-8"};
    return {.status = ToolStatus::Success,
            .value = {{"path", rel.generic_string()}, {"content", content}}};
}
} // namespace ai::agent
