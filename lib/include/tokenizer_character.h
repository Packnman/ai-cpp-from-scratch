#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tokenizer.h"

class TokenCharacter final : public Tokenizer<std::string_view,std::string>
{
public:
    explicit TokenCharacter(std::string_view corpus);

private:
    std::vector<char32_t> _szVocabulary;
    std::unordered_map<char32_t,std::int32_t> _mapTokenIdByCharacter;

private:
    static void _appendUtf8(std::string& text,char32_t codePoint);
    static std::vector<char32_t> _decodeUtf8(std::string_view text);
public:
    TokenIds encode(std::string_view text) const override;
    std::string decode(const TokenIds& tokenIds) const override;
    std::size_t vocabSize() const noexcept override;
};
