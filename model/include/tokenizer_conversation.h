#pragma once

#include "tokenizer_character.h"

class TokenConversation
{
public:
    static constexpr int PAD = 0;
    static constexpr int UNK = 1;
    static constexpr int BEGIN = 2;
    static constexpr int END = 3;
    static constexpr int SPEAKER_A = 4;
    static constexpr int SPEAKER_B = 5;
    static constexpr int UTTERANCE_END = 6;
    static constexpr int SPECIAL_COUNT = 7;

    explicit TokenConversation( std::string_view c_strTrainingText );
    TokenIds encode( std::string_view c_strText ) const;
    std::string decode( const TokenIds& c_nIds ) const;
    std::string vocabulary() const;
    int vocabSize() const;

private:
    TokenCharacter _tokCharacters;
};
