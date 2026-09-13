#pragma once

#include "tokenizer_character.h"
#include <memory>

namespace sentencepiece
{
class SentencePieceProcessor;
}

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
    // Construct BPE from train utterances only; serialized bytes are fixed on finetuning.
    static TokenConversation trainSubword(
        const std::vector<std::string>& sentences,
        int vocabularySize = 4096
    );
    static TokenConversation fromSubwordModel( const std::string& serialized );
    bool isSubword() const
    {
        return static_cast<bool>( _spSubword );
    }
    const std::string& subwordModel() const
    {
        return _strSubwordModel;
    }
    TokenIds encode( std::string_view c_strText ) const;
    std::string decode( const TokenIds& c_nIds ) const;
    std::string vocabulary() const;
    int vocabSize() const;

private:
    TokenCharacter _tokCharacters;
    std::shared_ptr<const sentencepiece::SentencePieceProcessor> _spSubword;
    std::string _strSubwordModel;

    TokenIds _encodeSubword( std::string_view text ) const;
    std::string _decodeSubword( const TokenIds& ids ) const;
    int _subwordSize() const;
};
