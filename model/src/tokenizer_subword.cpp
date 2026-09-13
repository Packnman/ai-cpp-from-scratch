#include "tokenizer_conversation.h"
#include <array>
#include <cstdio>
#include <sentencepiece_model.pb.h>
#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#include <stdexcept>
#include <unordered_map>

namespace
{
void g_check( const sentencepiece::util::Status& c_stsStatus )
{
    if ( !c_stsStatus.ok() )
    {
        throw std::runtime_error( "SentencePiece: " + c_stsStatus.ToString() );
    }
}

// Keep control/space bytes and literal SentencePiece whitespace marker lossless.
// These spans use the model's byte tokens, never user-defined special symbols.
std::size_t g_literalLength( std::string_view c_strText, std::size_t nOffset )
{
    const auto nByte = static_cast<unsigned char>( c_strText[nOffset] );
    if ( nByte <= 0x20 || nByte == 0x7f )
    {
        return 1;
    }
    return c_strText.substr( nOffset, 3 ) == "▁" ? 3 : 0;
}

template <class Text, class Literal>
void g_spans( std::string_view c_strText, Text fnText, Literal fnLiteral )
{
    std::size_t nStart = 0;
    for ( std::size_t nIndex = 0; nIndex < c_strText.size(); )
    {
        const auto nLength = g_literalLength( c_strText, nIndex );
        if ( nLength == 0 )
        {
            ++nIndex;
            continue;
        }
        if ( nIndex > nStart )
        {
            fnText( c_strText.substr( nStart, nIndex - nStart ) );
        }
        fnLiteral( c_strText.substr( nIndex, nLength ) );
        nIndex += nLength;
        nStart = nIndex;
    }
    if ( nStart < c_strText.size() )
    {
        fnText( c_strText.substr( nStart ) );
    }
}

std::string g_bytePiece( unsigned char nByte )
{
    char strPiece[7];
    std::snprintf( strPiece, sizeof( strPiece ), "<0x%02X>", nByte );
    return strPiece;
}
} // namespace

TokenConversation TokenConversation::trainSubword( const std::vector<std::string>& c_strSentences,
                                                   int nVocabularySize )
{
    if ( nVocabularySize < SPECIAL_COUNT + 256 )
    {
        throw std::invalid_argument( "Subword vocabulary must include 7 special and 256 byte tokens" );
    }
    std::vector<std::string> strWords;
    for ( const auto& c_strText : c_strSentences )
    {
        TokenCharacter tokValidate( c_strText ); // Reject malformed UTF-8, as in character mode.
        g_spans(
            c_strText, [&]( std::string_view c_strPart ) { strWords.emplace_back( c_strPart ); },
            []( std::string_view ) {} );
    }
    if ( strWords.empty() )
    {
        throw std::invalid_argument( "Subword training text is empty" );
    }
    const std::unordered_map<std::string, std::string> c_mapOptions = {
        { "model_type", "bpe" },
        { "vocab_size", std::to_string( nVocabularySize ) },
        { "byte_fallback", "true" },
        { "character_coverage", "0.9995" },
        { "normalization_rule_name", "identity" },
        { "add_dummy_prefix", "false" },
        { "remove_extra_whitespaces", "false" },
        { "hard_vocab_limit", "false" },
        { "pad_id", "0" },
        { "unk_id", "1" },
        { "bos_id", "2" },
        { "eos_id", "3" },
        { "control_symbols", "<speaker_a>,<speaker_b>,<utterance_end>" },
        { "num_threads", "1" },
        { "shuffle_input_sentence", "false" },
        { "max_sentence_length", "1048576" },
        { "minloglevel", "1" } };
    std::string c_strSerialized;
    g_check( sentencepiece::SentencePieceTrainer::Train( c_mapOptions, strWords, &c_strSerialized ) );
    return fromSubwordModel( c_strSerialized );
}

TokenConversation TokenConversation::fromSubwordModel( const std::string& c_strSerialized )
{
    auto spProcessor = std::make_shared<sentencepiece::SentencePieceProcessor>();
    g_check( spProcessor->LoadFromSerializedProto( c_strSerialized ) );
    const auto& c_prtModel = spProcessor->model_proto();
    const auto& c_nrmNormalizer = c_prtModel.normalizer_spec();
    if ( c_prtModel.trainer_spec().model_type() != sentencepiece::TrainerSpec::BPE ||
         !c_prtModel.trainer_spec().byte_fallback() || c_nrmNormalizer.name() != "identity" ||
         !c_nrmNormalizer.precompiled_charsmap().empty() || c_nrmNormalizer.add_dummy_prefix() ||
         c_nrmNormalizer.remove_extra_whitespaces() || !c_nrmNormalizer.escape_whitespaces() ||
         c_prtModel.has_denormalizer_spec() )
    {
        throw std::invalid_argument( "Unsupported subword normalization or model type" );
    }
    const std::array<std::string, SPECIAL_COUNT> c_strSpecials = {
        "<pad>", "<unk>", "<s>", "</s>", "<speaker_a>", "<speaker_b>", "<utterance_end>" };
    if ( spProcessor->GetPieceSize() < SPECIAL_COUNT + 256 )
    {
        throw std::invalid_argument( "Incomplete subword model" );
    }
    for ( int nIndex = 0; nIndex < SPECIAL_COUNT; ++nIndex )
    {
        if ( spProcessor->IdToPiece( nIndex ) != c_strSpecials[nIndex] ||
             ( nIndex == UNK ? !spProcessor->IsUnknown( nIndex ) : !spProcessor->IsControl( nIndex ) ) )
        {
            throw std::invalid_argument( "Subword special IDs mismatch" );
        }
    }
    for ( int nByte = 0; nByte < 256; ++nByte )
    {
        if ( !spProcessor->IsByte( spProcessor->PieceToId( g_bytePiece( nByte ) ) ) )
        {
            throw std::invalid_argument( "Subword byte fallback is required" );
        }
    }
    for ( int nId = SPECIAL_COUNT; nId < spProcessor->GetPieceSize(); ++nId )
    {
        if ( spProcessor->IsControl( nId ) || spProcessor->IsUnknown( nId ) || spProcessor->IsUnused( nId ) ||
             c_prtModel.pieces( nId ).type() == sentencepiece::ModelProto::SentencePiece::USER_DEFINED )
        {
            throw std::invalid_argument( "Unexpected subword special token" );
        }
    }
    TokenConversation resultValue( "" );
    resultValue._spSubword = std::move( spProcessor );
    resultValue._strSubwordModel = c_strSerialized;
    return resultValue;
}

TokenIds TokenConversation::_encodeSubword( std::string_view c_strText ) const
{
    TokenCharacter tokValidate( c_strText );
    TokenIds resultValue;
    g_spans(
        c_strText,
        [&]( std::string_view c_strPart )
        {
            std::vector<int> nIds;
            g_check( _spSubword->Encode( c_strPart, &nIds ) );
            resultValue.insert( resultValue.end(), nIds.begin(), nIds.end() );
        },
        [&]( std::string_view c_strPart )
        {
            for ( unsigned char nByte : c_strPart )
            {
                resultValue.push_back( _spSubword->PieceToId( g_bytePiece( nByte ) ) );
            }
        }
    );
    //
    return resultValue;
}

std::string TokenConversation::_decodeSubword( const TokenIds& nIds ) const
{
    for ( int nId : nIds )
    {
        if ( nId < 0 || nId >= vocabSize() )
        {
            throw std::out_of_range( "Invalid subword ID" );
        }
    }
    std::string resultValue;
    g_check( _spSubword->Decode( std::vector<int>( nIds.begin(), nIds.end() ), &resultValue ) );
    return resultValue;
}

int TokenConversation::_subwordSize() const
{
    return _spSubword->GetPieceSize();
}
