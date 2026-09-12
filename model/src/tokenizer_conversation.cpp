#include "tokenizer_conversation.h"
#include <numeric>
#include <stdexcept>

TokenConversation::TokenConversation( std::string_view c_strTrainingText )
    : _tokCharacters( c_strTrainingText )
{
}

TokenIds TokenConversation::encode( std::string_view c_strText ) const
{
    auto nIds = _tokCharacters.encodeUnknown( c_strText, -1 );
    for( auto& nId : nIds )
    {
        nId = nId < 0 ? UNK : nId + SPECIAL_COUNT;
    }
    return nIds;
}

std::string TokenConversation::decode( const TokenIds& c_nIds ) const
{
    std::string strResult;
    for( int nId : c_nIds )
    {
        if( nId < 0 || nId >= vocabSize() )
        {
            throw std::out_of_range( "TokenConversation: invalid ID" );
        }
        if( nId == UNK )
        {
            strResult += "�";
        }
        else if( nId >= SPECIAL_COUNT )
        {
            strResult += _tokCharacters.decode( { nId - SPECIAL_COUNT } );
        }
    }
    return strResult;
}

std::string TokenConversation::vocabulary() const
{
    TokenIds nIds( _tokCharacters.vocabSize() );
    std::iota( nIds.begin(), nIds.end(), 0 );
    return _tokCharacters.decode( nIds );
}

int TokenConversation::vocabSize() const
{
    return static_cast<int>( _tokCharacters.vocabSize() ) + SPECIAL_COUNT;
}
