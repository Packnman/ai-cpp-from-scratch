#include "tokenizer_character.h"

#include <algorithm>
#include <limits>
#include <stdexcept>


std::vector<char32_t> TokenCharacter::_decodeUtf8(std::string_view text)
{
    std::vector<char32_t> result;
    result.reserve( text.size() );

    std::size_t offset =0;
    while( offset<text.size() )
    {
        const auto first =static_cast<unsigned char>(text[offset]);
        char32_t codePoint =0;
        std::size_t length =0;
        char32_t minimum =0;

        if( first<=0x7f )
        {
            codePoint =first;
            length =1;
        }
        else if( (first>=0xc2)&&(first<=0xdf) )
        {
            codePoint =first&0x1f;
            length =2;
            minimum =0x80;
        }
        else if( (first>=0xe0)&&(first<=0xef) )
        {
            codePoint =first&0x0f;
            length =3;
            minimum =0x800;
        }
        else if( (first>=0xf0)&&(first<=0xf4) )
        {
            codePoint =first&0x07;
            length =4;
            minimum =0x10000;
        }
        else
        {
            throw std::invalid_argument("TokenCharacter: invalid UTF-8");
        }

        if( length>text.size()-offset )
        {
            throw std::invalid_argument("TokenCharacter: truncated UTF-8");
        }
        for( std::size_t byte=1;byte<length;++byte )
        {
            const auto continuation =
                static_cast<unsigned char>(text[offset+byte]);
            if( (continuation&0xc0)!=0x80 )
            {
                throw std::invalid_argument("TokenCharacter: invalid UTF-8 continuation byte");
            }
            codePoint =(codePoint<<6)|(continuation&0x3f);
        }

        if( ((length>1)&&(codePoint<minimum))||
            ((codePoint>=0xd800)&&(codePoint<=0xdfff))||
            (codePoint>0x10ffff) )
        {
            throw std::invalid_argument("TokenCharacter: invalid UTF-8 code point");
        }

        result.push_back( codePoint );
        offset +=length;
    }
    return result;
}

void TokenCharacter::_appendUtf8(std::string& destination,char32_t codePoint)
{
    if( codePoint<=0x7f )
    {
        destination.push_back( static_cast<char>(codePoint) );
    }
    else if( codePoint<=0x7ff )
    {
        destination.push_back( static_cast<char>(0xc0|(codePoint>>6)) );
        destination.push_back( static_cast<char>(0x80|(codePoint&0x3f)) );
    }
    else if( codePoint<=0xffff )
    {
        destination.push_back( static_cast<char>(0xe0|(codePoint>>12)) );
        destination.push_back( static_cast<char>(0x80|((codePoint>>6)&0x3f)) );
        destination.push_back( static_cast<char>(0x80|(codePoint&0x3f)) );
    }
    else
    {
        destination.push_back( static_cast<char>(0xf0|(codePoint>>18)) );
        destination.push_back( static_cast<char>(0x80|((codePoint>>12)&0x3f)) );
        destination.push_back( static_cast<char>(0x80|((codePoint>>6)&0x3f)) );
        destination.push_back( static_cast<char>(0x80|(codePoint&0x3f)) );
    }
}

TokenCharacter::TokenCharacter(std::string_view corpus)
    :_szVocabulary( _decodeUtf8(corpus) )
{
    std::sort( _szVocabulary.begin(),_szVocabulary.end() );
    _szVocabulary.erase(
        std::unique(_szVocabulary.begin(),_szVocabulary.end()),
        _szVocabulary.end()
    );
    if( _szVocabulary.size()>
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) )
    {
        throw std::length_error("TokenCharacter: vocabulary is too large");
    }

    for( std::size_t tokenId=0;tokenId<_szVocabulary.size();++tokenId )
    {
        _mapTokenIdByCharacter.emplace(
            _szVocabulary[tokenId],
            static_cast<std::int32_t>(tokenId)
        );
    }
}

TokenIds
TokenCharacter::encode(std::string_view text) const
{
    const auto codePoints =_decodeUtf8( text );
    TokenIds result;
    result.reserve( codePoints.size() );
    for( char32_t codePoint : codePoints )
    {
        const auto token =_mapTokenIdByCharacter.find( codePoint );
        if( token==_mapTokenIdByCharacter.end() )
        {
            throw std::out_of_range("TokenCharacter::encode: unknown character");
        }
        result.push_back( token->second );
    }
    return result;
}

std::string
TokenCharacter::decode(const TokenIds& tokenIds) const
{
    std::string result;
    for( std::int32_t tokenId : tokenIds )
    {
        if( (tokenId<0)||
            (static_cast<std::size_t>(tokenId)>=_szVocabulary.size()) )
        {
            throw std::out_of_range("TokenCharacter::decode: invalid token ID");
        }
        _appendUtf8( result,_szVocabulary[static_cast<std::size_t>(tokenId)] );
    }
    return result;
}

std::size_t TokenCharacter::vocabSize() const noexcept
{
    return _szVocabulary.size();
}


TokenIds TokenCharacter::encodeUnknown( std::string_view c_strText, std::int32_t nUnknownId ) const
{
    TokenIds nResult;
    for( char32_t nCodePoint : _decodeUtf8( c_strText ) )
    {
        const auto itrToken = _mapTokenIdByCharacter.find( nCodePoint );
        nResult.push_back( itrToken == _mapTokenIdByCharacter.end() ? nUnknownId : itrToken->second );
    }
    return nResult;
}
