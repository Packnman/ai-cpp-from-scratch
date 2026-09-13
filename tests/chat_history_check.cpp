#include "validation.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace
{
void require( bool condition, const char* message )
{
    if( !condition ) throw std::runtime_error( message );
}
}

int main()
{
    TokenConversation tokenizer( "abc" );
    std::vector<TokenIds> histories;
    std::istringstream endInput( "a\nb\n" );
    std::ostringstream endOutput;
    int call = 0;
    Chat( tokenizer, 32, endInput, endOutput,
          [&]( const TokenIds& history )
          {
              histories.push_back( history );
              return call++ == 0 ? TokenIds{ TokenConversation::END }
                                 : TokenIds{ TokenConversation::UTTERANCE_END };
          } );
    const int a = tokenizer.encode( "a" )[0];
    require( histories.size() == 2 && histories[1].front() == TokenConversation::BEGIN &&
                 std::find( histories[1].begin(), histories[1].end(), a ) == histories[1].end(),
             "END must reset history before the next input" );
    require( endOutput.str().find(
                 "B> \n[system] 会話終端を検出したため、履歴をリセットしました。\n" ) !=
                 std::string::npos,
             "END must print an empty response and reset notification" );

    histories.clear();
    std::istringstream utteranceInput( "a\nb\n" );
    std::ostringstream utteranceOutput;
    Chat( tokenizer, 32, utteranceInput, utteranceOutput,
          [&]( const TokenIds& history )
          {
              histories.push_back( history );
              return TokenIds{ TokenConversation::UTTERANCE_END };
          } );
    require( histories.size() == 2 &&
                 std::find( histories[1].begin(), histories[1].end(), a ) != histories[1].end(),
             "UTTERANCE_END must retain history" );
    require( utteranceOutput.str().find( "[system]" ) == std::string::npos,
             "UTTERANCE_END must not print a reset notification" );

    histories.clear();
    std::istringstream truncatedInput( "abc\nb\n" );
    std::ostringstream truncatedOutput;
    Chat( tokenizer, 4, truncatedInput, truncatedOutput,
          [&]( const TokenIds& history )
          {
              histories.push_back( history );
              return TokenIds{ TokenConversation::UTTERANCE_END };
          } );
    const int b = tokenizer.encode( "b" )[0];
    const int c = tokenizer.encode( "c" )[0];
    const TokenIds expectedPrefix = {
        c, TokenConversation::UTTERANCE_END, TokenConversation::SPEAKER_B,
        TokenConversation::UTTERANCE_END
    };
    require( histories.size() == 2 && histories[1].size() == 8 &&
                 std::equal( expectedPrefix.begin(), expectedPrefix.end(), histories[1].begin() ) &&
                 histories[1][5] == b,
             "Context trimming must preserve the most recent configured tokens" );
    return 0;
}
