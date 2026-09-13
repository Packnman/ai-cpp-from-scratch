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
    TokenConversation tokenizer( "abcあいう" );
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
    histories.clear();
    std::istringstream deleteInput( "ab\x7f" "c\n" );
    std::ostringstream deleteOutput;
    Chat( tokenizer, 32, deleteInput, deleteOutput,
          [&]( const TokenIds& history )
          {
              histories.push_back( history );
              return TokenIds{ TokenConversation::UTTERANCE_END };
          } );
    TokenIds expectedDelete = { TokenConversation::BEGIN, TokenConversation::SPEAKER_A };
    const auto encodedDelete = tokenizer.encode( "ac" );
    expectedDelete.insert( expectedDelete.end(), encodedDelete.begin(), encodedDelete.end() );
    expectedDelete.push_back( TokenConversation::UTTERANCE_END );
    expectedDelete.push_back( TokenConversation::SPEAKER_B );
    require( histories.size() == 1 && histories[0] == expectedDelete,
             "Delete must remove the preceding input character" );

    histories.clear();
    std::istringstream backspaceInput( "あい\bう\n" );
    std::ostringstream backspaceOutput;
    Chat( tokenizer, 32, backspaceInput, backspaceOutput,
          [&]( const TokenIds& history )
          {
              histories.push_back( history );
              return TokenIds{ TokenConversation::UTTERANCE_END };
          } );
    TokenIds expectedBackspace = { TokenConversation::BEGIN, TokenConversation::SPEAKER_A };
    const auto encodedBackspace = tokenizer.encode( "あう" );
    expectedBackspace.insert( expectedBackspace.end(), encodedBackspace.begin(), encodedBackspace.end() );
    expectedBackspace.push_back( TokenConversation::UTTERANCE_END );
    expectedBackspace.push_back( TokenConversation::SPEAKER_B );
    require( histories.size() == 1 && histories[0] == expectedBackspace,
             "Backspace must remove one complete UTF-8 character" );
    return 0;
}
