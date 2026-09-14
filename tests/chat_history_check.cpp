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
    std::istringstream truncatedInput( "abc\n" );
    std::ostringstream truncatedOutput;
    bool overBudget = false;
    try
    {
        Chat( tokenizer, 4, truncatedInput, truncatedOutput,
              [&]( const TokenIds& history )
              {
                  histories.push_back( history );
                  return TokenIds{ TokenConversation::UTTERANCE_END };
              } );
    }
    catch( const std::invalid_argument& )
    {
        overBudget = true;
    }
    require( overBudget && histories.empty(),
             "An over-budget current question must be rejected without token truncation" );
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

    histories.clear();
    std::istringstream documentInput( "a\nb\n" );
    std::ostringstream documentOutput;
    DocumentChat( tokenizer, "a.b.", 12, documentInput, documentOutput,
                  [&]( const TokenIds& prompt )
                  {
                      histories.push_back( prompt );
                      return TokenIds{ TokenConversation::UTTERANCE_END };
                  } );
    require( histories.size() == 2 && histories[0].size() <= 12 &&
                 histories[1].size() <= 12,
             "Every document question prompt must remain inside its budget" );
    require( documentOutput.str().find( "Q> A> " ) != std::string::npos,
             "Document mode must use Q> / A> prompts" );
    return 0;
}
