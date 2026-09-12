#include "validation.h"
#include "cli_options.h"
#include <iostream>

int main( int nArgc, char** lpArgv )
{
    try
    {
        if( nArgc < 3 )
            throw std::invalid_argument(
                "Usage: main_validation DATA MODEL [--split validation|test --batch N --max-batches N] "
                "| main_validation chat MODEL [--temperature F --top-k N --max-tokens N --seed N]" );
        CliOptions optOptions( nArgc, lpArgv, 3 );
        // chat は対話生成、それ以外はデータセットに対する損失評価へ進む。
        if( std::string( lpArgv[1] ) == "chat" )
        {
            const int nSeed = optOptions.integer( "--seed", 42 );
            if( nSeed < 0 ) throw std::invalid_argument( "Seed must be nonnegative" );
            ConfigGeneration cfgGeneration;
            {
                cfgGeneration.fTemperature  =optOptions.real( "--temperature", cfgGeneration.fTemperature );
                cfgGeneration.nTopK         =optOptions.integer( "--top-k", cfgGeneration.nTopK );
                cfgGeneration.nMaxTokens    =optOptions.integer( "--max-tokens", cfgGeneration.nMaxTokens );
            }
            optOptions.finish();
            // モデルは対話開始時に一度だけ読み込み、会話履歴と乱数状態を保持する。
            auto bunModel = g_loadConversation( lpArgv[2] );
            std::mt19937 rngRandom( nSeed );
            TokenIds nHistory = { TokenConversation::BEGIN };
            std::string strInput;
            std::cout << "A> " << std::flush;
            while( std::getline( std::cin, strInput ) )
            {
                // 入力を「話者A・本文・発話終端・話者B」の形にし、B の応答生成を開始する。
                nHistory.push_back( TokenConversation::SPEAKER_A );
                const auto nText = bunModel.tokTokenizer.encode( strInput );
                nHistory.insert( nHistory.end(), nText.begin(), nText.end() );
                nHistory.push_back( TokenConversation::UTTERANCE_END );
                nHistory.push_back( TokenConversation::SPEAKER_B );
                const auto nResponse = Generate(
                    *bunModel.spModel, bunModel.tokTokenizer, nHistory, rngRandom, cfgGeneration );
                std::cout << "B> " << bunModel.tokTokenizer.decode( nResponse ) << '\n';
                nHistory.insert( nHistory.end(), nResponse.begin(), nResponse.end() );
                // 会話終端なら履歴を初期化し、長さ上限で切れた応答には発話終端を補う。
                if( !nResponse.empty() && nResponse.back() == TokenConversation::END )
                {
                    nHistory = { TokenConversation::BEGIN };
                }
                else if( nResponse.empty() || nResponse.back() != TokenConversation::UTTERANCE_END )
                {
                    nHistory.push_back( TokenConversation::UTTERANCE_END );
                }
                // 長い対話では古い履歴を捨て、次回に保持する ID 数を制限する。
                if( nHistory.size() > 128 )
                {
                    nHistory.erase( nHistory.begin(), nHistory.end() - 128 );
                }
                std::cout << "A> " << std::flush;
            }
        }
        else
        {
            const auto strSplit = optOptions.string( "--split", "validation" );
            const int nBatch = optOptions.integer( "--batch", 64 );
            const int nMaxBatches = optOptions.integer( "--max-batches", 0 );
            optOptions.finish();
            Validation( lpArgv[1], lpArgv[2], strSplit, nBatch, nMaxBatches, std::cout );
        }
        return 0;
    }
    // CLI の最上位で例外を受け取り、エラーメッセージと失敗の終了コードを返す。
    catch( const std::exception& c_excError )
    {
        std::cerr << c_excError.what() << '\n';
        return 1;
    }
}
