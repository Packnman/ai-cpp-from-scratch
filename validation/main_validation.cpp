#include "validation.h"
#include "cli_options.h"
#include <iostream>

int main( int nArgc, char** lpArgv )
{
    try
    {
        if( nArgc < 3 )
            throw std::invalid_argument(
                "Usage: main_validation DATA MODEL [--split validation|test --batch N --max-batches N --loss-target all|response] "
                "| main_validation chat MODEL [--temperature F --top-k N --max-tokens N --input-context N --seed N]" );
        CliOptions optOptions( nArgc, lpArgv, 3 );
        // chat は対話生成、それ以外はデータセットに対する損失評価へ進む。
        if( std::string( lpArgv[1] ) == "chat" )
        {
            const int nSeed = optOptions.integer( "--seed", 42 );
            if( nSeed < 0 ) throw std::invalid_argument( "Seed must be nonnegative" );
            ConfigGeneration cfgGeneration;
            {
                cfgGeneration.fTemperature  = optOptions.real( "--temperature", cfgGeneration.fTemperature );
                cfgGeneration.nTopK         = optOptions.integer( "--top-k", cfgGeneration.nTopK );
                cfgGeneration.nMaxTokens    = optOptions.integer( "--max-tokens", cfgGeneration.nMaxTokens );
                cfgGeneration.nContext      = optOptions.integer( "--input-context", 0 );
            }
            optOptions.finish();
            // モデルは対話開始時に一度だけ読み込み、会話履歴と乱数状態を保持する。
            auto bunModel = g_loadConversation( lpArgv[2] );
            const int nContext = cfgGeneration.nContext == 0 ? bunModel.spModel->config().nContext : cfgGeneration.nContext;
            if( nContext <= 0 || nContext > bunModel.spModel->config().nContext )
                throw std::invalid_argument( "Input context must not exceed saved model context" );
            std::mt19937 rngRandom( nSeed );
            Chat( bunModel.tokTokenizer, nContext, std::cin, std::cout,
                  [&]( const TokenIds& c_nHistory )
                  {
                      return Generate( *bunModel.spModel, bunModel.tokTokenizer,
                                       c_nHistory, rngRandom, cfgGeneration );
                  } );
        }
        else
        {
            const auto strSplit = optOptions.string( "--split", "validation" );
            const int nBatch = optOptions.integer( "--batch", 64 );
            const int nMaxBatches = optOptions.integer( "--max-batches", 0 );
            const auto enmTarget = g_parseConversationLossTarget(
                optOptions.string( "--loss-target", "all" ) );
            optOptions.finish();
            Validation( lpArgv[1], lpArgv[2], strSplit, nBatch, nMaxBatches, std::cout, enmTarget );
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
