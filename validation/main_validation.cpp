#include "validation.h"
#include "cli_options.h"
#include <fstream>
#include <iostream>
#include <iterator>

int main( int nArgc, char** lpArgv )
{
    try
    {
        if( nArgc < 3 )
            throw std::invalid_argument(
                "Usage: main_validation DATA MODEL [evaluation options] "
                "| main_validation chat MODEL [generation options] "
                "| main_validation document MODEL FILE [generation options]" );
        const std::string strMode = lpArgv[1];
        if( strMode == "chat" || strMode == "document" )
        {
            if( strMode == "document" && nArgc < 4 )
                throw std::invalid_argument( "Document mode requires a UTF-8 text file" );
            CliOptions optOptions( nArgc, lpArgv, strMode == "document" ? 4 : 3 );
            const int nSeed = optOptions.integer( "--seed", 42 );
            if( nSeed < 0 ) throw std::invalid_argument( "Seed must be nonnegative" );
            ConfigGeneration cfgGeneration;
            cfgGeneration.fTemperature =
                optOptions.real( "--temperature", cfgGeneration.fTemperature );
            cfgGeneration.nTopK = optOptions.integer( "--top-k", cfgGeneration.nTopK );
            cfgGeneration.nMaxTokens =
                optOptions.integer( "--max-tokens", cfgGeneration.nMaxTokens );
            cfgGeneration.nContext = optOptions.integer( "--input-context", 0 );
            optOptions.finish();

            auto bunModel = g_loadConversation( lpArgv[2] );
            const int nContext = cfgGeneration.nContext == 0
                ? bunModel.spModel->config().nContext : cfgGeneration.nContext;
            if( nContext <= 0 || nContext > bunModel.spModel->config().nContext ||
                cfgGeneration.nMaxTokens <= 0 || cfgGeneration.nMaxTokens >= nContext )
                throw std::invalid_argument(
                    "Input context must fit the model and exceed max-tokens" );
            const int nInputBudget = nContext - cfgGeneration.nMaxTokens;
            std::mt19937 rngRandom( nSeed );
            const auto fnGenerate = [&]( const TokenIds& c_nHistory )
            {
                return Generate( *bunModel.spModel, bunModel.tokTokenizer,
                                 c_nHistory, rngRandom, cfgGeneration );
            };
            if( strMode == "chat" )
            {
                Chat( bunModel.tokTokenizer, nInputBudget, std::cin, std::cout,
                      fnGenerate );
            }
            else
            {
                std::ifstream ifsDocument( lpArgv[3], std::ios::binary );
                if( !ifsDocument )
                    throw std::runtime_error( "Cannot read document file" );
                const std::string strDocument(
                    ( std::istreambuf_iterator<char>( ifsDocument ) ),
                    std::istreambuf_iterator<char>() );
                DocumentChat( bunModel.tokTokenizer, strDocument, nInputBudget,
                              std::cin, std::cout, fnGenerate );
            }
        }
        else
        {
            CliOptions optOptions( nArgc, lpArgv, 3 );
            const auto strSplit = optOptions.string( "--split", "validation" );
            const int nBatch = optOptions.integer( "--batch", 64 );
            const int nMaxBatches = optOptions.integer( "--max-batches", 0 );
            const auto enmTarget = g_parseConversationLossTarget(
                optOptions.string( "--loss-target", "all" ) );
            optOptions.finish();
            Validation( lpArgv[1], lpArgv[2], strSplit, nBatch, nMaxBatches,
                        std::cout, enmTarget );
        }
        return 0;
    }
    catch( const std::exception& c_excError )
    {
        std::cerr << c_excError.what() << '\n';
        return 1;
    }
}
