#include "train.h"
#include "cli_options.h"
#include <iostream>

int main( int nArgc, char** lpArgv )
{
    try
    {
        // prepare は元データを train / validation / test に変換する専用の入口。
        if( nArgc >= 2 && std::string( lpArgv[1] ) == "prepare" )
        {
            if( nArgc < 4 || nArgc > 5 )
                throw std::invalid_argument( "Usage: main_train prepare SOURCE_REPO OUTPUT_DIR [REVISION]" );
            g_prepareConversations( lpArgv[2], lpArgv[3], nArgc == 5 ? lpArgv[4] : "unspecified" );
            std::cout << "Prepared train/validation/test and metadata.json\n";
            return 0;
        }
        if( nArgc < 3 )
            throw std::invalid_argument(
                "Usage: main_train DATA MODEL [--from-model SOURCE --epochs N --batch N --lr F "
                "--clip F --seed N --dropout F --blocks N --embedding N --heads N --hidden N "
                "--context N --max-batches N] | main_train prepare SOURCE_REPO OUTPUT_DIR [REVISION]" );
        // DATA と MODEL に続くオプションを、モデル設定と学習条件に振り分ける。
        CliOptions optOptions( nArgc, lpArgv, 3 );
        const int nSeed = optOptions.integer( "--seed", 42 );
        if( nSeed < 0 ) throw std::invalid_argument( "Seed must be nonnegative" );
        const bool isFinetuning = optOptions.contains( "--from-model" );
        const auto strSource = optOptions.string( "--from-model" );
        // 保存重みと形状が食い違わないよう、追加学習ではモデル構成の上書きを拒否する。
        if( isFinetuning )
        {
            for( const auto* lpOption : { "--blocks", "--embedding", "--heads", "--hidden",
                                         "--context", "--dropout" } )
            {
                if( optOptions.contains( lpOption ) )
                    throw std::invalid_argument( std::string( "Cannot override saved model option: " ) + lpOption );
            }
        }
        TransformerConfig cfgModel;
        cfgModel.nBlocks = optOptions.integer( "--blocks", cfgModel.nBlocks );
        cfgModel.nEmbedding = optOptions.integer( "--embedding", cfgModel.nEmbedding );
        cfgModel.nHeads = optOptions.integer( "--heads", cfgModel.nHeads );
        cfgModel.nHidden = optOptions.integer( "--hidden", cfgModel.nHidden );
        cfgModel.nContext = optOptions.integer( "--context", cfgModel.nContext );
        cfgModel.fDropout = optOptions.real( "--dropout", cfgModel.fDropout );
        ConfigTraining cfgTraining;
        cfgTraining.nSeed = nSeed;
        cfgTraining.nEpochs = optOptions.integer( "--epochs", cfgTraining.nEpochs );
        cfgTraining.nBatchSize = optOptions.integer( "--batch", cfgTraining.nBatchSize );
        cfgTraining.fLearningRate = optOptions.real( "--lr", cfgTraining.fLearningRate );
        cfgTraining.fClipNorm = optOptions.real( "--clip", cfgTraining.fClipNorm );
        cfgTraining.nMaxBatches = optOptions.integer( "--max-batches", cfgTraining.nMaxBatches );
        // 読み取られていないオプションを検出し、指定ミスを見逃さない。
        optOptions.finish();
        if( isFinetuning )
        {
            Finetuning( lpArgv[1], strSource, lpArgv[2], cfgTraining, std::cout );
        }
        else
        {
            Training( lpArgv[1], lpArgv[2], cfgModel, cfgTraining, std::cout );
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
