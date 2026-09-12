#include "cuda_memory.h"
#include "train.h"
#include "optimizer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>

using Json = nlohmann::json;

namespace
{
// CUDA の失敗を例外に変換し、呼出元のエラー処理へ伝える。
void _CheckCuda( cudaError_t cudStatus )
{
    if( cudStatus != cudaSuccess )
    {
        throw std::runtime_error( cudaGetErrorString( cudStatus ) );
    }
}

// 空の split と会話 ID の重複を拒否し、学習・評価データの混入を防ぐ。
void _CheckSplits(
    const std::vector<Conversation>& c_cnvTrain,
    const std::vector<Conversation>& c_cnvValidation,
    const std::vector<Conversation>& c_cnvTest
)
{
    std::set<std::int64_t> nIds;
    for( const auto* c_lpSplit : { &c_cnvTrain, &c_cnvValidation, &c_cnvTest } )
    {
        if( c_lpSplit->empty() )
        {
            throw std::invalid_argument( "All splits must be nonempty" );
        }
        for( const auto& c_cnvDialogue : *c_lpSplit )
        {
            if( !nIds.insert( c_cnvDialogue.nId ).second )
            {
                throw std::invalid_argument( "Dialogue leakage across splits" );
            }
        }
    }
}
} // namespace

// 全パラメータの勾配を同じ倍率で縮小する。戻り値は縮小前の全体 L2 ノルム。
double ClipGradients( Model& mdlModel, float fMaximum )
{
    if( !std::isfinite( fMaximum ) || fMaximum <= 0.0f )
    {
        throw std::invalid_argument( "Clip norm must be finite and positive" );
    }
    cublasHandle_t cblHandle;
    if( cublasCreate( &cblHandle ) != CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error( "Cannot create gradient norm handle" );
    }
    // 各勾配の L2 ノルムを GPU で求め、二乗和からモデル全体のノルムを求める。
    double dblSquared = 0.0;
    try
    {
        for( const auto* c_lpParameter : mdlModel.getParams() )
        {
            float fNorm = 0.0f;
            if( c_lpParameter->_mGrad.numel() > std::numeric_limits<int>::max() ||
                cublasSnrm2( cblHandle, static_cast<int>( c_lpParameter->_mGrad.numel() ),
                             c_lpParameter->_mGrad.data(), 1, &fNorm ) != CUBLAS_STATUS_SUCCESS ||
                !std::isfinite( fNorm ) )
            {
                throw std::runtime_error( "Invalid gradient norm" );
            }
            dblSquared += static_cast<double>( fNorm ) * fNorm;
        }
    }
    // ノルム計算が途中で失敗した場合も cuBLAS のハンドルを解放する。
    catch( ... )
    {
        cublasDestroy( cblHandle );
        throw;
    }
    cublasDestroy( cblHandle );
    const double dblNorm = std::sqrt( dblSquared );
    // 上限を超えたときだけ maximum / norm を掛け、勾配全体の方向を保つ。
    if( dblNorm > fMaximum )
    {
        for( auto* lpParameter : mdlModel.getParams() )
        {
            cuda_scale(
                lpParameter->_mGrad,
                static_cast<float>( fMaximum / dblNorm )
            );
        }
    }
    return dblNorm;
}

namespace
{
// 新規学習と追加学習の共通本体。モデル準備、各 split の実行、最良重みの保存を行う。
void _Prop( const std::string& c_strDataDirectory,
                          const std::string& c_strModelDirectory, TransformerConfig cfgModel,
                          const ConfigTraining& c_cfgTraining, std::ostream& stmLog,
                          const std::string& c_strSource = {} )
{
    if( c_cfgTraining.nEpochs <= 0 || c_cfgTraining.nBatchSize <= 0 ||
        c_cfgTraining.nMaxBatches < 0 || !std::isfinite( c_cfgTraining.fLearningRate ) ||
        c_cfgTraining.fLearningRate <= 0.0f || !std::isfinite( c_cfgTraining.fClipNorm ) ||
        c_cfgTraining.fClipNorm <= 0.0f )
    {
        throw std::invalid_argument( "Invalid training configuration" );
    }
    // 学習・検証・最終テストのデータを読み込み、分割の妥当性を確認する。
    const auto cnvTrain = g_readConversations( c_strDataDirectory + "/train.jsonl" );
    const auto cnvValidation = g_readConversations( c_strDataDirectory + "/validation.jsonl" );
    const auto cnvTest = g_readConversations( c_strDataDirectory + "/test.jsonl" );
    _CheckSplits( cnvTrain, cnvValidation, cnvTest );
    const bool isFinetuning = !c_strSource.empty();
    if( isFinetuning )
    {
        // パスを正規化して同一ディレクトリへの上書きと既存出力の破壊を防ぐ。
        const auto pthOutput = std::filesystem::weakly_canonical( c_strModelDirectory );
        const auto pthSource = std::filesystem::canonical( c_strSource );
        if( pthOutput == pthSource ||
            ( std::filesystem::exists( pthOutput ) &&
              ( !std::filesystem::is_directory( pthOutput ) ||
                !std::filesystem::is_empty( pthOutput ) ) ) )
        {
            throw std::invalid_argument( "Finetuning output must be a different new or empty directory" );
        }
    }
    // 追加学習では重みと語彙を読み込む。新規学習の語彙は train の文章だけから作る。
    auto bunModel = [&]() -> ConversationBundle
    {
        if( isFinetuning )
        {
            return g_loadConversation( c_strSource );
        }
        TokenConversation tokTokenizer( g_trainingText( cnvTrain ) );
        cfgModel.nVocabulary = tokTokenizer.vocabSize();
        cfgModel.nSeed = c_cfgTraining.nSeed;
        return { std::make_unique<Transformer>( cfgModel ), std::move( tokTokenizer ) };
    }();
    auto& trnModel = *bunModel.spModel;
    const auto& tokTokenizer = bunModel.tokTokenizer;
    cfgModel = trnModel.config();
    ConversationDataset datTrain( cnvTrain, tokTokenizer, cfgModel.nContext );
    ConversationDataset datValidation( cnvValidation, tokTokenizer, cfgModel.nContext );
    ConversationDataset datTest( cnvTest, tokTokenizer, cfgModel.nContext );
    // 追加学習でも Adam の移動平均・更新回数は引き継がず、新しく初期化する。
    Adam optAdam( &trnModel, c_cfgTraining.fLearningRate );
    optAdam.init();
    std::mt19937 rngRandom( static_cast<std::mt19937::result_type>( c_cfgTraining.nSeed ) );
    std::filesystem::create_directories( c_strModelDirectory );
    std::ofstream ofsMetrics( c_strModelDirectory + "/metrics.jsonl" );
    if( !ofsMetrics )
    {
        throw std::runtime_error( "Cannot write metrics" );
    }
    stmLog << "vocabulary=" << cfgModel.nVocabulary << " windows=" << datTrain.size()
           << " max_batches=" << c_cfgTraining.nMaxBatches << '\n';
    double dblBest = std::numeric_limits<double>::infinity();
    const auto memoryMode = cu_memory::statistics();
    // 再現に必要な設定と実際のメモリ確保方式を、開始ログに記録する。
    const Json jsnTraining = {
        { "cuda_allocator", memoryMode.pooled ? "pool" : "legacy" },
        { "event", "training_start" }, { "source_model", c_strSource },
        { "optimizer", "Adam newly initialized" }, { "epochs", c_cfgTraining.nEpochs },
        { "batch_size", c_cfgTraining.nBatchSize }, { "learning_rate", c_cfgTraining.fLearningRate },
        { "clip_norm", c_cfgTraining.fClipNorm }, { "shuffle_seed", c_cfgTraining.nSeed },
        { "dropout_seed", cfgModel.nSeed }, { "max_batches", c_cfgTraining.nMaxBatches },
        { "blocks", cfgModel.nBlocks }, { "embedding", cfgModel.nEmbedding },
        { "heads", cfgModel.nHeads }, { "hidden", cfgModel.nHidden },
        { "context", cfgModel.nContext }, { "dropout", cfgModel.fDropout } };
    stmLog << jsnTraining.dump() << std::endl;
    ofsMetrics << jsnTraining.dump() << std::endl;
    // 1つの split を処理する。学習時だけ dropout と重み更新を有効にする。
    auto fnRun = [&]( const ConversationDataset& c_datData, bool isTraining,
                      const std::string& c_strSplit, int nEpoch )
    {
        trnModel.setTraining( isTraining );
        // 学習データだけ毎回並べ替える。評価は元の順序で処理する。
        std::vector<std::size_t> nOrder( c_datData.size() );
        std::iota( nOrder.begin(), nOrder.end(), 0 );
        if( isTraining )
        {
            std::shuffle( nOrder.begin(), nOrder.end(), rngRandom );
        }
        double dblLossSum = 0.0;
        std::size_t nValid = 0;
        std::size_t nPeakUsed = 0;
        std::size_t nBaselineFree = 0;
        std::size_t nTotal = 0;
        _CheckCuda( cudaMemGetInfo( &nBaselineFree, &nTotal ) );
        // 先行する GPU 処理を完了させてから、今回の split の計時を始める。
        _CheckCuda( cudaDeviceSynchronize() );
        const auto clkStart = std::chrono::steady_clock::now();
        int nBatches = 0;
        for( std::size_t nStart = 0; nStart < nOrder.size(); nStart += c_cfgTraining.nBatchSize )
        {
            if( c_cfgTraining.nMaxBatches > 0 && nBatches >= c_cfgTraining.nMaxBatches )
            {
                break;
            }
            // 入力 ID と次トークンの正解 ID をバッチ化し、GPU に転送する。
            const auto batBatch = c_datData.batch( nOrder, nStart, c_cfgTraining.nBatchSize );
            auto spmInputs = std::make_shared<cunMat>( batBatch.nSequence, batBatch.nBatch );
            cunMat mTargets( batBatch.nSequence, batBatch.nBatch );
            spmInputs->copyFromHost( batBatch.nInputs.data(), batBatch.nInputs.size() );
            mTargets.copyFromHost( batBatch.nTargets.data(), batBatch.nTargets.size() );
            // 前回の勾配を消して順伝播する。PAD を除いた平均損失で状態を確認する。
            trnModel.zero_grads();
            auto spmLoss = trnModel.loss( spmInputs, mTargets );
            const float fLoss = spmLoss->_mData.toHost()[0];
            if( !std::isfinite( fLoss ) )
            {
                throw std::runtime_error( "Nonfinite loss" );
            }
            if( isTraining )
            {
                // 逆伝播 → 勾配の大きさ制限 → Adam の順で重みを更新する。
                spmLoss->backward();
                ClipGradients( trnModel, c_cfgTraining.fClipNorm );
                optAdam.update();
            }
            std::size_t nFree = 0;
            _CheckCuda( cudaMemGetInfo( &nFree, &nTotal ) );
            nPeakUsed = std::max( nPeakUsed, nTotal - nFree );
            // バッチごとの平均を有効トークン数で重み付けし、端数バッチも正しく集計する。
            dblLossSum += static_cast<double>( fLoss ) * batBatch.nValid;
            nValid += batBatch.nValid;
            ++nBatches;
        }
        _CheckCuda( cudaDeviceSynchronize() );
        // 直前の同期で GPU の完了を待ち、実際の処理時間と処理量を算出する。
        const double dblSeconds =
            std::chrono::duration<double>( std::chrono::steady_clock::now() - clkStart ).count();
        // プールの予約量は再利用待ち領域も含むため、使用中 bytes と分けて記録する。
        const auto memoryStats = cu_memory::statistics();
        const double dblLoss = dblLossSum / nValid;
        Json jsnMetric = { { "epoch", nEpoch },
                           { "split", c_strSplit },
                           { "loss", dblLoss },
                           { "perplexity", std::exp( dblLoss ) },
                           { "valid_tokens", nValid },
                           { "batches", nBatches },
                           { "seconds", dblSeconds },
                           { "tokens_per_second", nValid / dblSeconds },
                           { "sampled_device_used_bytes", nPeakUsed },
                           { "pool_used_bytes", memoryStats.usedBytes },
                           { "pool_reserved_bytes", memoryStats.reservedBytes },
                           { "baseline_device_used_bytes", nTotal - nBaselineFree },
                           { "max_batches", c_cfgTraining.nMaxBatches } };
        stmLog << jsnMetric.dump() << std::endl;
        ofsMetrics << jsnMetric.dump() << std::endl;
        //
        return dblLoss;
    };
    if( isFinetuning )
    {
        // 追加学習前のモデルも最良候補にする。悪化した場合はこの重みを残す。
        dblBest = fnRun( datValidation, false, "validation", 0 );
        g_saveConversation( trnModel, tokTokenizer, c_strModelDirectory );
    }
    for( int nEpoch = 1; nEpoch <= c_cfgTraining.nEpochs; ++nEpoch )
    {
        fnRun( datTrain, true, "train", nEpoch );
        const double dblValidation = fnRun( datValidation, false, "validation", nEpoch );
        // 検証損失が改善した場合だけ保存し、学習損失だけでモデルを選ばない。
        if( dblValidation < dblBest )
        {
            dblBest = dblValidation;
            g_saveConversation( trnModel, tokTokenizer, c_strModelDirectory );
        }
    }
    // 最後の epoch ではなく、検証で最良だった重みを読み直して test を評価する。
    trnModel.load( ( c_strModelDirectory + "/weights.bin" ).c_str() );
    fnRun( datTest, false, "test_best", c_cfgTraining.nEpochs );
}

} // namespace

// 新規モデルの設定を共通処理へ渡し、最初から学習する。
void
Training(
    const std::string& c_strDataDirectory,
    const std::string& c_strModelDirectory,
    TransformerConfig cfgModel,
    const ConfigTraining& c_cfgTraining,
    std::ostream& stmLog
)
{
    _Prop(
        c_strDataDirectory,
        c_strModelDirectory,
        cfgModel,
        c_cfgTraining,
        stmLog
    );
}

// 保存済みモデルを出発点にする。モデル構成は読み込んだ設定を使用する。
void
Finetuning(
    const std::string& c_strDataDirectory,
    const std::string& c_strSourceModelDirectory,
    const std::string& c_strOutputDirectory,
    const ConfigTraining& c_cfgTraining,
    std::ostream& stmLog
)
{
    if( c_strSourceModelDirectory.empty() )
    {
        throw std::invalid_argument( "Source model directory must not be empty" );
    }
    _Prop(
        c_strDataDirectory,
        c_strOutputDirectory,
        {},
        c_cfgTraining,
        stmLog,
        c_strSourceModelDirectory
    );
}

