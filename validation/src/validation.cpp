#include "validation.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <nlohmann/json.hpp>

namespace
{
// 生成中だけ評価モードにし、正常終了・例外のどちらでも元のモードに戻す。
struct ModeRestore
{
    Module& mdlModel;
    bool isTraining;
    ~ModeRestore()
    {
        mdlModel.setTraining( isTraining );
    }
};

} // namespace

// 直近の会話履歴から次の ID を1個ずつ選び、終端または長さ上限まで生成する。
TokenIds
Generate(
    Transformer& trnModel,
    const TokenConversation& c_tokTokenizer,
    const TokenIds& c_nHistory,
    std::mt19937& rngRandom,
    const ConfigGeneration& c_cfgGeneration
)
{
    if( c_nHistory.empty() || trnModel.config().nVocabulary != c_tokTokenizer.vocabSize() ||
        !std::isfinite( c_cfgGeneration.fTemperature ) || c_cfgGeneration.fTemperature <= 0.0f ||
        c_cfgGeneration.nTopK <= 0 || c_cfgGeneration.nMaxTokens <= 0 ||
        c_cfgGeneration.nContext < 0 || c_cfgGeneration.nContext > trnModel.config().nContext )
    {
        throw std::invalid_argument( "Invalid generation configuration or history" );
    }
    ModeRestore modRestore{ trnModel, trnModel.isTraining() };
    trnModel.setTraining( false );
    const int nContext = c_cfgGeneration.nContext == 0 ? trnModel.config().nContext : c_cfgGeneration.nContext;
    TokenIds nHistory( c_nHistory.end() - std::min( c_nHistory.size(), static_cast<std::size_t>( nContext ) ),
                       c_nHistory.end() );
    TokenIds nGenerated;
    for( int nStep = 0; nStep < c_cfgGeneration.nMaxTokens; ++nStep )
    {
        // 文脈長を超えた古い履歴を除き、直近のトークンだけをモデルへ渡す。
        const int nSequence = static_cast<int>(
            std::min( nHistory.size(),
                      static_cast<std::size_t>( nContext ) ) );
        auto spmInput = std::make_shared<cunMat>( nSequence, 1 );
        spmInput->copyFromHost( nHistory.data() + nHistory.size() - nSequence, nSequence );
        const auto fLogits = trnModel.forward( spmInput )->_mData.toHost();
        // 通常文字と終端だけを候補にし、PAD や話者 ID が応答本文に出るのを防ぐ。
        std::vector<int> nCandidates = { TokenConversation::END, TokenConversation::UTTERANCE_END };
        for( int nId = TokenConversation::SPECIAL_COUNT; nId < c_tokTokenizer.vocabSize(); ++nId )
        {
            nCandidates.push_back( nId );
        }
        // logits は [語彙, 系列, バッチ]。バッチ1の末尾位置が次トークンの予測になる。
        auto fnLogit = [&]( int nId )
        {
            const float fValue = fLogits[nId * nSequence + nSequence - 1];
            if( !std::isfinite( fValue ) )
            {
                throw std::runtime_error( "Nonfinite generation logits" );
            }
            return fValue;
        };
        // スコア上位 top-k に候補を絞る。
        std::sort( nCandidates.begin(), nCandidates.end(),
                   [&]( int nA, int nB )
                   {
                       return fnLogit( nA ) > fnLogit( nB );
                   } );
        nCandidates.resize(
            std::min( nCandidates.size(), static_cast<std::size_t>( c_cfgGeneration.nTopK ) ) );
        // 最大値を引いて指数計算の桁あふれを防ぎ、temperature で選択の偏りを調整する。
        std::vector<double> dblWeights;
        const float fMaximum = fnLogit( nCandidates[0] );
        for( int nId : nCandidates )
        {
            dblWeights.push_back( std::exp( static_cast<double>( fnLogit( nId ) - fMaximum ) /
                                            c_cfgGeneration.fTemperature ) );
        }
        // 未正規化の重みから抽選し、選んだ ID を次回予測の履歴にも追加する。
        std::discrete_distribution<std::size_t> dstSample( dblWeights.begin(), dblWeights.end() );
        const int nNext = nCandidates[dstSample( rngRandom )];
        nGenerated.push_back( nNext );
        nHistory.push_back( nNext );
        if( nHistory.size() > static_cast<std::size_t>( nContext ) ) nHistory.erase( nHistory.begin() );
        if( nNext == TokenConversation::END || nNext == TokenConversation::UTTERANCE_END )
        {
            break;
        }
    }
    return nGenerated;
}

// 保存モデルを指定 split で評価し、有効トークンあたりの損失と perplexity を出す。
double Validation(
    const std::string& c_strDataDirectory,
    const std::string& c_strModelDirectory,
    const std::string& c_strSplit,
    int nBatchSize,
    int nMaxBatches,
    std::ostream& stmLog
)
{
    if( nBatchSize <= 0 || nMaxBatches < 0 ||
        ( c_strSplit != "validation" && c_strSplit != "test" ) )
    {
        throw std::invalid_argument( "Expected positive batch, nonnegative max-batches and validation/test split" );
    }
    // 保存時と同じ語彙・モデル構成を使う。読み込んだモデルは評価モードになる。
    auto bunModel = g_loadConversation( c_strModelDirectory );
    const auto cnvData = g_readConversations( c_strDataDirectory + "/" + c_strSplit + ".jsonl" );
    if( cnvData.empty() )
    {
        throw std::invalid_argument( "Evaluation split must be nonempty" );
    }
    ConversationDataset datData( cnvData, bunModel.tokTokenizer, bunModel.spModel->config().nContext );
    std::vector<std::size_t> nOrder( datData.size() );
    std::iota( nOrder.begin(), nOrder.end(), 0 );
    double dblLossSum = 0.0;
    std::size_t nValid = 0;
    int nBatches = 0;
    for( std::size_t nStart = 0; nStart < nOrder.size(); nStart += nBatchSize )
    {
        if( nMaxBatches > 0 && nBatches >= nMaxBatches )
        {
            break;
        }
        const auto batBatch = datData.batch( nOrder, nStart, nBatchSize );
        auto spmInputs = std::make_shared<cunMat>( batBatch.nSequence, batBatch.nBatch );
        cunMat mTargets( batBatch.nSequence, batBatch.nBatch );
        spmInputs->copyFromHost( batBatch.nInputs.data(), batBatch.nInputs.size() );
        mTargets.copyFromHost( batBatch.nTargets.data(), batBatch.nTargets.size() );
        // 順伝播だけを実行する。逆伝播と optimizer の更新は行わない。
        const float fLoss = bunModel.spModel->loss( spmInputs, mTargets )->_mData.toHost()[0];
        if( !std::isfinite( fLoss ) )
        {
            throw std::runtime_error( "Nonfinite evaluation loss" );
        }
        // PAD を除いたトークン数で重み付けし、split 全体の平均損失に集約する。
        dblLossSum += static_cast<double>( fLoss ) * batBatch.nValid;
        nValid += batBatch.nValid;
        ++nBatches;
    }
    if( nValid == 0 )
    {
        throw std::invalid_argument( "Evaluation split has no valid tokens" );
    }
    const double dblLoss = dblLossSum / nValid;
    const nlohmann::json jsnMetric = {
        { "split", c_strSplit },
        { "loss", dblLoss },
        { "perplexity", std::exp( dblLoss ) },
        { "valid_tokens", nValid },
        { "batches", nBatches },
        { "batch_size", nBatchSize },
        { "max_batches", nMaxBatches },
        { "source_model", c_strModelDirectory }
    };
    stmLog << jsnMetric.dump() << std::endl;
    //
    return dblLoss;
}
