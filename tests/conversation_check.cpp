#include "../model/src/transformer_ops.h"
#include "train.h"
#include "validation.h"
#include "cuda_function_IndexCrossEntropy.h"
#include "optimizer.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <limits>
#include <set>
#include <sstream>
#include <unistd.h>

namespace
{
void g_require( bool isCondition, const char* c_lpMessage )
{
    if( !isCondition )
    {
        throw std::runtime_error( c_lpMessage );
    }
}

void g_near( double dblActual, double dblExpected, double dblTolerance, const char* c_lpMessage )
{
    g_require( std::isfinite( dblActual ) && std::abs( dblActual - dblExpected ) <= dblTolerance,
               c_lpMessage );
}

template <class Callable> void g_throws( Callable fnCall )
{
    bool isThrown = false;
    try
    {
        fnCall();
    }
    catch( const std::exception& )
    {
        isThrown = true;
    }
    g_require( isThrown, "Expected exception" );
}

void g_data( const std::filesystem::path& c_pthRoot )
{
    const auto pthSource = c_pthRoot / "source";
    std::filesystem::create_directories( pthSource / "real_persona_chat/dialogues" );
    std::ofstream( pthSource / "VERSION" ) << "fixture-v1\n";
    for( int nId = 0; nId < 40; ++nId )
    {
        nlohmann::json jsnDialogue = { { "dialogue_id", nId },
                                       { "interlocutors", { "Y", "X" } },
                                       { "utterances",
                                         { { { "interlocutor_id", "X" }, { "text", "あ\n😀" } },
                                           { { "interlocutor_id", "Y" }, { "text", "い" } } } },
                                       { "persona", "DO NOT TRAIN ON THIS" } };
        std::ofstream( pthSource / "real_persona_chat/dialogues" /
                       ( std::to_string( nId ) + ".json" ) )
            << jsnDialogue.dump();
    }
    std::filesystem::copy_file( pthSource / "real_persona_chat/dialogues/0.json",
                                pthSource / "real_persona_chat/dialogues/duplicate.json" );
    g_prepareConversations( pthSource.string(), ( c_pthRoot / "prepared" ).string(), "fixture" );
    g_prepareConversations( pthSource.string(), ( c_pthRoot / "again" ).string(), "fixture" );
    std::set<std::int64_t> nIds;
    int nSplitIndex = 0;
    for( const auto* c_lpSplit : { "train", "validation", "test" } )
    {
        const auto cnvSplit = g_readConversations(
            ( c_pthRoot / "prepared" / ( std::string( c_lpSplit ) + ".jsonl" ) ).string() );
        const auto cnvAgain = g_readConversations(
            ( c_pthRoot / "again" / ( std::string( c_lpSplit ) + ".jsonl" ) ).string() );
        g_require( cnvSplit.size() == ( nSplitIndex++ == 0 ? 36 : 2 ), "90/5/5 split counts" );
        for( std::size_t nIndex = 0; nIndex < cnvSplit.size(); ++nIndex )
        {
            g_require( nIds.insert( cnvSplit[nIndex].nId ).second, "Dialogue leakage" );
            g_require( cnvSplit[nIndex].nId == cnvAgain[nIndex].nId, "Nondeterministic split" );
            g_require( cnvSplit[nIndex].uttUtterances[0].nSpeaker == 1,
                       "Speaker mapping by interlocutor ID" );
        }
    }
    TokenConversation tokTokenizer( "あい😀\n" );
    g_require( tokTokenizer.encode( "😀あ未知" ) == TokenIds( { 10, 8, 1, 1 } ),
               "Unicode/UNK encoding" );
    g_require( tokTokenizer.decode( tokTokenizer.encode( "あ😀\n" ) ) == "あ😀\n",
               "Unicode roundtrip" );
    TokenCharacter tokStrict( "あ" );
    g_throws(
        [&]
        {
            tokStrict.encode( "い" );
        } );
    g_throws(
        [&]
        {
            tokTokenizer.encode( std::string( "\xc0\x80", 2 ) );
        } );
    const std::vector<Conversation> cnvTiny = { { 1, { { 0, "あい" }, { 1, "😀" } } },
                                                { 2, { { 1, "あ" } } } };
    ConversationDataset datData( cnvTiny, tokTokenizer, 4 );
    std::vector<std::size_t> nOrder( datData.size() );
    std::iota( nOrder.begin(), nOrder.end(), 0 );
    const auto batFirst = datData.batch( nOrder, 0, 1 );
    g_require( batFirst.nInputs == TokenIds( { 2, 4, 8, 9 } ), "Input positions" );
    g_require( batFirst.nTargets == TokenIds( { 4, 8, 9, 6 } ), "Teacher shift" );
    const auto batSecond = datData.batch( nOrder, 1, 1 );
    g_require( batSecond.nInputs == TokenIds( { 6, 5, 10, 6 } ), "Window stride" );
    g_require( batSecond.nTargets == TokenIds( { 5, 10, 6, 3 } ), "Dialogue ending target" );
    const auto batLast = datData.batch( nOrder, 2, 8 );
    g_require( batLast.nBatch == 1 && batLast.nTargets == TokenIds( { 5, 8, 6, 3 } ),
               "Partial batch and dialogue boundary" );
    ConversationDataset datShort( { { 3, { { 0, "" } } } }, tokTokenizer, 8 );
    const auto batShort = datShort.batch( { 0 }, 0, 8 );
    g_require( batShort.nValid == 3 && batShort.nTargets[3] == 0 && batShort.nInputs[3] == TokenConversation::END && batShort.nInputs[4] == 0,
               "Right padding" );
    std::cout << "data/tokenizer checks passed\n";
}

void g_lossCheck()
{
    auto spmLogits = std::make_shared<Tensor>( std::vector<std::int64_t>{ 3, 2, 1 } );
    const std::vector<float> fLogits = { 1.0f, 9.0f, 2.0f, -4.0f, 3.0f, 2.0f };
    spmLogits->_mData.copyFromHost( fLogits.data(), fLogits.size() );
    cunMat mTargets( 2, 1 );
    std::vector<int> nTargets = { 2, 0 };
    mTargets.copyFromHost( nTargets.data(), nTargets.size() );
    auto spmLoss = g_apply<IndexCrossEntropy>( { spmLogits }, mTargets, 0 );
    nTargets = { 1, 1 };
    mTargets.copyFromHost( nTargets.data(), nTargets.size() );
    g_near( spmLoss->_mData.toHost()[0],
            std::log( std::exp( 1.0 ) + std::exp( 2.0 ) + std::exp( 3.0 ) ) - 3.0, 1e-6,
            "Index loss" );
    spmLoss->backward();
    const auto fGradient = spmLogits->_mGrad.toHost();
    for( int nClass = 0; nClass < 3; ++nClass )
    {
        g_near( fGradient[nClass * 2],
                std::exp( nClass + 1.0 ) / ( std::exp( 1.0 ) + std::exp( 2.0 ) + std::exp( 3.0 ) ) -
                    ( nClass == 2 ),
                1e-6, "Loss gradient" );
        g_near( fGradient[nClass * 2 + 1], 0.0, 0.0, "PAD gradient" );
    }
    nTargets = { 0, 0 };
    mTargets.copyFromHost( nTargets.data(), nTargets.size() );
    auto spmAllPad = g_apply<IndexCrossEntropy>( { spmLogits }, mTargets );
    g_near( spmAllPad->_mData.toHost()[0], 0.0, 0.0, "All PAD loss" );
    cuda_fill( spmLogits->_mGrad, 0.0f );
    spmAllPad->backward();
    for( float fValue : spmLogits->_mGrad.toHost() )
    {
        g_near( fValue, 0.0, 0.0, "All PAD gradient" );
    }
    nTargets = { 3, 0 };
    mTargets.copyFromHost( nTargets.data(), nTargets.size() );
    g_throws(
        [&]
        {
            g_apply<IndexCrossEntropy>( { spmLogits }, mTargets );
        } );
    std::cout << "integer loss checks passed\n";
}

void g_attention()
{
    Attention attModel( 4, 2 );
    std::mt19937 rngRandom( 42 );
    attModel.init( rngRandom );
    attModel.setTraining( false );
    auto spmInput = std::make_shared<Tensor>( std::vector<std::int64_t>{ 4, 3, 2 } );
    std::vector<float> fInput( 24 );
    for( std::size_t nIndex = 0; nIndex < fInput.size(); ++nIndex )
    {
        fInput[nIndex] = std::sin( static_cast<float>( nIndex ) ) * 0.2f;
    }
    spmInput->_mData.copyFromHost( fInput.data(), fInput.size() );
    TensorList spmInputs = { spmInput };
    auto spmOutput = attModel.forward( spmInputs );
    const auto fReference = spmOutput->_mData.toHost();
    auto fChanged = fInput;
    for( int nFeature = 0; nFeature < 4; ++nFeature )
    {
        fChanged[nFeature * 6 + 4] += 2.0f;
        fChanged[nFeature * 6 + 5] -= 2.0f;
    }
    spmInput->_mData.copyFromHost( fChanged.data(), fChanged.size() );
    const auto fFuture = attModel.forward( spmInputs )->_mData.toHost();
    for( int nFeature = 0; nFeature < 4; ++nFeature )
    {
        for( int nPosition = 0; nPosition < 4; ++nPosition )
        {
            g_near( fFuture[nFeature * 6 + nPosition], fReference[nFeature * 6 + nPosition], 1e-6,
                    "Causal attention" );
        }
    }
    spmInput->_mData.copyFromHost( fInput.data(), fInput.size() );
    // Keep the old graph, run a different shape, then backpropagate the old graph.
    TensorList spmShort = { std::make_shared<Tensor>( std::vector<std::int64_t>{ 4, 1, 1 } ) };
    cuda_fill( spmShort[0]->_mData, 0.1f );
    auto spmOther = attModel.forward( spmShort );
    spmOutput->backward();
    const auto fInputGradient = spmInput->_mGrad.toHost();
    auto fnValue = [&]()
    {
        const auto fValues = attModel.forward( spmInputs )->_mData.toHost();
        return std::accumulate( fValues.begin(), fValues.end(), 0.0 );
    };
    const float fEpsilon = 1.0e-3f;
    for( std::size_t nIndex = 0; nIndex < fInput.size(); ++nIndex )
    {
        auto fPerturbed = fInput;
        fPerturbed[nIndex] += fEpsilon;
        spmInput->_mData.copyFromHost( fPerturbed.data(), fPerturbed.size() );
        const double dblPlus = fnValue();
        fPerturbed[nIndex] -= 2.0f * fEpsilon;
        spmInput->_mData.copyFromHost( fPerturbed.data(), fPerturbed.size() );
        const double dblMinus = fnValue();
        g_near( fInputGradient[nIndex], ( dblPlus - dblMinus ) / ( 2 * fEpsilon ), 2e-3,
                "Attention input numerical gradient" );
    }
    spmInput->_mData.copyFromHost( fInput.data(), fInput.size() );
    for( auto* lpParameter : attModel.getParams() )
    {
        const auto fOriginal = lpParameter->_mData.toHost();
        const auto fGradient = lpParameter->_mGrad.toHost();
        for( std::size_t nIndex = 0; nIndex < fOriginal.size(); ++nIndex )
        {
            auto fPerturbed = fOriginal;
            fPerturbed[nIndex] += fEpsilon;
            lpParameter->_mData.copyFromHost( fPerturbed.data(), fPerturbed.size() );
            const double dblPlus = fnValue();
            fPerturbed[nIndex] -= 2.0f * fEpsilon;
            lpParameter->_mData.copyFromHost( fPerturbed.data(), fPerturbed.size() );
            const double dblMinus = fnValue();
            g_near( fGradient[nIndex], ( dblPlus - dblMinus ) / ( 2 * fEpsilon ), 2e-3,
                    "Attention parameter numerical gradient" );
        }
        lpParameter->_mData.copyFromHost( fOriginal.data(), fOriginal.size() );
    }
    std::cout << "attention numerical gradient/causality/graph lifetime checks passed\n";
}

void g_dropoutLifetime()
{
    FeedForward ffdFirst( 4, 8, 0.3f, 42 );
    FeedForward ffdReference( 4, 8, 0.3f, 42 );
    std::mt19937 rngFirst( 42 );
    std::mt19937 rngReference( 42 );
    ffdFirst.init( rngFirst );
    ffdReference.init( rngReference );
    TensorList spmFirst = { std::make_shared<Tensor>( std::vector<std::int64_t>{ 4, 3, 2 } ) };
    TensorList spmReference = { std::make_shared<Tensor>( std::vector<std::int64_t>{ 4, 3, 2 } ) };
    cuda_fill( spmFirst[0]->_mData, 0.2f );
    cuda_fill( spmReference[0]->_mData, 0.2f );
    auto spmOld = ffdFirst.forward( spmFirst );
    TensorList spmShort = { std::make_shared<Tensor>( std::vector<std::int64_t>{ 4, 1, 1 } ) };
    cuda_fill( spmShort[0]->_mData, 0.7f );
    auto spmNew = ffdFirst.forward( spmShort );
    spmOld->backward();
    ffdReference.forward( spmReference )->backward();
    const auto fActual = spmFirst[0]->_mGrad.toHost();
    const auto fExpected = spmReference[0]->_mGrad.toHost();
    for( std::size_t nIndex = 0; nIndex < fActual.size(); ++nIndex )
    {
        g_near( fActual[nIndex], fExpected[nIndex], 1e-6, "Dropout graph-local state" );
    }
}

void g_model( const std::filesystem::path& c_pthRoot )
{
    TokenConversation tokTokenizer( "あ" );
    TransformerConfig cfgModel;
    cfgModel.nVocabulary = tokTokenizer.vocabSize();
    cfgModel.nBlocks = 1;
    cfgModel.nEmbedding = 16;
    cfgModel.nHeads = 2;
    cfgModel.nHidden = 32;
    cfgModel.nContext = 8;
    cfgModel.fDropout = 0.0f;
    Transformer trnModel( cfgModel );
    auto spmInputs = std::make_shared<cunMat>( 4, 1 );
    const TokenIds nInputs = { 2, 4, 7, 6 };
    const TokenIds nTargets = { 4, 7, 6, 3 };
    spmInputs->copyFromHost( nInputs.data(), nInputs.size() );
    cunMat mTargets( 4, 1 );
    mTargets.copyFromHost( nTargets.data(), nTargets.size() );
    Adam optAdam( &trnModel, 0.02f );
    optAdam.init();
    const float fInitial = trnModel.loss( spmInputs, mTargets )->_mData.toHost()[0];
    for( int nStep = 0; nStep < 80; ++nStep )
    {
        trnModel.zero_grads();
        auto spmLoss = trnModel.loss( spmInputs, mTargets );
        spmLoss->backward();
        ClipGradients( trnModel, 1.0f );
        optAdam.update();
    }
    trnModel.setTraining( false );
    const float fFinal = trnModel.loss( spmInputs, mTargets )->_mData.toHost()[0];
    g_require( fFinal < 0.1f && fFinal < fInitial * 0.1f, "Tiny corpus overfit" );
    const auto fBefore = trnModel.forward( spmInputs )->_mData.toHost();
    g_saveConversation( trnModel, tokTokenizer, ( c_pthRoot / "model" ).string() );
    auto bunLoaded = g_loadConversation( ( c_pthRoot / "model" ).string() );
    const auto fAfter = bunLoaded.spModel->forward( spmInputs )->_mData.toHost();
    g_require( fBefore == fAfter, "Saved/reloaded logits differ" );
    for( const auto& c_nmtParameter : bunLoaded.spModel->namedParameters() )
    {
        if( c_nmtParameter.strName == "output_weight" )
        {
            cuda_fill( c_nmtParameter.lpTensor->_mData, 0.0f );
        }
        if( c_nmtParameter.strName == "output_bias" )
        {
            std::vector<float> fBias( cfgModel.nVocabulary, -100.0f );
            fBias[TokenConversation::PAD] = 100.0f;
            fBias[TokenConversation::SPEAKER_A] = 99.0f;
            fBias[TokenConversation::UTTERANCE_END] = 10.0f;
            c_nmtParameter.lpTensor->_mData.copyFromHost( fBias.data(), fBias.size() );
        }
    }
    std::mt19937 rngRandom( 42 );
    ConfigGeneration cfgGeneration;
    cfgGeneration.nTopK = 1;
    g_require( Generate( *bunLoaded.spModel, bunLoaded.tokTokenizer,
                                       { 2, 4, 7, 6, 5 }, rngRandom,
                                       cfgGeneration ) == TokenIds{ 6 },
               "Generation exclusion/stopping" );
    ConfigTraining cfgTraining;
    cfgTraining.nEpochs = 1;
    cfgTraining.nMaxBatches = 1;
    std::ostringstream stmLog;
    Training( ( c_pthRoot / "prepared" ).string(), ( c_pthRoot / "trained" ).string(),
                         cfgModel, cfgTraining, stmLog );
    g_require( stmLog.str().find( "test_best" ) != std::string::npos,
               "Train/validation/best test flow" );
    std::cout << "overfit loss " << fInitial << " -> " << fFinal
              << "; save/reload/generation checks passed\n";
}
void g_finetune( const std::filesystem::path& c_pthRoot )
{
    const auto pthSource = c_pthRoot / "finetune-source";
    const auto pthOutput = c_pthRoot / "finetuned";
    const auto strData = ( c_pthRoot / "prepared" ).string();
    TokenConversation tokTokenizer( "あ" );
    TransformerConfig cfgModel;
    cfgModel.nVocabulary = tokTokenizer.vocabSize();
    cfgModel.nBlocks = 1;
    cfgModel.nEmbedding = 16;
    cfgModel.nHeads = 2;
    cfgModel.nHidden = 32;
    cfgModel.nContext = 8;
    cfgModel.fDropout = 0.1f;
    cfgModel.nSeed = 17;
    Transformer trnModel( cfgModel );
    trnModel.setTraining( false );
    g_saveConversation( trnModel, tokTokenizer, pthSource.string() );
    auto fnRead = []( const std::filesystem::path& pthFile )
    {
        std::ifstream stmFile( pthFile, std::ios::binary );
        return std::string( std::istreambuf_iterator<char>( stmFile ), {} );
    };
    const auto strWeights = fnRead( pthSource / "weights.bin" );
    const auto strManifest = fnRead( pthSource / "manifest.json" );
    auto bunSource = g_loadConversation( pthSource.string() );
    ConversationDataset datValidation( g_readConversations( strData + "/validation.jsonl" ),
                                      bunSource.tokTokenizer, cfgModel.nContext );
    std::vector<std::size_t> nOrder( datValidation.size() );
    std::iota( nOrder.begin(), nOrder.end(), 0 );
    const auto batBatch = datValidation.batch( nOrder, 0, 32 );
    g_require( std::find( batBatch.nInputs.begin(), batBatch.nInputs.end(), TokenConversation::UNK ) !=
                   batBatch.nInputs.end(), "Finetuning dataset must encode unknown characters as UNK" );
    auto spmInputs = std::make_shared<cunMat>( batBatch.nSequence, batBatch.nBatch );
    spmInputs->copyFromHost( batBatch.nInputs.data(), batBatch.nInputs.size() );
    cunMat mTargets( batBatch.nSequence, batBatch.nBatch );
    mTargets.copyFromHost( batBatch.nTargets.data(), batBatch.nTargets.size() );
    const auto fLogits = trnModel.forward( spmInputs )->_mData.toHost();
    g_require( fLogits == bunSource.spModel->forward( spmInputs )->_mData.toHost(),
               "Finetuning initial logits must match source" );
    const double dblInitial = bunSource.spModel->loss( spmInputs, mTargets )->_mData.toHost()[0];
    ConfigTraining cfgTraining;
    cfgTraining.nEpochs = 2;
    cfgTraining.nBatchSize = 32;
    cfgTraining.nMaxBatches = 0;
    cfgTraining.fLearningRate = 0.01f;
    cfgTraining.nSeed = 99;
    std::ostringstream stmLog;
    Finetuning( strData, pthSource.string(), pthOutput.string(), cfgTraining, stmLog );
    std::ifstream stmMetrics( pthOutput / "metrics.jsonl" );
    std::string strLine;
    int nTrain = 0;
    double dblBest = dblInitial;
    bool isBaseline = false;
    while( std::getline( stmMetrics, strLine ) )
    {
        const auto jsnMetric = nlohmann::json::parse( strLine );
        if( jsnMetric.contains( "event" ) )
        {
            g_require( jsnMetric.at( "shuffle_seed" ) == 99 && jsnMetric.at( "dropout_seed" ) == 17,
                       "Shuffle seed must not replace saved dropout seed" );
            continue;
        }
        if( jsnMetric.at( "split" ) == "validation" )
        {
            const double dblLoss = jsnMetric.at( "loss" );
            if( jsnMetric.at( "epoch" ) == 0 )
            {
                g_near( dblLoss, dblInitial, 1e-6, "Validation before any update" );
                isBaseline = true;
            }
            dblBest = std::min( dblBest, dblLoss );
        }
        if( jsnMetric.at( "split" ) == "train" )
        {
            ++nTrain;
            g_require( jsnMetric.at( "batches" ) == 3, "max-batches zero must train all windows" );
        }
    }
    g_require( isBaseline && nTrain == 2 && dblBest < dblInitial, "Additional epochs must improve loss" );
    auto bunUpdated = g_loadConversation( pthOutput.string() );
    g_require( fnRead( pthOutput / "weights.bin" ) != strWeights, "Finetuning must update weights" );
    g_require( fnRead( pthOutput / "manifest.json" ) == strManifest,
               "Finetuning must preserve configuration, vocabulary and special IDs" );
    g_near( bunUpdated.spModel->loss( spmInputs, mTargets )->_mData.toHost()[0], dblBest, 1e-6,
            "Saved model must be the best candidate" );
    g_near( Validation( strData, pthOutput.string(), "validation", 32, 0, stmLog ),
            dblBest, 1e-6, "Standalone validation must match training validation" );
    g_near( Validation( strData, pthOutput.string(), "validation", 1, 0, stmLog ),
            dblBest, 1e-5, "Evaluation must weight variable valid-token counts across batches" );
    g_throws( [&] { Validation( strData, pthOutput.string(), "../train", 32, 0, stmLog ); } );
    g_throws( [&] { Validation( strData, pthOutput.string(), "test", 0, 0, stmLog ); } );
    g_throws( [&] { Validation( strData, pthOutput.string(), "test", 32, -1, stmLog ); } );
    // A subnormal learning rate rounds parameter updates away, exercising baseline retention.
    cfgTraining.fLearningRate = std::numeric_limits<float>::denorm_min();
    cfgTraining.nEpochs = 1;
    cfgTraining.nMaxBatches = 1;
    const auto pthAgain = c_pthRoot / "finetuned-again";
    std::filesystem::create_directories( pthAgain );
    Finetuning( strData, pthOutput.string(), pthAgain.string(), cfgTraining, stmLog );
    auto bunAgain = g_loadConversation( pthAgain.string() );
    g_require( bunAgain.spModel->forward( spmInputs )->_mData.toHost() ==
                   bunUpdated.spModel->forward( spmInputs )->_mData.toHost(),
               "No improvement must retain initial candidate logits" );
    g_require( fnRead( pthSource / "weights.bin" ) == strWeights &&
                   fnRead( pthSource / "manifest.json" ) == strManifest, "Source bundle changed" );
    g_throws( [&] { Finetuning( strData, pthSource.string(), pthSource.string(), cfgTraining, stmLog ); } );
    g_throws( [&] { Finetuning( strData, pthSource.string(), pthOutput.string(), cfgTraining, stmLog ); } );
    const auto pthAlias = c_pthRoot / "source-alias";
    std::filesystem::create_directory_symlink( pthSource, pthAlias );
    g_throws( [&] { Finetuning( strData, pthSource.string(), pthAlias.string(), cfgTraining, stmLog ); } );
    const auto pthFile = c_pthRoot / "output-file";
    std::ofstream( pthFile ) << "keep";
    g_throws( [&] { Finetuning( strData, pthSource.string(), pthFile.string(), cfgTraining, stmLog ); } );
    const auto pthInvalid = c_pthRoot / "invalid-model";
    std::filesystem::create_directories( pthInvalid );
    std::ofstream( pthInvalid / "manifest.json" ) << "{}";
    const auto pthUnused = c_pthRoot / "unused-output";
    g_throws( [&] { Finetuning( strData, pthInvalid.string(), pthUnused.string(), cfgTraining, stmLog ); } );
    g_require( !std::filesystem::exists( pthUnused ), "Invalid source must not create output" );
    std::cout << "finetuning checks passed: validation " << dblInitial << " -> " << dblBest << '\n';
}

} // namespace

int main()
{
    try
    {
        const auto pthRoot = std::filesystem::temp_directory_path() /
                             ( "ai-cpp-conversation-check-" + std::to_string( getpid() ) );
        std::filesystem::create_directories( pthRoot );
        g_data( pthRoot );
        g_lossCheck();
        g_attention();
        g_dropoutLifetime();
        g_model( pthRoot );
        g_finetune( pthRoot );
        std::filesystem::remove_all( pthRoot );
        return 0;
    }
    catch( const std::exception& c_excError )
    {
        std::cerr << c_excError.what() << '\n';
        return 1;
    }
}
