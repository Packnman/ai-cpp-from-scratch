#include "model_transformer.h"
#include "cuda_function_IndexCrossEntropy.h"
#include "transformer_ops.h"

void TransformerConfig::validate() const
{
    if( nVocabulary < 7 || nBlocks <= 0 || nEmbedding <= 0 || nHeads <= 0 ||
        nEmbedding % nHeads != 0 || nHidden <= 0 || nContext <= 0 || !std::isfinite( fDropout ) ||
        fDropout < 0.0f || fDropout >= 1.0f )
    {
        throw std::invalid_argument( "TransformerConfig: invalid dimensions or dropout" );
    }
}

namespace
{
TransformerConfig g_validated( const TransformerConfig& c_cfgConfig )
{
    c_cfgConfig.validate();
    return c_cfgConfig;
}
} // namespace

TransformerBlock::TransformerBlock( int nEmbedding, int nHeads, int nHidden, float fDropout,
                                    std::uint64_t nSeed )
    : _tenGamma1( nEmbedding, 1 ), _tenBeta1( nEmbedding, 1 ), _tenGamma2( nEmbedding, 1 ),
      _tenBeta2( nEmbedding, 1 ), _lynAttention( &_tenGamma1, &_tenBeta1 ),
      _lynFeedForward( &_tenGamma2, &_tenBeta2 ),
      _attAttention( nEmbedding, nHeads, fDropout, nSeed ),
      _ffdFeedForward( nEmbedding, nHidden, fDropout, nSeed + 1 )
{
    // パラメータと子モジュールを登録し、更新・保存・モード切替の対象に含める。
    registerParameter( "gamma1", &_tenGamma1 );
    registerParameter( "beta1", &_tenBeta1 );
    registerParameter( "gamma2", &_tenGamma2 );
    registerParameter( "beta2", &_tenBeta2 );
    registerModule( "attention", &_attAttention );
    registerModule( "feed_forward", &_ffdFeedForward );
}

void TransformerBlock::init( std::mt19937& rngRandom )
{
    // LayerNorm の倍率を1、加算を0にして、学習可能な補正を中立な状態から始める。
    cuda_fill( _tenGamma1._mData, 1.0f );
    cuda_fill( _tenGamma2._mData, 1.0f );
    cuda_fill( _tenBeta1._mData, 0.0f );
    cuda_fill( _tenBeta2._mData, 0.0f );
    _attAttention.init( rngRandom );
    _ffdFeedForward.init( rngRandom );
}

std::vector<std::uint64_t> TransformerBlock::dropoutCounters() const
{
    return { _attAttention.dropoutCounter(), _ffdFeedForward.dropoutCounter() };
}

void TransformerBlock::setDropoutCounters( const std::vector<std::uint64_t>& c_nCounters )
{
    if( c_nCounters.size() != 2 )
        throw std::invalid_argument( "TransformerBlock: invalid dropout state" );
    _attAttention.setDropoutCounter( c_nCounters[0] );
    _ffdFeedForward.setDropoutCounter( c_nCounters[1] );
}

TensorPtr TransformerBlock::forward( TensorList& spmInputs )
{
    // Pre-LN 構成：正規化 → Attention → 入力との残差加算の順に処理する。
    TensorList spmNormalized = { _lynAttention( spmInputs ) };
    auto spmResidual =
        _addResidual( { spmInputs.at( 0 ), _attAttention.forward( spmNormalized ) } );
    // 続いて正規化 → FeedForward を適用し、もう一度残差を加える。
    spmNormalized = { _lynFeedForward( { spmResidual } ) };
    return _addResidual( { spmResidual, _ffdFeedForward.forward( spmNormalized ) } );
}

Transformer::Transformer( const TransformerConfig& c_cfgConfig )
    : _cfgConfig( g_validated( c_cfgConfig ) ),
      _tenTokens( _cfgConfig.nEmbedding, _cfgConfig.nVocabulary ),
      _tenPositions( _cfgConfig.nEmbedding, _cfgConfig.nContext ),
      _tenGamma( _cfgConfig.nEmbedding, 1 ), _tenBeta( _cfgConfig.nEmbedding, 1 ),
      _tenOutputWeight( _cfgConfig.nVocabulary, _cfgConfig.nEmbedding ),
      _tenOutputBias( _cfgConfig.nVocabulary, 1 ), _embTokens( &_tenTokens ),
      _embPositions( &_tenPositions ), _lynFinal( &_tenGamma, &_tenBeta ),
      _lnrOutput( &_tenOutputWeight, &_tenOutputBias )
{
    registerParameter( "tokens", &_tenTokens );
    registerParameter( "positions", &_tenPositions );
    registerParameter( "gamma", &_tenGamma );
    registerParameter( "beta", &_tenBeta );
    registerParameter( "output_weight", &_tenOutputWeight );
    registerParameter( "output_bias", &_tenOutputBias );
    // 同じ seed から同じ初期重みを作り、ブロックごとに dropout の seed を分ける。
    std::mt19937 rngRandom( static_cast<std::mt19937::result_type>( _cfgConfig.nSeed ) );
    g_initialize( _tenTokens, _cfgConfig.nEmbedding, rngRandom );
    g_initialize( _tenPositions, _cfgConfig.nEmbedding, rngRandom );
    g_initialize( _tenOutputWeight, _cfgConfig.nEmbedding, rngRandom );
    cuda_fill( _tenGamma._mData, 1.0f );
    cuda_fill( _tenBeta._mData, 0.0f );
    cuda_fill( _tenOutputBias._mData, 0.0f );
    for( int nBlock = 0; nBlock < _cfgConfig.nBlocks; ++nBlock )
    {
        auto spBlock = std::make_unique<TransformerBlock>( _cfgConfig.nEmbedding, _cfgConfig.nHeads,
                                                           _cfgConfig.nHidden, _cfgConfig.fDropout,
                                                           _cfgConfig.nSeed + 2 * nBlock );
        spBlock->init( rngRandom );
        registerModule( "block" + std::to_string( nBlock ), spBlock.get() );
        _spBlocks.push_back( std::move( spBlock ) );
    }
}

const TransformerConfig& Transformer::config() const
{
    return _cfgConfig;
}

std::vector<std::uint64_t> Transformer::dropoutCounters() const
{
    std::vector<std::uint64_t> result;
    result.reserve( _spBlocks.size() * 2 );
    for( const auto& block : _spBlocks )
    {
        const auto counters = block->dropoutCounters();
        result.insert( result.end(), counters.begin(), counters.end() );
    }
    return result;
}

void Transformer::setDropoutCounters( const std::vector<std::uint64_t>& c_nCounters )
{
    if( c_nCounters.size() != _spBlocks.size() * 2 )
        throw std::invalid_argument( "Transformer: dropout state count mismatch" );
    for( std::size_t i = 0; i < _spBlocks.size(); ++i )
        _spBlocks[i]->setDropoutCounters( { c_nCounters[2 * i], c_nCounters[2 * i + 1] } );
}

TensorPtr Transformer::forward( TensorList& spmInputs )
{
    (void)spmInputs;
    throw std::invalid_argument( "Transformer: use integer ID forward" );
}

TensorPtr Transformer::forward( const std::shared_ptr<const cunMat>& c_spmIds )
{
    if( !c_spmIds || c_spmIds->dim() != 2 || c_spmIds->size( 0 ) <= 0 ||
        c_spmIds->size( 0 ) > _cfgConfig.nContext || c_spmIds->size( 1 ) <= 0 )
    {
        throw std::invalid_argument( "Transformer: expected IDs [1..context, positive batch]" );
    }
    // Snapshot indices: caller edits cannot change an outstanding graph.
    auto spmIds = std::make_shared<cunMat>( *c_spmIds );
    // ID 配列 [系列, バッチ] の各列に共通の位置番号 0, 1, ... を用意する。
    auto spmPositions = std::make_shared<cunMat>( c_spmIds->shape() );
    std::vector<std::int32_t> nPositions( spmPositions->numel() );
    for( std::size_t nIndex = 0; nIndex < nPositions.size(); ++nIndex )
    {
        nPositions[nIndex] = static_cast<int>( nIndex / c_spmIds->size( 1 ) );
    }
    spmPositions->copyFromHost( nPositions.data(), nPositions.size() );
    // トークンの意味と系列内の位置を加算し、[埋め込み, 系列, バッチ] の特徴を作る。
    auto spmResult = _addPositions( { _embTokens( spmIds ), _embPositions( spmPositions ) } );
    for( auto& spBlock : _spBlocks )
    {
        TensorList spmInputs = { spmResult };
        spmResult = spBlock->forward( spmInputs );
    }
    // 最終正規化の後、各位置の特徴を語彙ごとの未正規化スコア（logits）に変換する。
    return _lnrOutput( { _lynFinal( { spmResult } ) } );
}

TensorPtr Transformer::loss( const std::shared_ptr<const cunMat>& c_spmIds,
                             const cunMat& c_mTargets, int nPadId )
{
    // 正解を one-hot 化せず ID のまま照合し、PAD を除いた交差エントロピーを求める。
    return g_apply<IndexCrossEntropy>( { forward( c_spmIds ) }, c_mTargets, nPadId );
}
