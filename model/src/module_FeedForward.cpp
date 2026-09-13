#include "module_FeedForward.h"
#include "transformer_ops.h"

FeedForward::FeedForward( int nEmbeddingSize, int nHiddenSize, float fDropoutProbability,
                          std::uint64_t nDropoutSeed )
    : _nEmbeddingSize( nEmbeddingSize ),
      _nHiddenSize( nHiddenSize ),
      _fDropoutProbability( fDropoutProbability ),
      _nDropoutSeed( nDropoutSeed ),
      _spmWeight1( std::make_shared<Tensor>( nHiddenSize, nEmbeddingSize ) ),
      _spmBias1( std::make_shared<Tensor>( nHiddenSize, 1 ) ),
      _spmWeight2( std::make_shared<Tensor>( nEmbeddingSize, nHiddenSize ) ),
      _spmBias2( std::make_shared<Tensor>( nEmbeddingSize, 1 ) ),
      _lnrInput( _spmWeight1.get(),_spmBias1.get() ),
      _lnrOutput( _spmWeight2.get(), _spmBias2.get() ),
      _drpOutput( fDropoutProbability, nDropoutSeed )
{
    if( nEmbeddingSize <= 0 || nHiddenSize <= 0 || !std::isfinite( fDropoutProbability ) )
    {
        throw std::invalid_argument( "FeedForward: invalid configuration" );
    }
    registerParameter( "weight1", _spmWeight1.get() );
    registerParameter( "bias1", _spmBias1.get() );
    registerParameter( "weight2", _spmWeight2.get() );
    registerParameter( "bias2", _spmBias2.get() );
}

FeedForward::~FeedForward() = default;

void FeedForward::_initializeWeight( Tensor& tenWeight, int nFanIn, std::mt19937& rngRandom )
{
    g_initialize( tenWeight, nFanIn, rngRandom );
}

void FeedForward::init( std::mt19937& rngRandom )
{
    _initializeWeight( *_spmWeight1, _nEmbeddingSize, rngRandom );
    _initializeWeight( *_spmWeight2, _nHiddenSize, rngRandom );
    cuda_fill( _spmBias1->_mData, 0.0f );
    cuda_fill( _spmBias2->_mData, 0.0f );
}

std::uint64_t FeedForward::dropoutCounter() const noexcept { return _nDropoutSeed; }
void FeedForward::setDropoutCounter( std::uint64_t nCounter ) noexcept { _nDropoutSeed = nCounter; }

TensorPtr FeedForward::forward( TensorList& spmInputs )
{
    // 各位置に共通の変換を適用する：埋め込み → 隠れ層 → GELU → 元の埋め込み幅。
    auto spmResult = _lnrOutput( { _gelActivation( { _lnrInput( spmInputs ) } ) } );
    if( isTraining() && _fDropoutProbability > 0.0f )
    {
        // 学習時のみ dropout を適用し、各 forward のマスクを計算グラフごとに保持する。
        spmResult = g_apply<Dropout>( { spmResult }, _fDropoutProbability, _nDropoutSeed++ );
    }
    return spmResult;
}
