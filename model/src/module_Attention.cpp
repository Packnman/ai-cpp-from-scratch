#include "module_Attention.h"
#include "transformer_ops.h"

Attention::Attention( int nEmbeddingSize, int nHeads, float fDropoutProbability,
                      std::uint64_t nDropoutSeed )
    : _nEmbeddingSize( nEmbeddingSize ),
      _nHeads( nHeads ),
      _nHeadSize( nHeads > 0 ? nEmbeddingSize / nHeads : 0 ),
      _fDropoutProbability( fDropoutProbability ),
      _nDropoutSeed( nDropoutSeed ),
      _spmQueryWeight( std::make_shared<Tensor>( nEmbeddingSize, nEmbeddingSize ) ),
      _spmQueryBias( std::make_shared<Tensor>( nEmbeddingSize, 1 ) ),
      _spmKeyWeight( std::make_shared<Tensor>( nEmbeddingSize, nEmbeddingSize ) ),
      _spmKeyBias( std::make_shared<Tensor>( nEmbeddingSize, 1 ) ),
      _spmValueWeight( std::make_shared<Tensor>( nEmbeddingSize, nEmbeddingSize ) ),
      _spmValueBias( std::make_shared<Tensor>( nEmbeddingSize, 1 ) ),
      _spmOutputWeight( std::make_shared<Tensor>( nEmbeddingSize, nEmbeddingSize ) ),
      _spmOutputBias( std::make_shared<Tensor>( nEmbeddingSize, 1 ) ),
      _lnrQuery( _spmQueryWeight.get(), _spmQueryBias.get() ),
      _lnrKey( _spmKeyWeight.get(), _spmKeyBias.get() ),
      _lnrValue( _spmValueWeight.get(), _spmValueBias.get() ),
      _lnrOutput( _spmOutputWeight.get(), _spmOutputBias.get() ),
      _sclScores( _nHeadSize > 0 ? 1.0f / std::sqrt( static_cast<float>( _nHeadSize ) ) : 1.0f ),
      _sftWeights( 1 ),
      _drpAttention( fDropoutProbability, nDropoutSeed )
{
    if( nEmbeddingSize <= 0 || nHeads <= 0 || nEmbeddingSize % nHeads != 0 ||
        !std::isfinite( fDropoutProbability ) )
    {
        throw std::invalid_argument( "Attention: invalid configuration" );
    }
    registerParameter( "query_weight", _spmQueryWeight.get() );
    registerParameter( "query_bias", _spmQueryBias.get() );
    registerParameter( "key_weight", _spmKeyWeight.get() );
    registerParameter( "key_bias", _spmKeyBias.get() );
    registerParameter( "value_weight", _spmValueWeight.get() );
    registerParameter( "value_bias", _spmValueBias.get() );
    registerParameter( "output_weight", _spmOutputWeight.get() );
    registerParameter( "output_bias", _spmOutputBias.get() );
}

Attention::~Attention() = default;

void Attention::_initializeWeight( Tensor& tenWeight, int nFanIn, std::mt19937& rngRandom )
{
    g_initialize( tenWeight, nFanIn, rngRandom );
}

void Attention::init( std::mt19937& rngRandom )
{
    for( auto* lpWeight : { _spmQueryWeight.get(), _spmKeyWeight.get(), _spmValueWeight.get(),
                            _spmOutputWeight.get() } )
    {
        _initializeWeight( *lpWeight, _nEmbeddingSize, rngRandom );
    }
    for( auto* lpBias :
         { _spmQueryBias.get(), _spmKeyBias.get(), _spmValueBias.get(), _spmOutputBias.get() } )
    {
        cuda_fill( lpBias->_mData, 0.0f );
    }
}

std::uint64_t Attention::dropoutCounter() const noexcept { return _nDropoutSeed; }
void Attention::setDropoutCounter( std::uint64_t nCounter ) noexcept { _nDropoutSeed = nCounter; }

TensorPtr Attention::forward( TensorList& spmInputs )
{
    if( spmInputs.size() != 1 || !spmInputs[0] || spmInputs[0]->_mData.dim() != 3 ||
        spmInputs[0]->_mData.size( 0 ) != _nEmbeddingSize )
    {
        throw std::invalid_argument( "Attention: expected [embedding, sequence, batch]" );
    }
    const int nSequence = static_cast<int>( spmInputs[0]->_mData.size( 1 ) );
    const int nBatch = static_cast<int>( spmInputs[0]->_mData.size( 2 ) );
    // 埋め込み軸を head に分割し、Q・K・V の軸を head ごとの行列積に合わせて並べる。
    const std::vector<std::int64_t> nHeads = { _nHeads, _nHeadSize, nSequence, nBatch };
    auto spmQuery = g_permute( g_reshape( _lnrQuery( spmInputs ), nHeads ), { 2, 1, 0, 3 } );
    auto spmKey = g_permute( g_reshape( _lnrKey( spmInputs ), nHeads ), { 1, 2, 0, 3 } );
    auto spmValue = g_permute( g_reshape( _lnrValue( spmInputs ), nHeads ), { 2, 1, 0, 3 } );

    // Scores[q, k, h, b] = dot(Q[q, :, h, b], K[:, k, h, b]) / sqrt(d_head).
    auto spmScores = _sclScores( { _bmmQueryKey( { spmQuery, spmKey } ) } );
    // key <= query だけを許可する因果マスクで、未来の正解トークンの参照を防ぐ。
    auto spmMask = std::make_shared<Tensor>( nSequence, nSequence );
    std::vector<float> fMask( static_cast<std::size_t>( nSequence ) * nSequence );
    for( int nQuery = 0; nQuery < nSequence; ++nQuery )
    {
        for( int nKey = 0; nKey <= nQuery; ++nKey )
        {
            fMask[nQuery * nSequence + nKey] = 1.0f;
        }
    }
    spmMask->_mData.copyFromHost( fMask.data(), fMask.size() );
    // マスク後に key 軸で softmax を取り、各位置が過去のどこを参照するかを重みにする。
    auto spmWeights = _sftWeights( { g_apply<CausalMask>( { spmScores }, spmMask ) } );
    if( isTraining() && _fDropoutProbability > 0.0f )
    {
        // forward ごとに dropout を作り、保持中の古い計算グラフにも固有のマスクを残す。
        spmWeights = g_apply<Dropout>( { spmWeights }, _fDropoutProbability, _nDropoutSeed++ );
    }
    // 参照重みで V を集約し、head を埋め込み軸に結合して出力射影する。
    auto spmAttended = _bmmAttentionValue( { spmWeights, spmValue } );
    auto spmMerged = g_reshape( g_permute( spmAttended, { 2, 1, 0, 3 } ),
                                { _nEmbeddingSize, nSequence, nBatch } );
    return _lnrOutput( { spmMerged } );
}
