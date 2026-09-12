#include "cuda_function_Linear.h"
#include "cuda_tensor.h"

#include <climits>
#include <stdexcept>

namespace
{
int g_positions( const TensorList& c_spmInputs, const Tensor* c_lpWeight, const Tensor* c_lpBias )
{
    if( c_spmInputs.size() != 1 || !c_spmInputs[0] || !c_lpWeight || !c_lpBias )
    {
        throw std::invalid_argument( "Linear: input and parameters are required" );
    }
    const auto& c_mInput = c_spmInputs[0]->_mData;
    if( c_mInput.dim() < 2 || c_mInput.size( 0 ) <= 0 || c_lpWeight->_mData.dim() != 2 ||
        c_lpWeight->_mData.size( 1 ) != c_mInput.size( 0 ) ||
        c_lpBias->_mData.shape() != std::vector<std::int64_t>{ c_lpWeight->_mData.size( 0 ), 1 } ||
        !c_mInput.isContiguous() || c_mInput.numel() == 0 || c_mInput.numel() > INT_MAX )
    {
        throw std::invalid_argument( "Linear: invalid feature shape or storage" );
    }
    return static_cast<int>( c_mInput.numel() / c_mInput.size( 0 ) );
}
} // namespace

Linear::Linear( Tensor* lpWeight, Tensor* lpBias ) : _lpmWeight( lpWeight ), _lpmBias( lpBias ) {}

Linear::~Linear() = default;

TensorList Linear::forward( const TensorList& c_spmInputs )
{
    const int nPositions = g_positions( c_spmInputs, _lpmWeight, _lpmBias );
    auto nShape = c_spmInputs[0]->_mData.shape();
    const auto mInput = c_spmInputs[0]->_mData.reshape( { nShape[0], nPositions } );
    nShape[0] = _lpmWeight->_mData.size( 0 );
    auto spmResult = std::make_shared<Tensor>( nShape );
    auto mOutput = spmResult->_mData.reshape( { nShape[0], nPositions } );
    cufMat mOnes( 1, nPositions );
    cuda_fill( mOnes, 1.0f );

    // Y[f, t, b] = W[f, d] X[d, t, b] + bias[f].
    cuda_gemm( mOutput, _lpmWeight->_mData, mInput );
    cuda_gemm( mOutput, _lpmBias->_mData, mOnes, false, false, 1.0f, 1.0f );
    return { spmResult };
}

void Linear::backward( const TensorGradList& c_lpmOutputGrads, const TensorList& c_spmInputs,
                       const TensorList& c_spmOutputs )
{
    const int nPositions = g_positions( c_spmInputs, _lpmWeight, _lpmBias );
    const auto& c_mGrad = requireSingleOutputGrad( c_lpmOutputGrads, "Linear" );
    if( c_spmOutputs.size() != 1 || !c_spmOutputs[0] ||
        c_mGrad.shape() != c_spmOutputs[0]->_mData.shape() )
    {
        throw std::invalid_argument( "Linear: gradient shape mismatch" );
    }
    auto mInput = c_spmInputs[0]->_mData.reshape( { _lpmWeight->_mData.size( 1 ), nPositions } );
    auto mInputGrad = c_spmInputs[0]->_mGrad.reshape( mInput.shape() );
    auto mGrad = c_mGrad.reshape( { _lpmWeight->_mData.size( 0 ), nPositions } );
    cufMat mOnes( 1, nPositions );
    cuda_fill( mOnes, 1.0f );

    // dX += W^T dY; dW += dY X^T; db += sum_positions(dY).
    cuda_gemm( mInputGrad, _lpmWeight->_mData, mGrad, true, false, 1.0f, 1.0f );
    cuda_gemm( _lpmWeight->_mGrad, mGrad, mInput, false, true, 1.0f, 1.0f );
    cuda_gemm( _lpmBias->_mGrad, mGrad, mOnes, false, true, 1.0f, 1.0f );
}
