#include "cuda_function_IndexCrossEntropy.h"
#include "cuda_tensor.h"
#include <climits>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

namespace
{
__global__ void g_loss( const float* c_lpLogits, const int* c_lpTargets, float* lpProbabilities,
                        float* lpLoss, int nVocabulary, int nPositions, int nPad, int nValid )
{
    __shared__ float fReduction[128];
    const int nPosition = blockIdx.x;
    const int nThread = threadIdx.x;
    float fMaximum = -INFINITY;
    for( int nToken = nThread; nToken < nVocabulary; nToken += 128 )
    {
        fMaximum = fmaxf( fMaximum, c_lpLogits[nToken * nPositions + nPosition] );
    }
    fReduction[nThread] = fMaximum;
    __syncthreads();
    for( int nStride = 64; nStride > 0; nStride /= 2 )
    {
        if( nThread < nStride )
        {
            fReduction[nThread] = fmaxf( fReduction[nThread], fReduction[nThread + nStride] );
        }
        __syncthreads();
    }
    fMaximum = fReduction[0];
    __syncthreads();
    float fSum = 0.0f;
    for( int nToken = nThread; nToken < nVocabulary; nToken += 128 )
    {
        fSum += expf( c_lpLogits[nToken * nPositions + nPosition] - fMaximum );
    }
    fReduction[nThread] = fSum;
    __syncthreads();
    for( int nStride = 64; nStride > 0; nStride /= 2 )
    {
        if( nThread < nStride )
        {
            fReduction[nThread] += fReduction[nThread + nStride];
        }
        __syncthreads();
    }
    fSum = fReduction[0];
    for( int nToken = nThread; nToken < nVocabulary; nToken += 128 )
    {
        lpProbabilities[nToken * nPositions + nPosition] =
            expf( c_lpLogits[nToken * nPositions + nPosition] - fMaximum ) / fSum;
    }
    if( nThread == 0 && c_lpTargets[nPosition] != nPad && nValid > 0 )
    {
        atomicAdd( lpLoss, ( logf( fSum ) + fMaximum -
                             c_lpLogits[c_lpTargets[nPosition] * nPositions + nPosition] ) /
                               nValid );
    }
}

__global__ void g_gradient( float* lpGradient, const float* c_lpProbabilities,
                            const int* c_lpTargets, const float* c_lpUpstream, int nElements,
                            int nPositions, int nPad, int nValid )
{
    const int nIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if( nIndex < nElements && nValid > 0 && c_lpTargets[nIndex % nPositions] != nPad )
    {
        const float fTarget = c_lpTargets[nIndex % nPositions] == nIndex / nPositions ? 1.0f : 0.0f;
        lpGradient[nIndex] += c_lpUpstream[0] * ( c_lpProbabilities[nIndex] - fTarget ) / nValid;
    }
}

void g_checkLaunch()
{
    const auto cudStatus = cudaGetLastError();
    if( cudStatus != cudaSuccess )
    {
        throw std::runtime_error( cudaGetErrorString( cudStatus ) );
    }
}
} // namespace

IndexCrossEntropy::IndexCrossEntropy( const cunMat& c_mTargets, int nPadId )
    : _mTargets( c_mTargets ), _nPadId( nPadId ), _nValid( 0 )
{
    for( int nTarget : _mTargets.toHost() )
    {
        if( nTarget != _nPadId )
        {
            ++_nValid;
        }
    }
}

TensorList IndexCrossEntropy::forward( const TensorList& c_spmInputs )
{
    if( _isUsed || c_spmInputs.size() != 1 || !c_spmInputs[0] )
    {
        throw std::invalid_argument(
            "IndexCrossEntropy: one input and a fresh operation are required" );
    }
    const auto& c_mLogits = c_spmInputs[0]->_mData;
    auto nShape = c_mLogits.shape();
    if( nShape.size() < 2 || nShape[0] <= 0 || c_mLogits.numel() == 0 ||
        c_mLogits.numel() > INT_MAX || !c_mLogits.isContiguous() )
    {
        throw std::invalid_argument( "IndexCrossEntropy: invalid logits" );
    }
    const int nVocabulary = static_cast<int>( nShape[0] );
    nShape.erase( nShape.begin() );
    if( nShape != _mTargets.shape() )
    {
        throw std::invalid_argument( "IndexCrossEntropy: target shape mismatch" );
    }
    for( int nTarget : _mTargets.toHost() )
    {
        if( nTarget != _nPadId && ( nTarget < 0 || nTarget >= nVocabulary ) )
        {
            throw std::out_of_range( "IndexCrossEntropy: target ID out of range" );
        }
    }
    _isUsed = true;
    _mProbabilities = cufMat( c_mLogits.shape() );
    auto spmLoss = std::make_shared<Tensor>( 1, 1 );
    cuda_fill( spmLoss->_mData, 0.0f );
    g_loss<<<static_cast<int>( _mTargets.numel() ), 128>>>(
        c_mLogits.data(), _mTargets.data(), _mProbabilities.data(), spmLoss->_mData.data(),
        nVocabulary, static_cast<int>( _mTargets.numel() ), _nPadId, _nValid );
    g_checkLaunch();
    return { spmLoss };
}

void IndexCrossEntropy::backward( const TensorGradList& c_lpmOutputGrads,
                                  const TensorList& c_spmInputs, const TensorList& c_spmOutputs )
{
    (void)c_spmOutputs;
    const auto& c_mGrad = requireSingleOutputGrad( c_lpmOutputGrads, "IndexCrossEntropy" );
    if( !_isUsed || c_mGrad.numel() != 1 || c_spmInputs.size() != 1 || !c_spmInputs[0] ||
        c_spmInputs[0]->_mGrad.shape() != _mProbabilities.shape() )
    {
        throw std::invalid_argument( "IndexCrossEntropy: invalid backward state" );
    }
    const int nElements = static_cast<int>( _mProbabilities.numel() );
    g_gradient<<<( nElements + 255 ) / 256, 256>>>(
        c_spmInputs[0]->_mGrad.data(), _mProbabilities.data(), _mTargets.data(), c_mGrad.data(),
        nElements, static_cast<int>( _mTargets.numel() ), _nPadId, _nValid );
    g_checkLaunch();
}
