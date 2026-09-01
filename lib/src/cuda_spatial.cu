#include "cuda_matrix.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <stdexcept>
#include <string>

__global__ void kernel_Conv2D_im2col(float* lpfColumns,const float* c_lpfInput,int nChannels,int nInputHeight,int nInputWidth, int nKernelSize, int nStride,int nPadding,int nOutputHeight,int nOutputWidth,int nBatch,int nSize);
__global__ void kernel_Conv2D_pack(float* lpfOutput,const float* c_lpfGemm,const float* c_lpfBias,int nOutputChannels,int nPositions,int nBatch,int nSize);
__global__ void kernel_Conv2D_unpack(float* lpfGemmGrad,const float* c_lpfOutputGrad,int nOutputChannels,int nPositions,int nSize);
__global__ void kernel_Conv2D_col2im(float* lpfInputGrad,const float* c_lpfColumnGrad,int nChannels,int nInputHeight,int nInputWidth,int nKernelSize,int nStride,int nPadding,int nOutputHeight,int nOutputWidth,int nSize);
__global__ void kernel_Conv2D_bias(float* lpfBiasGrad,const float* c_lpfGemmGrad,int nChannels,int nSize);
__global__ void kernel_MaxPool2D_forward(float* lpfOutput,const float* c_lpfInput,int nChannels,int nInputHeight,int nInputWidth,int nKernelSize,int nStride,int nOutputWidth,int nPositions,int nSize);
__global__ void kernel_MaxPool2D_backward(float* lpfInputGrad,const float* c_lpfInput,const float* c_lpfOutputGrad,int nChannels,int nInputHeight,int nInputWidth,int nKernelSize,int nStride,int nOutputWidth,int nPositions,int nSize);


void cuda_Conv2D_im2col(
    cuMat& mResult,
    const cuMat& c_mInput,
    int nInputChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nPadding,
    int nOutputHeight,
    int nOutputWidth
)
{
    int nSize       =mResult._nRows * mResult._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Conv2D_im2col<<<nBlocks,nThreads>>>(
        mResult._lpfDevice,
        c_mInput._lpfDevice,
        nInputChannels,
        nInputHeight,
        nInputWidth,
        nKernelSize,
        nStride,
        nPadding,
        nOutputHeight,
        nOutputWidth,
        c_mInput._nCols,
        nSize
    );
    //
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "cuda_Conv2D_im2col: " )
            + cudaGetErrorString( cudError )
        );
    }
}
void cuda_Conv2D_pack_output(
    cuMat& mResult,
    const cuMat& c_mGemm,
    const cuMat& c_mBias,
    int nOutputHeight,
    int nOutputWidth
)
{
    int nSize       =mResult._nRows * mResult._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Conv2D_pack<<<nBlocks, nThreads>>>(
        mResult._lpfDevice,
        c_mGemm._lpfDevice,
        c_mBias._lpfDevice,
        c_mGemm._nRows,
        nOutputHeight*nOutputWidth,
        mResult._nCols,
        nSize
    );
    //
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "cuda_Conv2D_pack_output: " )
            + cudaGetErrorString( cudError )
        );
    }
}
void cuda_Conv2D_unpack_grad( cuMat& mResult, const cuMat& c_mOutputGrad, int nOutputChannels,
                              int nOutputHeight, int nOutputWidth )
{
    int nSize       =mResult._nRows * mResult._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Conv2D_unpack<<<nBlocks, nThreads>>>(
        mResult._lpfDevice,
        c_mOutputGrad._lpfDevice,
        nOutputChannels, nOutputHeight * nOutputWidth,
        nSize
    );
    //
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "cuda_Conv2D_unpack_grad: " )
            + cudaGetErrorString( cudError )
        );
    }
}
void cuda_Conv2D_col2im(
    cuMat& mResult,
    const cuMat& c_mColumnGrad,
    int nInputChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nPadding,
    int nOutputHeight,
    int nOutputWidth
)
{
    int nSize       =mResult._nRows * mResult._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Conv2D_col2im<<<nBlocks, nThreads>>>(
        mResult._lpfDevice,
        c_mColumnGrad._lpfDevice,
        nInputChannels,
        nInputHeight,
        nInputWidth,
        nKernelSize,
        nStride,
        nPadding,
        nOutputHeight,
        nOutputWidth,
        nSize
    );
    //
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "cuda_Conv2D_col2im: " )
            + cudaGetErrorString( cudError )
        );
    }
}
void cuda_Conv2D_bias_backward(
    cuMat& mResult,
    const cuMat& c_mGemmGrad
)
{
    int nSize       =mResult._nRows * mResult._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Conv2D_bias<<<nBlocks, nThreads>>>(
        mResult._lpfDevice,
        c_mGemmGrad._lpfDevice,
        c_mGemmGrad._nRows,
        nSize
    );
    //
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "cuda_Conv2D_bias_backward: " )
            + cudaGetErrorString( cudError )
        );
    }
}
void cuda_MaxPool2D_forward(
    cuMat& mResult,
    const cuMat& c_mInput,
    int nChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nOutputHeight,
    int nOutputWidth
)
{
    int nSize       =mResult._nRows * mResult._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_MaxPool2D_forward<<<nBlocks, nThreads>>>(
        mResult._lpfDevice,
        c_mInput._lpfDevice,
        nChannels,
        nInputHeight,
        nInputWidth,
        nKernelSize,
        nStride,
        nOutputWidth,
        nOutputHeight*nOutputWidth,
        nSize
    );
    //
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "cuda_MaxPool2D_forward: " )
            + cudaGetErrorString( cudError )
        );
    }
}
void cuda_MaxPool2D_backward(
    cuMat& mResult,
    const cuMat& c_mInput,
    const cuMat& c_mOutputGrad,
    int nChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nOutputHeight,
    int nOutputWidth
)
{
    int nSize       =mResult._nRows * mResult._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_MaxPool2D_backward<<<nBlocks, nThreads>>>(
        mResult._lpfDevice,
        c_mInput._lpfDevice,
        c_mOutputGrad._lpfDevice,
        nChannels,
        nInputHeight,
        nInputWidth,
        nKernelSize,
        nStride,
        nOutputWidth,
        nOutputHeight*nOutputWidth,
        nSize
    );
    //
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "cuda_MaxPool2D_backward: " )
            + cudaGetErrorString( cudError )
        );
    }
}


//

__global__ void kernel_Conv2D_im2col(
    float* lpfColumns,
    const float* c_lpfInput,
    int nChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nPadding,
    int nOutputHeight,
    int nOutputWidth,
    int nBatch,
    int nSize
)
{
    const int nIndex    =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex<nSize )
    {
        // p = (k_y*K+k_x)*C_in+c_i, q = n*(H_out*W_out)+o
        // C[p,q] = X[n,o_y*S+k_y-P,o_x*S+k_x-P,c_i]
        const int nPatchSize    = nKernelSize * nKernelSize * nChannels;
        const int nPatchIndex   = nIndex % nPatchSize;
        const int nColumn       = nIndex / nPatchSize;
        const int nPosition     = nColumn % ( nOutputHeight * nOutputWidth );
        const int nSample       = nColumn / ( nOutputHeight * nOutputWidth );
        const int nChannel      = nPatchIndex % nChannels;
        const int nKernelPixel  = nPatchIndex / nChannels;
        const int nKernelY      = nKernelPixel / nKernelSize;
        const int nKernelX      = nKernelPixel % nKernelSize;
        const int nOutputY      = nPosition / nOutputWidth;
        const int nOutputX      = nPosition % nOutputWidth;
        const int nInputY       = nOutputY * nStride + nKernelY - nPadding;
        const int nInputX       = nOutputX * nStride + nKernelX - nPadding;

        lpfColumns[nIndex]  =
            ( (nInputY>=0)&&(nInputY<nInputHeight)&&(nInputX>=0)&&(nInputX<nInputWidth) )
                ? c_lpfInput[nSample * ( nInputHeight * nInputWidth * nChannels ) +
                            ( nInputY * nInputWidth + nInputX ) * nChannels + nChannel]
                : 0.0f;
    }
}

__global__ void kernel_Conv2D_pack(
    float* lpfOutput,
    const float* c_lpfGemm,
    const float* c_lpfBias,
    int nOutputChannels,
    int nPositions,
    int nBatch,
    int nSize
)
{
    const int nIndex    =blockIdx.x * blockDim.x + threadIdx.x;
    
    if( nIndex >= nSize )
    {
        return;
    }
    const int nRow      = nIndex % ( nPositions * nOutputChannels );
    const int nSample   = nIndex / ( nPositions * nOutputChannels );
    const int nChannel  = nRow % nOutputChannels;
    const int nPosition = nRow / nOutputChannels;
    // Y[n,o,c_o] = (W*C)[c_o,n*(H_out*W_out)+o] + b[c_o]
    lpfOutput[nIndex] =
        c_lpfGemm[( nSample * nPositions + nPosition ) * nOutputChannels + nChannel] +
        c_lpfBias[nChannel];
}

__global__ void kernel_Conv2D_unpack(
    float* lpfGemmGrad,
    const float* c_lpfOutputGrad,
    int nOutputChannels,
    int nPositions,
    int nSize
)
{
    const int nIndex    =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex >= nSize )
    {
        return;
    }
    const int nChannel  = nIndex % nOutputChannels;
    const int nColumn   = nIndex / nOutputChannels;
    const int nSample   = nColumn / nPositions;
    const int nPosition = nColumn % nPositions;
    // G[c_o,n*(H_out*W_out)+o] = dL/dY[n,o,c_o]
    lpfGemmGrad[nIndex] = c_lpfOutputGrad[nSample * ( nPositions * nOutputChannels ) +
                                          nPosition * nOutputChannels + nChannel];
}

__global__ void kernel_Conv2D_col2im(
    float* lpfInputGrad,
    const float* c_lpfColumnGrad,
    int nChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nPadding,
    int nOutputHeight,
    int nOutputWidth,
    int nSize
)
{
    const int nIndex    =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex >= nSize )
    {
        return;
    }
    const int nPatchSize    = nKernelSize * nKernelSize * nChannels;
    const int nPatchIndex   = nIndex % nPatchSize;
    const int nColumn       = nIndex / nPatchSize;
    const int nPosition     = nColumn % ( nOutputHeight * nOutputWidth );
    const int nSample       = nColumn / ( nOutputHeight * nOutputWidth );
    const int nChannel      = nPatchIndex % nChannels;
    const int nPixel        = nPatchIndex / nChannels;
    const int nInputY       = ( nPosition / nOutputWidth ) * nStride + nPixel / nKernelSize - nPadding;
    const int nInputX       = ( nPosition % nOutputWidth ) * nStride + nPixel % nKernelSize - nPadding;

    if( nInputY >= 0 && nInputY < nInputHeight && nInputX >= 0 && nInputX < nInputWidth )
    {
        // dL/dX[n,y,x,c_i] += sum_(o,k_y,k_x) dL/dC[p,q]
        // 複数patchが同じ入力要素へ重なるためatomicAddする。
        atomicAdd( lpfInputGrad + nSample * ( nInputHeight * nInputWidth * nChannels ) +
                       ( nInputY * nInputWidth + nInputX ) * nChannels + nChannel,
                   c_lpfColumnGrad[nIndex] );
    }
}

__global__ void kernel_Conv2D_bias(
    float* lpfBiasGrad,
    const float* c_lpfGemmGrad,
    int nChannels,
    int nSize
)
{
    const int nIndex    =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex<nSize )
    {
        // dL/db[c_o] += sum_(n,o) G[c_o,n*(H_out*W_out)+o]
        atomicAdd( lpfBiasGrad + nIndex % nChannels, c_lpfGemmGrad[nIndex] );
    }
}

__global__ void kernel_MaxPool2D_forward(
    float* lpfOutput,
    const float* c_lpfInput,
    int nChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nOutputWidth,
    int nPositions,
    int nSize
)
{
    const int nIndex    =blockIdx.x * blockDim.x + threadIdx.x;
    if( nIndex >= nSize )
    {
        return;
    }
    const int nChannel  = nIndex % nChannels;
    const int nPosition = ( nIndex / nChannels ) % nPositions;
    const int nSample   = nIndex / ( nChannels * nPositions );
    const int nOutputY  = nPosition / nOutputWidth;
    const int nOutputX  = nPosition % nOutputWidth;
    // Y[n,o_y,o_x,c] = max_(0<=k_y,k_x<K) X[n,o_y*S+k_y,o_x*S+k_x,c]
    float fMaximum = -FLT_MAX;
    for( int nKernelY = 0; nKernelY < nKernelSize; ++nKernelY )
    {
        for( int nKernelX = 0; nKernelX < nKernelSize; ++nKernelX )
        {
            const float fValue = c_lpfInput[nSample * ( nInputHeight * nInputWidth * nChannels ) +
                                            ( ( nOutputY * nStride + nKernelY ) * nInputWidth +
                                              nOutputX * nStride + nKernelX ) *
                                                nChannels +
                                            nChannel];
            if( fValue > fMaximum )
            {
                fMaximum = fValue;
            }
        }
    }
    lpfOutput[nIndex] = fMaximum;
}

__global__ void kernel_MaxPool2D_backward(
    float* lpfInputGrad,
    const float* c_lpfInput,
    const float* c_lpfOutputGrad,
    int nChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nOutputWidth,
    int nPositions,
    int nSize
)
{
    const int nIndex    =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex >= nSize )
    {
        return;
    }
    const int nChannel  = nIndex % nChannels;
    const int nPosition = ( nIndex / nChannels ) % nPositions;
    const int nSample   = nIndex / ( nChannels * nPositions );
    const int nOutputY  = nPosition / nOutputWidth;
    const int nOutputX  = nPosition % nOutputWidth;
    // a = argmax_(k_y,k_x) X[n,o_y*S+k_y,o_x*S+k_x,c]
    // 比較は > のみなので、同値なら走査順で最初の位置を保持する。
    int nBestRow = ( nOutputY * nStride * nInputWidth + nOutputX * nStride ) * nChannels + nChannel;
    float fMaximum = c_lpfInput[nSample * ( nInputHeight * nInputWidth * nChannels ) + nBestRow];
    for( int nKernelY = 0; nKernelY < nKernelSize; ++nKernelY )
    {
        for( int nKernelX = 0; nKernelX < nKernelSize; ++nKernelX )
        {
            const int nRow = ( ( nOutputY * nStride + nKernelY ) * nInputWidth +
                               nOutputX * nStride + nKernelX ) *
                                 nChannels +
                             nChannel;
            const float fValue =
                c_lpfInput[nSample * ( nInputHeight * nInputWidth * nChannels ) + nRow];
            if( fValue > fMaximum )
            {
                fMaximum = fValue;
                nBestRow = nRow;
            }
        }
    }
    // dL/dX[n,a,c] += dL/dY[n,o_y,o_x,c]
    // windowが重なる場合は同じ入力へ複数の勾配が流れるためatomicAddする。
    atomicAdd( lpfInputGrad + nSample * ( nInputHeight * nInputWidth * nChannels ) + nBestRow,
               c_lpfOutputGrad[nIndex] );
}