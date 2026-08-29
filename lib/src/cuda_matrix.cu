#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include "cuda_matrix.h"

__global__ void kernel_fill(float* lpfResult,float c_fValue,int nSize);
__global__ void kernel_mul_elementwise(float* lpfResult,const float* c_lpfA,const float* c_lpfB,int nSize);
__global__ void kernel_ReLU_forward(float* lpfResult,const float* c_lpfValue,int nSize);
__global__ void kernel_ReLU_backward(float* lpfResult,const float* c_lpfData,const float* c_lpfGrad,int nSize);
__global__ void kernel_GELU_forward(float* lpfResult,const float* c_lpfValue,int nSize);
__global__ void kernel_GELU_backward(float* lpfResult,const float* c_lpfData,const float* c_lpfGrad,int nSize);
__global__ void kernel_SoftmaxCrossEntropy_forward(float* lpfResult,const float* c_lpfLogits,const float* c_lpfTarget,int nClass,int nBatch);
__global__ void kernel_SoftmaxCrossEntropy_backward(float* lpfResult,const float* c_lpfLogits,const float* c_lpfTarget,const float* c_lpfGrad,int nClass,int nBatch);
__global__ void kernel_Adam_update(float* lpfData,const float* c_lpfGrad,float* lpfFirstMoment,float* lpfSecondMoment,float fLearningRate,float fBeta1,float fBeta2,float fBeta1Correction,float fBeta2Correction,float fEpsilon,int nSize);

void cuMat::ones()
{
    int nSize       =_nRows * _nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_fill<<<nBlocks,nThreads>>>(
        _lpfDevice,
        1.0f,
        nSize
    );
    // 
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "kernel_fill failed: " )
            + cudaGetErrorString( cudError )
        );
    }
}
void cuda_fill(cuMat& mResult,float fValue)
{
    // R[:] = fValue

    int nSize       =mResult._nRows * mResult._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_fill<<<nBlocks,nThreads>>>(
        mResult._lpfDevice,
        fValue,nSize
    );
    // 
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "kernel_fill failed: " )
            + cudaGetErrorString( cudError )
        );
    }
}
void cuda_mul_elementwise(cuMat& mResult,const cuMat& c_mA,const cuMat& c_mB)
{
    // R = c_mA ⦿ c_mB
    
    // 行列数の確認
    if( (c_mA._nRows!=c_mB._nRows)||(c_mA._nCols!=c_mB._nCols) )
    {
        throw std::runtime_error(
            "cuda_mul_elementwise: A and B size mismatch"
        );
    }
    if( (mResult._nRows!=c_mA._nRows)||(mResult._nCols!=c_mA._nCols) )
    {
        throw std::runtime_error(
            "cuda_mul_elementwise: result size mismatch"
        );
    }

    int nSize    =c_mA._nRows * c_mA._nCols;
    int nThreads =256;
    int nBlocks  =(nSize + nThreads - 1) / nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_mul_elementwise<<<nBlocks, nThreads>>>(
        mResult._lpfDevice,
        c_mA._lpfDevice,
        c_mB._lpfDevice,
        nSize
    );

    cudaError_t cudError = cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_mul_elementwise: kernel launch failed"
        );
    }
}
void cuda_ReLU_forward(cuMat& mResult,const cuMat& c_mValue)
{
    if( (mResult._nRows!=c_mValue._nRows)||(mResult._nCols!=c_mValue._nCols) )
    {
        throw std::runtime_error(
            "cuda_ReLU_forward: matrix size mismatch"
        );
    }
    //
    int nSize       =c_mValue._nRows * c_mValue._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_ReLU_forward<<<nBlocks,nThreads>>>(
        mResult._lpfDevice,
        c_mValue._lpfDevice,
        nSize
    );

    cudaError_t cudError = cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_ReLU_forward: kernel launch failed"
        );
    }
}
void cuda_ReLU_backward(cuMat& mResult,const cuMat& c_mData,const cuMat& c_mGrad)
{
    if( (mResult._nRows!=c_mData._nRows)||(mResult._nCols!=c_mData._nCols)||
        (c_mData._nRows!=c_mGrad._nRows)||(c_mData._nCols!=c_mGrad._nCols) )
    {
        throw std::runtime_error(
            "cuda_ReLU_backward: matrix size mismatch"
        );
    }
    //
    int nSize       =c_mData._nRows * c_mData._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_ReLU_backward<<<nBlocks,nThreads>>>(
        mResult._lpfDevice,
        c_mData._lpfDevice,
        c_mGrad._lpfDevice,
        nSize
    );
    
    cudaError_t cudError = cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_ReLU_backward: kernel launch failed"
        );
    }
}
void cuda_GELU_forward(cuMat& mResult,const cuMat& c_mValue)
{
    if( (mResult._nRows!=c_mValue._nRows)||(mResult._nCols!=c_mValue._nCols) )
    {
        throw std::runtime_error(
            "cuda_ReLU_forward: matrix size mismatch"
        );
    }
    //
    int nSize       =c_mValue._nRows * c_mValue._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_GELU_forward<<<nBlocks,nThreads>>>(
        mResult._lpfDevice,
        c_mValue._lpfDevice,
        nSize
    );
    //
    cudaError_t cudError = cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_ReLU_forward: kernel launch failed"
        );
    }
}
void cuda_GELU_backward(cuMat& mResult,const cuMat& c_mData,const cuMat& c_mGrad)
{
    if( (mResult._nRows!=c_mData._nRows)||(mResult._nCols!=c_mData._nCols)||
        (c_mData._nRows!=c_mGrad._nRows)||(c_mData._nCols!=c_mGrad._nCols) )
    {
        throw std::runtime_error(
            "cuda_GELU_backward: matrix size mismatch"
        );
    }
    //
    int nSize       =c_mData._nRows * c_mData._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_GELU_backward<<<nBlocks,nThreads>>>(
        mResult._lpfDevice,
        c_mData._lpfDevice,
        c_mGrad._lpfDevice,
        nSize
    );
    //
    cudaError_t cudError = cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_GELU_backward: kernel launch failed"
        );
    }
}
void cuda_SoftmaxCrossEntropy_forward(
    cuMat& mResult,
    const cuMat& c_mLogits,
    const cuMat& c_mTarget
)
{
    if( (c_mLogits._nRows!=c_mTarget._nRows)||(c_mLogits._nCols!=c_mTarget._nCols) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_forward: matrix size mismatch"
        );
    }
    if( (mResult._nRows!=1)||(mResult._nCols!=1) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_forward: result must be 1x1"
        );
    }
    //
    int nClass      =c_mLogits._nRows;
    int nBatch      =c_mLogits._nCols;
    int nThreads    =256;
    int nBlocks     =(nBatch + nThreads - 1)/nThreads;
    if( (nClass<=0)||(nBatch<=0) )  {return;}
    //
    cuda_fill( mResult,0.0f );      // atomicAddするので最初は0
    kernel_SoftmaxCrossEntropy_forward<<<nBlocks,nThreads>>>(
        mResult._lpfDevice,
        c_mLogits._lpfDevice,
        c_mTarget._lpfDevice,
        nClass,
        nBatch
    );
    //
    cudaError_t cudError = cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_forward: kernel launch failed"
        );
    }
}
void cuda_SoftmaxCrossEntropy_backward(
    cuMat& mLogitsGrad,
    const cuMat& c_mLogits,
    const cuMat& c_mTarget,
    const cuMat& c_mGrad
)
{
    if( (c_mLogits._nRows!=c_mTarget._nRows)||(c_mLogits._nCols!=c_mTarget._nCols) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_backward: matrix size mismatch"
        );
    }
    if( (c_mLogits._nRows!=mLogitsGrad._nRows)||(c_mLogits._nCols!=mLogitsGrad._nCols) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_backward: gradient size mismatch"
        );
    }
    if( (c_mGrad._nRows!=1)||(c_mGrad._nCols!=1) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_backward: grad must be 1x1"
        );
    }
    //
    int nClass      =c_mLogits._nRows;
    int nBatch      =c_mLogits._nCols;
    int nThreads    =256;
    int nBlocks     =(nBatch + nThreads - 1)/nThreads;
    if( (nClass<=0)||(nBatch<=0) )  {return;}
    //
    kernel_SoftmaxCrossEntropy_backward<<<nBlocks,nThreads>>>(
        mLogitsGrad._lpfDevice,
        c_mLogits._lpfDevice,
        c_mTarget._lpfDevice,
        c_mGrad._lpfDevice,
        nClass,
        nBatch
    );
    //
    cudaError_t cudError = cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_backward: kernel launch failed"
        );
    }
}
void cuda_Adam_update(
    cuMat& mData,
    const cuMat& c_mGrad,
    cuMat& mFirstMoment,
    cuMat& mSecondMoment,
    float fLearningRate,
    float fBeta1,
    float fBeta2,
    float fBeta1Correction,
    float fBeta2Correction,
    float fEpsilon
)
{
    if( (mData._nRows!=c_mGrad._nRows)||(mData._nCols!=c_mGrad._nCols)||
        (mData._nRows!=mFirstMoment._nRows)||(mData._nCols!=mFirstMoment._nCols)||
        (mData._nRows!=mSecondMoment._nRows)||(mData._nCols!=mSecondMoment._nCols) )
    {
        throw std::runtime_error(
            "cuda_Adam_update: matrix size mismatch"
        );
    }

    int nSize       =mData._nRows*mData._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize+nThreads-1)/nThreads;
    if( nSize<=0 )  {return;}
    //
    kernel_Adam_update<<<nBlocks,nThreads>>>(
        mData._lpfDevice,
        c_mGrad._lpfDevice,
        mFirstMoment._lpfDevice,
        mSecondMoment._lpfDevice,
        fLearningRate,
        fBeta1,
        fBeta2,
        fBeta1Correction,
        fBeta2Correction,
        fEpsilon,
        nSize
    );

    cudaError_t cudError   =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_Adam_update: kernel launch failed"
        );
    }
}


__global__ void kernel_fill(float* lpfResult,float c_fValue,int nSize)
{
    int nIndex   =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex<nSize )
    {
        lpfResult[nIndex]    =c_fValue;
    }
}

__global__ void kernel_mul_elementwise(float* lpfResult,const float* c_lpfA,const float* c_lpfB,int nSize)
{
    int nIndex   =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex<nSize )
    {
        lpfResult[nIndex]    =c_lpfA[nIndex] * c_lpfB[nIndex];
    }
}
__global__ void kernel_ReLU_forward(float* lpfResult,const float* c_lpfValue,int nSize)
{
    int nIndex   =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex<nSize )
    {
        lpfResult[nIndex]  =c_lpfValue[nIndex]>0.0f ? c_lpfValue[nIndex] : 0.0f;
    }
}
__global__ void kernel_ReLU_backward(float* lpfResult,const float* c_lpfData,const float* c_lpfGrad,int nSize)
{
    int nIndex   =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex<nSize )
    {
        if( c_lpfData[nIndex]>0.0f )
        {
            lpfResult[nIndex]  +=c_lpfGrad[nIndex];
        }
    }
}
__global__ void kernel_GELU_forward(float* lpfResult,const float* c_lpfValue,int nSize)
{
    int nIndex   =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex<nSize )
    {
        const float c_fValue =c_lpfValue[nIndex];
        const float c_fScale =0.7978845608f;
        const float c_fCubic =0.044715f;
        const float c_fInner =c_fScale*(c_fValue+c_fCubic*c_fValue*c_fValue*c_fValue);
        const float c_fTanh  =tanhf(c_fInner);
        
        const float c_fDerivative =0.5f*(1.0f+c_fTanh);

        lpfResult[nIndex] =c_lpfValue[nIndex]*c_fDerivative;
    }
}
__global__ void kernel_GELU_backward(float* lpfResult,const float* c_lpfData,const float* c_lpfGrad,int nSize)
{
    int nIndex   =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex<nSize )
    {
        const float c_fValue =c_lpfData[nIndex];
        const float c_fScale =0.7978845608f;
        const float c_fCubic =0.044715f;
        const float c_fInner =c_fScale*(c_fValue+c_fCubic*c_fValue*c_fValue*c_fValue);
        const float c_fTanh  =tanhf(c_fInner);

        const float c_fDerivative =0.5f*(1.0f+c_fTanh)+0.5f*c_fValue*(1.0f-c_fTanh*c_fTanh)*c_fScale*(1.0f+3.0f*c_fCubic*c_fValue*c_fValue);

        lpfResult[nIndex] +=c_lpfGrad[nIndex]*c_fDerivative;
    }
}
__global__ void kernel_SoftmaxCrossEntropy_forward(float* lpfResult,const float* c_lpfLogits,const float* c_lpfTarget,int nClass,int nBatch)
{
    int nBatchIndex   =blockIdx.x * blockDim.x + threadIdx.x;

    if( nBatchIndex<nBatch )
    {
        // max(c_lpfLogits)を求める
        float fMaxLogits    =c_lpfLogits[nBatchIndex*nClass];
        for( int nClassIndex=1;nClassIndex<nClass;nClassIndex++ )
        {
            if( c_lpfLogits[nBatchIndex*nClass+nClassIndex]>fMaxLogits )
            {
                fMaxLogits =c_lpfLogits[nBatchIndex*nClass+nClassIndex];
            }
        }
        // expの合計
        float fSumExp       =0.0f;
        for( int nClassIndex=0;nClassIndex<nClass;nClassIndex++ )
        {
            fSumExp +=expf(c_lpfLogits[nBatchIndex*nClass+nClassIndex]-fMaxLogits);
        }
        // CrossEntropy
        float fLoss         =0.0;
        float fSumExpLog    =fMaxLogits + logf( fSumExp );
        for( int nClassIndex=0;nClassIndex<nClass;nClassIndex++ )
        {
            int nIndex =nBatchIndex*nClass+nClassIndex;

            fLoss   -=c_lpfTarget[nIndex] * (c_lpfLogits[nIndex] - fSumExpLog);
        }
        // batch平均
        atomicAdd(
            lpfResult,
            fLoss/static_cast<float>(nBatch)
        );
    }
}
__global__ void kernel_SoftmaxCrossEntropy_backward(float* lpfResult,const float* c_lpfLogits,const float* c_lpfTarget,const float* c_lpfGrad,int nClass,int nBatch)
{
    int nBatchIndex   =blockIdx.x * blockDim.x + threadIdx.x;

    if( nBatchIndex<nBatch )
    {
        // max(c_lpfLogits)
        float fMaxLogits    =c_lpfLogits[nBatchIndex*nClass];
        for( int nClassIndex=0;nClassIndex<nClass;nClassIndex++ )
        {
            if( c_lpfLogits[nBatchIndex*nClass+nClassIndex]>fMaxLogits )
            {
                fMaxLogits =c_lpfLogits[nBatchIndex*nClass+nClassIndex];
            }
        }
        // expの合計
        float fSumExp   =0.0f;
        for( int nClassIndex=0;nClassIndex<nClass;nClassIndex++ )
        {
            fSumExp +=expf(c_lpfLogits[nBatchIndex*nClass+nClassIndex]-fMaxLogits);
        }
        // gradient
        for( int nClassIndex=0;nClassIndex<nClass;nClassIndex++ )
        {
            int nIndex =nBatchIndex*nClass+nClassIndex;
            float fSoftmax  =expf( c_lpfLogits[nIndex]-fMaxLogits )/fSumExp;

            lpfResult[nIndex]    +=c_lpfGrad[0]*(fSoftmax-c_lpfTarget[nIndex])/static_cast<float>(nBatch);
        }
    }
}
__global__ void kernel_Adam_update(float* lpfData,const float* c_lpfGrad,float* lpfFirstMoment,float* lpfSecondMoment,float fLearningRate,float fBeta1,float fBeta2,float fBeta1Correction,float fBeta2Correction,float fEpsilon,int nSize)
{
    int nIndex   =blockIdx.x * blockDim.x + threadIdx.x;

    if( nIndex<nSize )
    {
        float fGrad =c_lpfGrad[nIndex];

        // 1次モーメント
        lpfFirstMoment[nIndex] =fBeta1*lpfFirstMoment[nIndex]+(1.0f-fBeta1)*fGrad;
        // 2次モーメント
        lpfSecondMoment[nIndex] =fBeta2*lpfSecondMoment[nIndex]+(1.0f-fBeta2)*fGrad*fGrad;
        // Bias Correction
        float fFirstMomentHat =lpfFirstMoment[nIndex]/fBeta1Correction;
        float fSecondMomentHat =lpfSecondMoment[nIndex]/fBeta2Correction;
        // Parameter update
        lpfData[nIndex] -=fLearningRate*fFirstMomentHat/(sqrtf(fSecondMomentHat)+fEpsilon);
    }
}
