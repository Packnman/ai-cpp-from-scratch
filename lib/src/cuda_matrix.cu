#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <cfloat>
#include "cuda_matrix.h"

__global__ void kernel_fill(float* lpfResult,float c_fValue,int nSize);
__global__ void kernel_mul_elementwise(float* lpfResult,const float* c_lpfA,const float* c_lpfB,int nSize);
__global__ void kernel_ReLU_forward(float* lpfResult,const float* c_lpfValue,int nSize);
__global__ void kernel_ReLU_backward(float* lpfResult,const float* c_lpfData,const float* c_lpfGrad,int nSize);
__global__ void kernel_GELU_forward(float* lpfResult,const float* c_lpfValue,int nSize);
__global__ void kernel_GELU_backward(float* lpfResult,const float* c_lpfData,const float* c_lpfGrad,int nSize);
__global__ void kernel_Dropout_forward(float* lpfResult,const float* c_lpfValue,float* lpfMask,float fDropProbability,float fScale,int nSize);
__global__ void kernel_Dropout_backward(float* lpfResult,const float* c_lpfGrad,const float* c_lpfMask,int nSize);
__global__ void kernel_BatchNorm_forward_training(float* lpfResult,const float* c_lpfValue,const float* c_lpfGamma,const float* c_lpfBeta,float* lpfRunningMean,float* lpfRunningVar,float* lpfNormalized,float* lpfInvStd,float fMomentum,float fEpsilon,int nFeatures,int nBatch);
__global__ void kernel_BatchNorm_forward_evaluation(float* lpfResult,const float* c_lpfValue,const float* c_lpfGamma,const float* c_lpfBeta,const float* c_lpfRunningMean,const float* c_lpfRunningVar,float* lpfNormalized,float* lpfInvStd,float fEpsilon,int nFeatures,int nBatch);
__global__ void kernel_BatchNorm_backward(float* lpfInputGrad,float* lpfGammaGrad,float* lpfBetaGrad,const float* c_lpfOutputGrad,const float* c_lpfGamma,const float* c_lpfNormalized,const float* c_lpfInvStd,bool isTraining,int nFeatures,int nBatch);
__global__ void kernel_SoftmaxCrossEntropy_forward(float* lpfResult,const float* c_lpfLogits,const float* c_lpfTarget,int nClass,int nBatch);
__global__ void kernel_SoftmaxCrossEntropy_backward(float* lpfResult,const float* c_lpfLogits,const float* c_lpfTarget,const float* c_lpfGrad,int nClass,int nBatch);
__global__ void kernel_Adam_update(float* lpfData,const float* c_lpfGrad,float* lpfFirstMoment,float* lpfSecondMoment,float fLearningRate,float fBeta1,float fBeta2,float fBeta1Correction,float fBeta2Correction,float fEpsilon,int nSize);
__global__ void kernel_Conv2D_im2col(float* lpfColumns,const float* c_lpfInput,int nChannels,int nInputHeight,int nInputWidth, int nKernelSize, int nStride,int nPadding,int nOutputHeight,int nOutputWidth,int nBatch,int nSize);
__global__ void kernel_Conv2D_pack(float* lpfOutput,const float* c_lpfGemm,const float* c_lpfBias,int nOutputChannels,int nPositions,int nBatch,int nSize);
__global__ void kernel_Conv2D_unpack(float* lpfGemmGrad,const float* c_lpfOutputGrad,int nOutputChannels,int nPositions,int nSize);
__global__ void kernel_Conv2D_col2im(float* lpfInputGrad,const float* c_lpfColumnGrad,int nChannels,int nInputHeight,int nInputWidth,int nKernelSize,int nStride,int nPadding,int nOutputHeight,int nOutputWidth,int nSize);
__global__ void kernel_Conv2D_bias(float* lpfBiasGrad,const float* c_lpfGemmGrad,int nChannels,int nSize);
__global__ void kernel_Pooling_forward(float* lpfOutput,const float* c_lpfInput,int nChannels,int nInputHeight,int nInputWidth,int nKernelSize,int nStride,int nOutputWidth,int nPositions,int nSize);
__global__ void kernel_Pooling_backward(float* lpfInputGrad,const float* c_lpfInput,const float* c_lpfOutputGrad,int nChannels,int nInputHeight,int nInputWidth,int nKernelSize,int nStride,int nOutputWidth,int nPositions,int nSize);

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
void cuda_Dropout_forward(cuMat& mResult,const cuMat& c_mValue,cuMat& mMask,float fDropProbability)
{
    if( (mResult._nRows!=c_mValue._nRows)||(mResult._nCols!=c_mValue._nCols)||
        (mMask._nRows!=c_mValue._nRows)||(mMask._nCols!=c_mValue._nCols) )
    {
        throw std::runtime_error("cuda_Dropout_forward: matrix size mismatch");
    }
    const int nSize =c_mValue._nRows*c_mValue._nCols;
    if( nSize<=0 ) {return;}
    const int nThreads =256;
    const int nBlocks =(nSize+nThreads-1)/nThreads;
    kernel_Dropout_forward<<<nBlocks,nThreads>>>(
        mResult._lpfDevice,c_mValue._lpfDevice,mMask._lpfDevice,
        fDropProbability,1.0f/(1.0f-fDropProbability),nSize
    );
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(std::string("cuda_Dropout_forward: kernel launch failed: ")+cudaGetErrorString(cudError));
    }
}
void cuda_Dropout_backward(cuMat& mResult,const cuMat& c_mGrad,const cuMat& c_mMask)
{
    if( (mResult._nRows!=c_mGrad._nRows)||(mResult._nCols!=c_mGrad._nCols)||
        (c_mMask._nRows!=c_mGrad._nRows)||(c_mMask._nCols!=c_mGrad._nCols) )
    {
        throw std::runtime_error("cuda_Dropout_backward: matrix size mismatch");
    }
    const int nSize =c_mGrad._nRows*c_mGrad._nCols;
    if( nSize<=0 ) {return;}
    const int nThreads =256;
    const int nBlocks =(nSize+nThreads-1)/nThreads;
    kernel_Dropout_backward<<<nBlocks,nThreads>>>(mResult._lpfDevice,c_mGrad._lpfDevice,c_mMask._lpfDevice,nSize);
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(std::string("cuda_Dropout_backward: kernel launch failed: ")+cudaGetErrorString(cudError));
    }
}
void cuda_BatchNorm_forward_training(
    cuMat& mResult,const cuMat& c_mValue,
    const cuMat& c_mGamma,const cuMat& c_mBeta,
    cuMat& mRunningMean,cuMat& mRunningVar,
    cuMat& mNormalized,cuMat& mInvStd,
    float fMomentum,float fEpsilon
)
{
    const int nFeatures =c_mValue._nRows;
    const int nBatch    =c_mValue._nCols;
    if( (nBatch<=0)||(mResult._nRows!=nFeatures)||(mResult._nCols!=nBatch)||
        (c_mGamma._nRows!=nFeatures)||(c_mGamma._nCols!=1)||
        (c_mBeta._nRows!=nFeatures)||(c_mBeta._nCols!=1)||
        (mRunningMean._nRows!=nFeatures)||(mRunningMean._nCols!=1)||
        (mRunningVar._nRows!=nFeatures)||(mRunningVar._nCols!=1)||
        (mNormalized._nRows!=nFeatures)||(mNormalized._nCols!=nBatch)||
        (mInvStd._nRows!=nFeatures)||(mInvStd._nCols!=1) )
    {
        throw std::runtime_error("cuda_BatchNorm_forward_training: matrix size mismatch");
    }
    if( nFeatures<=0 ) {return;}
    const int nThreads =256;
    const int nBlocks =(nFeatures+nThreads-1)/nThreads;
    //
    kernel_BatchNorm_forward_training<<<nBlocks,nThreads>>>(
        mResult._lpfDevice,
        c_mValue._lpfDevice,
        c_mGamma._lpfDevice,
        c_mBeta._lpfDevice,
        mRunningMean._lpfDevice,
        mRunningVar._lpfDevice,
        mNormalized._lpfDevice,
        mInvStd._lpfDevice,
        fMomentum,
        fEpsilon,
        nFeatures,
        nBatch
    );
    //
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(std::string(
            "cuda_BatchNorm_forward_training: kernel launch failed: ")+cudaGetErrorString(cudError)
        );
    }
}
void cuda_BatchNorm_forward_evaluation(
    cuMat& mResult,const cuMat& c_mValue,
    const cuMat& c_mGamma,const cuMat& c_mBeta,
    const cuMat& c_mRunningMean,const cuMat& c_mRunningVar,
    cuMat& mNormalized,cuMat& mInvStd,float fEpsilon
)
{
    const int nFeatures =c_mValue._nRows;
    const int nBatch =c_mValue._nCols;
    if( (nBatch<=0)||(mResult._nRows!=nFeatures)||(mResult._nCols!=nBatch)||
        (c_mGamma._nRows!=nFeatures)||(c_mGamma._nCols!=1)||
        (c_mBeta._nRows!=nFeatures)||(c_mBeta._nCols!=1)||
        (c_mRunningMean._nRows!=nFeatures)||(c_mRunningMean._nCols!=1)||
        (c_mRunningVar._nRows!=nFeatures)||(c_mRunningVar._nCols!=1)||
        (mNormalized._nRows!=nFeatures)||(mNormalized._nCols!=nBatch)||
        (mInvStd._nRows!=nFeatures)||(mInvStd._nCols!=1) )
    {
        throw std::runtime_error("cuda_BatchNorm_forward_evaluation: matrix size mismatch");
    }
    if( nFeatures<=0 ) {return;}
    const int nThreads =256;
    const int nBlocks =(nFeatures+nThreads-1)/nThreads;
    //
    kernel_BatchNorm_forward_evaluation<<<nBlocks,nThreads>>>(
        mResult._lpfDevice,
        c_mValue._lpfDevice,
        c_mGamma._lpfDevice,
        c_mBeta._lpfDevice,
        c_mRunningMean._lpfDevice,
        c_mRunningVar._lpfDevice,
        mNormalized._lpfDevice,
        mInvStd._lpfDevice,
        fEpsilon,
        nFeatures,
        nBatch
    );
    //
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(std::string(
            "cuda_BatchNorm_forward_evaluation: kernel launch failed: ")+cudaGetErrorString(cudError)
        );
    }
}
void cuda_BatchNorm_backward(
    cuMat& mInputGrad,cuMat& mGammaGrad,cuMat& mBetaGrad,
    const cuMat& c_mOutputGrad,const cuMat& c_mGamma,
    const cuMat& c_mNormalized,const cuMat& c_mInvStd,
    bool isTraining
)
{
    const int nFeatures =c_mOutputGrad._nRows;
    const int nBatch =c_mOutputGrad._nCols;
    if( (nBatch<=0)||(mInputGrad._nRows!=nFeatures)||(mInputGrad._nCols!=nBatch)||
        (mGammaGrad._nRows!=nFeatures)||(mGammaGrad._nCols!=1)||
        (mBetaGrad._nRows!=nFeatures)||(mBetaGrad._nCols!=1)||
        (c_mGamma._nRows!=nFeatures)||(c_mGamma._nCols!=1)||
        (c_mNormalized._nRows!=nFeatures)||(c_mNormalized._nCols!=nBatch)||
        (c_mInvStd._nRows!=nFeatures)||(c_mInvStd._nCols!=1) )
    {
        throw std::runtime_error("cuda_BatchNorm_backward: matrix size mismatch");
    }
    if( nFeatures<=0 ) {return;}
    const int nThreads  =256;
    const int nBlocks   =(nFeatures+nThreads-1)/nThreads;
    //
    kernel_BatchNorm_backward<<<nBlocks,nThreads>>>(
        mInputGrad._lpfDevice,
        mGammaGrad._lpfDevice,
        mBetaGrad._lpfDevice,
        c_mOutputGrad._lpfDevice,
        c_mGamma._lpfDevice,
        c_mNormalized._lpfDevice,
        c_mInvStd._lpfDevice,
        isTraining,
        nFeatures,
        nBatch
    );
    //
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(std::string("cuda_BatchNorm_backward: kernel launch failed: ")+cudaGetErrorString(cudError));
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
    int nSize       =c_mColumnGrad._nRows * c_mColumnGrad._nCols;
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
    int nSize       =c_mGemmGrad._nRows * c_mGemmGrad._nCols;
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
void cuda_Pooling_forward(
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
    kernel_Pooling_forward<<<nBlocks, nThreads>>>(
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
            std::string( "cuda_Pooling_forward: " )
            + cudaGetErrorString( cudError )
        );
    }
}
void cuda_Pooling_backward(
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
    int nSize       =c_mOutputGrad._nRows * c_mOutputGrad._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Pooling_backward<<<nBlocks, nThreads>>>(
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
            std::string( "cuda_Pooling_backward: " )
            + cudaGetErrorString( cudError )
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
__global__ void kernel_Dropout_forward(float* lpfResult,const float* c_lpfValue,float* lpfMask,float fDropProbability,float fScale,int nSize)
{
    int nIndex =blockIdx.x*blockDim.x+threadIdx.x;

    if( nIndex<nSize )
    {
        const float fMask =lpfMask[nIndex]>fDropProbability ? fScale : 0.0f;

        lpfMask[nIndex]     =fMask;
        lpfResult[nIndex]   =c_lpfValue[nIndex]*fMask;
    }
}
__global__ void kernel_Dropout_backward(float* lpfResult,const float* c_lpfGrad,const float* c_lpfMask,int nSize)
{
    int nIndex =blockIdx.x*blockDim.x+threadIdx.x;

    if( nIndex<nSize )
    {
        lpfResult[nIndex] +=c_lpfGrad[nIndex]*c_lpfMask[nIndex];
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

// 学習時のBatchNorm。1 CUDA threadが1特徴量 f を担当し、
// その特徴量についてバッチ方向 b=0,...,B-1 を走査する。
// cuMatはcolumn-majorなので x_{f,b} の位置は b*F+f になる。
__global__ void kernel_BatchNorm_forward_training(
    float* lpfResult,const float* c_lpfValue,
    const float* c_lpfGamma,const float* c_lpfBeta,
    float* lpfRunningMean,float* lpfRunningVar,
    float* lpfNormalized,float* lpfInvStd,
    float fMomentum,float fEpsilon,int nFeatures,int nBatch
)
{
    int nFeature =blockIdx.x*blockDim.x+threadIdx.x;

    if( nFeature>=nFeatures ) {return;}

    // ミニバッチ平均: mu_f = (1/B) * sum_b x_{f,b}
    float fMean =0.0f;
    for( int nBatchIndex=0;nBatchIndex<nBatch;++nBatchIndex )
    {
        fMean +=c_lpfValue[nBatchIndex*nFeatures+nFeature];
    }
    fMean /=static_cast<float>(nBatch);

    // ミニバッチ分散（母分散）:
    // sigma_f^2 = (1/B) * sum_b (x_{f,b}-mu_f)^2
    float fVariance =0.0f;
    for( int nBatchIndex=0;nBatchIndex<nBatch;++nBatchIndex )
    {
        const float fCentered =c_lpfValue[nBatchIndex*nFeatures+nFeature]-fMean;
        fVariance +=fCentered*fCentered;
    }
    fVariance /=static_cast<float>(nBatch);

    // inv_std_f = 1/sqrt(sigma_f^2+epsilon)。backwardでも再利用する。
    const float fInvStd =rsqrtf(fVariance+fEpsilon);
    lpfInvStd[nFeature]         =fInvStd;

    // running_mean_f <- (1-m)*running_mean_f + m*mu_f
    lpfRunningMean[nFeature]    =(1.0f-fMomentum)*lpfRunningMean[nFeature]+fMomentum*fMean;

    // 学習時の正規化は分母Bの分散を使うが、running varianceには
    // 不偏分散 B/(B-1)*sigma_f^2 を保存する（B=1では補正しない）。
    // running_var_f <- (1-m)*running_var_f + m*unbiased_var_f
    const float fRunningVariance =nBatch>1
        ? fVariance*static_cast<float>(nBatch)/static_cast<float>(nBatch-1)
        : fVariance;
    lpfRunningVar[nFeature] =(1.0f-fMomentum)*lpfRunningVar[nFeature]+fMomentum*fRunningVariance;
    for( int nBatchIndex=0;nBatchIndex<nBatch;++nBatchIndex )
    {
        const int nIndex        =nBatchIndex*nFeatures+nFeature;

        // x_hat_{f,b} = (x_{f,b}-mu_f)/sqrt(sigma_f^2+epsilon)
        // y_{f,b} = gamma_f*x_hat_{f,b}+beta_f
        const float fNormalized =(c_lpfValue[nIndex]-fMean)*fInvStd;

        lpfNormalized[nIndex]   =fNormalized;
        lpfResult[nIndex]       =c_lpfGamma[nFeature]*fNormalized+c_lpfBeta[nFeature];
    }
}
// 評価時は現在のミニバッチから平均・分散を求めず、
// 学習中に蓄積したrunning_meanとrunning_varだけを使用する。
__global__ void kernel_BatchNorm_forward_evaluation(
    float* lpfResult,const float* c_lpfValue,
    const float* c_lpfGamma,const float* c_lpfBeta,
    const float* c_lpfRunningMean,const float* c_lpfRunningVar,
    float* lpfNormalized,float* lpfInvStd,
    float fEpsilon,int nFeatures,int nBatch
)
{
    int nFeature =blockIdx.x*blockDim.x+threadIdx.x;

    if( nFeature>=nFeatures ) {return;}
    // x_hat_{f,b} = (x_{f,b}-running_mean_f)
    //                 / sqrt(running_var_f+epsilon)
    const float fInvStd =rsqrtf(c_lpfRunningVar[nFeature]+fEpsilon);
    lpfInvStd[nFeature] =fInvStd;
    for( int nBatchIndex=0;nBatchIndex<nBatch;++nBatchIndex )
    {
        const int nIndex        =nBatchIndex*nFeatures+nFeature;
        const float fNormalized =(c_lpfValue[nIndex]-c_lpfRunningMean[nFeature])*fInvStd;

        lpfNormalized[nIndex]   =fNormalized;
        lpfResult[nIndex]       =c_lpfGamma[nFeature]*fNormalized+c_lpfBeta[nFeature];
    }
}
// g_{f,b}=dL/dy_{f,b} として入力、gamma、betaの勾配を計算する。
// forwardと同様に1 CUDA threadが1特徴量 f を担当する。
__global__ void kernel_BatchNorm_backward(
    float* lpfInputGrad,float* lpfGammaGrad,float* lpfBetaGrad,
    const float* c_lpfOutputGrad,const float* c_lpfGamma,
    const float* c_lpfNormalized,const float* c_lpfInvStd,
    bool isTraining,int nFeatures,int nBatch
)
{
    int nFeature =blockIdx.x*blockDim.x+threadIdx.x;

    if( nFeature>=nFeatures ) {return;}
    // S1_f = sum_b g_{f,b}
    // S2_f = sum_b g_{f,b}*x_hat_{f,b}
    float fGradSum              =0.0f;
    float fGradNormalizedSum    =0.0f;
    for( int nBatchIndex=0;nBatchIndex<nBatch;++nBatchIndex )
    {
        const int nIndex    =nBatchIndex*nFeatures+nFeature;
        const float fGrad   =c_lpfOutputGrad[nIndex];
        fGradSum            +=fGrad;
        fGradNormalizedSum  +=fGrad*c_lpfNormalized[nIndex];
    }
    // dL/dbeta_f = S1_f, dL/dgamma_f = S2_f
    lpfGammaGrad[nFeature]  +=fGradNormalizedSum;
    lpfBetaGrad[nFeature]   +=fGradSum;

    const float fGammaInvStd =c_lpfGamma[nFeature]*c_lpfInvStd[nFeature];
    for( int nBatchIndex=0;nBatchIndex<nBatch;++nBatchIndex )
    {
        const int nIndex    =nBatchIndex*nFeatures+nFeature;
        if( isTraining )
        {
            // dL/dx_{f,b} = gamma_f*inv_std_f/B *
            //   (B*g_{f,b} - S1_f - x_hat_{f,b}*S2_f)
            lpfInputGrad[nIndex]    +=fGammaInvStd/float(nBatch)*(
                float(nBatch)*c_lpfOutputGrad[nIndex]-fGradSum-
                c_lpfNormalized[nIndex]*fGradNormalizedSum
            );
        }
        else
        {
            // 評価時の平均・分散は定数なので、
            // dL/dx_{f,b} = gamma_f*inv_std_f*g_{f,b}
            lpfInputGrad[nIndex]    +=fGammaInvStd*c_lpfOutputGrad[nIndex];
        }
    }
}

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

__global__ void kernel_Pooling_forward(
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

__global__ void kernel_Pooling_backward(
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