#include "cuda_memory.h"
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdlib>
#include <iostream>
#include <cfloat>
#include <climits>
#include "cuda_matrix.h"

__global__ void kernel_fill(float* lpfResult,float c_fValue,int nSize);
__global__ void kernel_mul_elementwise(float* lpfResult,const float* c_lpfA,const float* c_lpfB,int nSize);
__global__ void kernel_BatchMatMul_forward(float* lpfResult,const float* c_lpfA,const float* c_lpfB,int nK,int nN,int nBatch,int nSize);
__global__ void kernel_BatchMatMul_backward_A(float* lpfAGrad,const float* c_lpfOutputGrad,const float* c_lpfB,int nK,int nN,int nBatch,int nSize);
__global__ void kernel_BatchMatMul_backward_B(float* lpfBGrad,const float* c_lpfOutputGrad,const float* c_lpfA,int nM,int nK,int nN,int nBatch,int nSize);
__global__ void kernel_LayerNorm_forward(float* lpfResult,const float* c_lpfInput,const float* c_lpfGamma,const float* c_lpfBeta,int nFeatures,int nPositions,float fEpsilon);
__global__ void kernel_LayerNorm_backward(float* lpfInputGrad,float* lpfGammaGrad,float* lpfBetaGrad,const float* c_lpfOutputGrad,const float* c_lpfInput,const float* c_lpfGamma,int nFeatures,int nPositions,float fEpsilon);
__global__ void kernel_LayerNorm_parameter_reduce(float* lpfGammaGrad,float* lpfBetaGrad,const float* c_lpfGammaContributions,const float* c_lpfBetaContributions,int nFeatures,int nPositions);
__global__ void kernel_Mask_forward(float* lpfResult,const float* c_lpfInput,const float* c_lpfMask,int* lpnError,int nKey,int nTrailing,bool isBroadcast,int nRows);
__global__ void kernel_Mask_backward(float* lpfInputGrad,const float* c_lpfOutputGrad,const float* c_lpfMask,int nTrailing,bool isBroadcast,int nSize);
__global__ void kernel_Permute(float* lpfResult,const float* c_lpfInput,const std::int64_t* c_lpnOutputShape,const std::int64_t* c_lpnInputStrides,int nRank,int nSize);
__global__ void kernel_Softmax_forward(float* lpfResult,const float* c_lpfInput,int* lpnError,int nAxisSize,int nInner,int nSlices);
__global__ void kernel_Softmax_backward(float* lpfInputGrad,const float* c_lpfOutputGrad,const float* c_lpfOutput,int nAxisSize,int nInner,int nSlices);
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
__global__ void kernel_Embedding_forward(float* lpfOutput,const float* c_lpfWeight,const std::int32_t* c_lpnIndices,int nPositions,int nVocabSize,int nSize);
__global__ void kernel_Embedding_backward(float* lpfWeightGrad,const float* c_lpfOutputGrad,const std::int32_t* c_lpnIndices,int nPositions,int nVocabSize,int nSize);

static void requireContiguousTensor(const cufMat& value,const char* operation)
{
    if( !value.isContiguous() )
    {
        throw std::invalid_argument(
            std::string(operation)+": contiguous tensor required"
        );
    }
}

void cu_detail::fillOnes(float* destination,std::size_t elements)
{
    int nSize       =static_cast<int>(elements);
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_fill<<<nBlocks,nThreads>>>(
        destination,
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
void cuda_fill(cufMat& mResult,float fValue)
{
    // R[:] = fValue

    requireContiguousTensor(mResult,"cuda_fill");
    int nSize       =static_cast<int>(mResult.numel());
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_fill<<<nBlocks,nThreads>>>(
        mResult.data(),
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
void cuda_mul_elementwise(cufMat& mResult,const cufMat& c_mA,const cufMat& c_mB)
{
    // R = c_mA ⦿ c_mB

    // 行列数の確認
    requireContiguousTensor(mResult,"cuda_mul_elementwise");
    requireContiguousTensor(c_mA,"cuda_mul_elementwise");
    requireContiguousTensor(c_mB,"cuda_mul_elementwise");
    if( c_mA.shape()!=c_mB.shape() )
    {
        throw std::runtime_error(
            "cuda_mul_elementwise: A and B size mismatch"
        );
    }
    if( mResult.shape()!=c_mA.shape() )
    {
        throw std::runtime_error(
            "cuda_mul_elementwise: result size mismatch"
        );
    }

    int nSize    =static_cast<int>(c_mA.numel());
    int nThreads =256;
    int nBlocks  =(nSize + nThreads - 1) / nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_mul_elementwise<<<nBlocks, nThreads>>>(
        mResult.data(),
        c_mA.data(),
        c_mB.data(),
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
void cuda_BatchMatMul_forward(
    cufMat& mResult,
    const cufMat& c_mA,
    const cufMat& c_mB,
    int nM,
    int nK,
    int nN,
    int nBatch
)
{
    requireContiguousTensor(mResult,"cuda_BatchMatMul_forward");
    requireContiguousTensor(c_mA,"cuda_BatchMatMul_forward");
    requireContiguousTensor(c_mB,"cuda_BatchMatMul_forward");
    if( (nM<0)||(nK<0)||(nN<0)||(nBatch<0) )
    {
        throw std::invalid_argument(
            "cuda_BatchMatMul_forward: negative dimension"
        );
    }
    const auto checkedElements =[](int nFirst,int nSecond,int nThird)
    {
        std::size_t nElements =static_cast<std::size_t>(nFirst);
        for( const int nExtent:{nSecond,nThird} )
        {
            if( (nExtent!=0)&&
                (nElements>static_cast<std::size_t>(INT_MAX)/nExtent) )
            {
                throw std::overflow_error(
                    "cuda_BatchMatMul_forward: tensor is too large"
                );
            }
            nElements *=static_cast<std::size_t>(nExtent);
        }
        return nElements;
    };
    if( (c_mA.numel()!=checkedElements(nM,nK,nBatch))||
        (c_mB.numel()!=checkedElements(nK,nN,nBatch))||
        (mResult.numel()!=checkedElements(nM,nN,nBatch)) )
    {
        throw std::invalid_argument(
            "cuda_BatchMatMul_forward: tensor size mismatch"
        );
    }

    const int nSize =static_cast<int>(mResult.numel());
    if( nSize<=0 ) return;
    const int nThreads =256;
    const int nBlocks =(nSize+nThreads-1)/nThreads;
    //
    kernel_BatchMatMul_forward<<<nBlocks,nThreads>>>(
        mResult.data(),
        c_mA.data(),
        c_mB.data(),
        nK,
        nN,
        nBatch,
        nSize
    );
    const cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_BatchMatMul_forward: kernel launch failed: ")+
            cudaGetErrorString(cudError)
        );
    }
}
void cuda_BatchMatMul_backward(
    cufMat& mAGrad,
    cufMat& mBGrad,
    const cufMat& c_mOutputGrad,
    const cufMat& c_mA,
    const cufMat& c_mB,
    int nM,
    int nK,
    int nN,
    int nBatch
)
{
    requireContiguousTensor(mAGrad,"cuda_BatchMatMul_backward");
    requireContiguousTensor(mBGrad,"cuda_BatchMatMul_backward");
    requireContiguousTensor(c_mOutputGrad,"cuda_BatchMatMul_backward");
    requireContiguousTensor(c_mA,"cuda_BatchMatMul_backward");
    requireContiguousTensor(c_mB,"cuda_BatchMatMul_backward");
    if( (nM<0)||(nK<0)||(nN<0)||(nBatch<0) )
    {
        throw std::invalid_argument(
            "cuda_BatchMatMul_backward: negative dimension"
        );
    }
    const auto checkedElements =[](int nFirst,int nSecond,int nThird)
    {
        std::size_t nElements =static_cast<std::size_t>(nFirst);
        for( const int nExtent:{nSecond,nThird} )
        {
            if( (nExtent!=0)&&
                (nElements>static_cast<std::size_t>(INT_MAX)/nExtent) )
            {
                throw std::overflow_error(
                    "cuda_BatchMatMul_backward: tensor is too large"
                );
            }
            nElements *=static_cast<std::size_t>(nExtent);
        }
        return nElements;
    };
    const std::size_t nAElements =checkedElements(nM,nK,nBatch);
    const std::size_t nBElements =checkedElements(nK,nN,nBatch);
    const std::size_t nOutputElements =checkedElements(nM,nN,nBatch);
    if( (mAGrad.numel()!=nAElements)||(c_mA.numel()!=nAElements)||
        (mBGrad.numel()!=nBElements)||(c_mB.numel()!=nBElements)||
        (c_mOutputGrad.numel()!=nOutputElements) )
    {
        throw std::invalid_argument(
            "cuda_BatchMatMul_backward: tensor size mismatch"
        );
    }

    const int nThreads =256;
    const int nASize =static_cast<int>(mAGrad.numel());
    if( nASize>0 )
    {
        kernel_BatchMatMul_backward_A<<<
            (nASize+nThreads-1)/nThreads,nThreads
        >>>(
            mAGrad.data(),
            c_mOutputGrad.data(),
            c_mB.data(),
            nK,
            nN,
            nBatch,
            nASize
        );
        //
        const cudaError_t cudError =cudaGetLastError();
        if( cudError!=cudaSuccess )
        {
            throw std::runtime_error(
                std::string("cuda_BatchMatMul_backward: A kernel launch failed: ")+
                cudaGetErrorString(cudError)
            );
        }
    }

    const int nBSize =static_cast<int>(mBGrad.numel());
    if( nBSize>0 )
    {
        kernel_BatchMatMul_backward_B<<<
            (nBSize+nThreads-1)/nThreads,nThreads
        >>>(
            mBGrad.data(),
            c_mOutputGrad.data(),
            c_mA.data(),
            nM,
            nK,
            nN,
            nBatch,
            nBSize
        );
        //
        const cudaError_t cudError =cudaGetLastError();
        if( cudError!=cudaSuccess )
        {
            throw std::runtime_error(
                std::string("cuda_BatchMatMul_backward: B kernel launch failed: ")+
                cudaGetErrorString(cudError)
            );
        }
    }
}
void cuda_LayerNorm_forward(
    cufMat& mResult,
    const cufMat& c_mInput,
    const cufMat* c_lpmGamma,
    const cufMat* c_lpmBeta,
    int nFeatures,
    int nPositions,
    float fEpsilon
)
{
    requireContiguousTensor(mResult,"cuda_LayerNorm_forward");
    requireContiguousTensor(c_mInput,"cuda_LayerNorm_forward");
    if( (nFeatures<=0)||(nPositions<0)||
        (static_cast<std::size_t>(nFeatures)*nPositions!=c_mInput.numel())||
        (mResult.shape()!=c_mInput.shape())||
        ((c_lpmGamma==nullptr)!=(c_lpmBeta==nullptr)) )
    {
        throw std::invalid_argument(
            "cuda_LayerNorm_forward: tensor shape mismatch"
        );
    }
    if( c_lpmGamma!=nullptr )
    {
        requireContiguousTensor(*c_lpmGamma,"cuda_LayerNorm_forward");
        requireContiguousTensor(*c_lpmBeta,"cuda_LayerNorm_forward");
        if( (c_lpmGamma->shape()!=
             std::vector<std::int64_t>{nFeatures,1})||
            (c_lpmBeta->shape()!=c_lpmGamma->shape()) )
        {
            throw std::invalid_argument(
                "cuda_LayerNorm_forward: parameter shape mismatch"
            );
        }
    }
    if( nPositions==0 ) return;

    const int nThreads =256;
    kernel_LayerNorm_forward<<<
        (nPositions+nThreads-1)/nThreads,nThreads
    >>>(
        mResult.data(),c_mInput.data(),
        c_lpmGamma==nullptr ? nullptr : c_lpmGamma->data(),
        c_lpmBeta==nullptr ? nullptr : c_lpmBeta->data(),
        nFeatures,nPositions,fEpsilon
    );
    const cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_LayerNorm_forward: kernel launch failed: ")+
            cudaGetErrorString(cudError)
        );
    }
}
void cuda_LayerNorm_backward(
    cufMat& mInputGrad,
    cufMat* lpmGammaGrad,
    cufMat* lpmBetaGrad,
    const cufMat& c_mOutputGrad,
    const cufMat& c_mInput,
    const cufMat* c_lpmGamma,
    int nFeatures,
    int nPositions,
    float fEpsilon
)
{
    requireContiguousTensor(mInputGrad,"cuda_LayerNorm_backward");
    requireContiguousTensor(c_mOutputGrad,"cuda_LayerNorm_backward");
    requireContiguousTensor(c_mInput,"cuda_LayerNorm_backward");
    if( (nFeatures<=0)||(nPositions<0)||
        (static_cast<std::size_t>(nFeatures)*nPositions!=c_mInput.numel())||
        (mInputGrad.shape()!=c_mInput.shape())||
        (c_mOutputGrad.shape()!=c_mInput.shape())||
        ((c_lpmGamma==nullptr)!=(lpmGammaGrad==nullptr))||
        ((c_lpmGamma==nullptr)!=(lpmBetaGrad==nullptr)) )
    {
        throw std::invalid_argument(
            "cuda_LayerNorm_backward: tensor shape mismatch"
        );
    }
    if( c_lpmGamma!=nullptr )
    {
        requireContiguousTensor(*c_lpmGamma,"cuda_LayerNorm_backward");
        requireContiguousTensor(*lpmGammaGrad,"cuda_LayerNorm_backward");
        requireContiguousTensor(*lpmBetaGrad,"cuda_LayerNorm_backward");
        const std::vector<std::int64_t> parameterShape{nFeatures,1};
        if( (c_lpmGamma->shape()!=parameterShape)||
            (lpmGammaGrad->shape()!=parameterShape)||
            (lpmBetaGrad->shape()!=parameterShape) )
        {
            throw std::invalid_argument(
                "cuda_LayerNorm_backward: parameter shape mismatch"
            );
        }
    }
    if( nPositions==0 ) return;

    const int nThreads =256;
    cufMat mGammaContributions;
    cufMat mBetaContributions;
    if( lpmGammaGrad!=nullptr )
    {
        mGammaContributions =cufMat({nFeatures,nPositions});
        mBetaContributions =cufMat({nFeatures,nPositions});
    }
    kernel_LayerNorm_backward<<<
        (nPositions+nThreads-1)/nThreads,nThreads
    >>>(
        mInputGrad.data(),
        lpmGammaGrad==nullptr ? nullptr : mGammaContributions.data(),
        lpmBetaGrad==nullptr ? nullptr : mBetaContributions.data(),
        c_mOutputGrad.data(),c_mInput.data(),
        c_lpmGamma==nullptr ? nullptr : c_lpmGamma->data(),
        nFeatures,nPositions,fEpsilon
    );
    if( lpmGammaGrad!=nullptr )
    {
        kernel_LayerNorm_parameter_reduce<<<(nFeatures+nThreads-1)/nThreads,nThreads>>>(
            lpmGammaGrad->data(),lpmBetaGrad->data(),mGammaContributions.data(),
            mBetaContributions.data(),nFeatures,nPositions
        );
    }
    const cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_LayerNorm_backward: kernel launch failed: ")+
            cudaGetErrorString(cudError)
        );
    }
}
void cuda_Mask_forward(
    cufMat& mResult,
    const cufMat& c_mInput,
    const cufMat& c_mMask,
    int nQuery,
    int nKey,
    int nTrailing,
    bool isBroadcast
)
{
    requireContiguousTensor(mResult,"cuda_Mask_forward");
    requireContiguousTensor(c_mInput,"cuda_Mask_forward");
    requireContiguousTensor(c_mMask,"cuda_Mask_forward");
    const std::size_t nElements =
        static_cast<std::size_t>(nQuery)*nKey*nTrailing;
    const std::size_t nMaskElements =isBroadcast ?
        static_cast<std::size_t>(nQuery)*nKey : nElements;
    if( (nQuery<=0)||(nKey<=0)||(nTrailing<0)||
        (c_mInput.numel()!=nElements)||
        (mResult.shape()!=c_mInput.shape())||
        (c_mMask.numel()!=nMaskElements) )
    {
        throw std::invalid_argument("cuda_Mask_forward: tensor shape mismatch");
    }
    const int nRows =nQuery*nTrailing;
    if( nRows<=0 ) return;

    cu_memory::Buffer errorBuffer(sizeof(int));
    int* lpnError =static_cast<int*>(errorBuffer.data());
    cudaError_t cudError =cudaMemset(lpnError,0,sizeof(int));
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_Mask_forward: error flag allocation failed: ")+
            cudaGetErrorString(cudError)
        );
    }

    const int nThreads =256;
    kernel_Mask_forward<<<(nRows+nThreads-1)/nThreads,nThreads>>>(
        mResult.data(),c_mInput.data(),c_mMask.data(),lpnError,
        nKey,nTrailing,isBroadcast,nRows
    );
    cudError =cudaGetLastError();
    int nError =0;
    if( cudError==cudaSuccess )
    {
        cudError =cudaMemcpy(
            &nError,lpnError,sizeof(int),cudaMemcpyDeviceToHost
        );
    }
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_Mask_forward: CUDA operation failed: ")+
            cudaGetErrorString(cudError)
        );
    }
    errorBuffer.release();
    if( (nError&1)!=0 )
    {
        throw std::invalid_argument(
            "cuda_Mask_forward: mask must be finite"
        );
    }
    if( (nError&2)!=0 )
    {
        throw std::invalid_argument(
            "cuda_Mask_forward: every key is masked"
        );
    }
}
void cuda_Mask_backward(
    cufMat& mInputGrad,
    const cufMat& c_mOutputGrad,
    const cufMat& c_mMask,
    int nQuery,
    int nKey,
    int nTrailing,
    bool isBroadcast
)
{
    requireContiguousTensor(mInputGrad,"cuda_Mask_backward");
    requireContiguousTensor(c_mOutputGrad,"cuda_Mask_backward");
    requireContiguousTensor(c_mMask,"cuda_Mask_backward");
    const std::size_t nElements =
        static_cast<std::size_t>(nQuery)*nKey*nTrailing;
    const std::size_t nMaskElements =isBroadcast ?
        static_cast<std::size_t>(nQuery)*nKey : nElements;
    if( (nQuery<=0)||(nKey<=0)||(nTrailing<0)||
        (mInputGrad.numel()!=nElements)||
        (mInputGrad.shape()!=c_mOutputGrad.shape())||
        (c_mMask.numel()!=nMaskElements) )
    {
        throw std::invalid_argument(
            "cuda_Mask_backward: tensor shape mismatch"
        );
    }

    const int nSize =static_cast<int>(mInputGrad.numel());
    if( nSize<=0 ) return;
    const int nThreads =256;
    kernel_Mask_backward<<<(nSize+nThreads-1)/nThreads,nThreads>>>(
        mInputGrad.data(),c_mOutputGrad.data(),c_mMask.data(),
        nTrailing,isBroadcast,nSize
    );
    const cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_Mask_backward: kernel launch failed: ")+
            cudaGetErrorString(cudError)
        );
    }
}
void cuda_Permute(
    cufMat& mResult,
    const cufMat& c_mInput,
    const std::vector<std::size_t>& c_nDimensions
)
{
    requireContiguousTensor(mResult,"cuda_Permute");
    requireContiguousTensor(c_mInput,"cuda_Permute");
    if( c_nDimensions.size()!=c_mInput.dim() )
    {
        throw std::invalid_argument("cuda_Permute: rank mismatch");
    }

    std::vector<bool> isUsed(c_nDimensions.size(),false);
    std::vector<std::int64_t> shapeOutput(c_nDimensions.size());
    std::vector<std::int64_t> nMetadata(c_nDimensions.size()*2);
    for( std::size_t nOutput=0;nOutput<c_nDimensions.size();++nOutput )
    {
        const std::size_t nInput =c_nDimensions[nOutput];
        if( (nInput>=c_nDimensions.size())||isUsed[nInput] )
        {
            throw std::invalid_argument("cuda_Permute: invalid permutation");
        }
        isUsed[nInput] =true;
        shapeOutput[nOutput] =c_mInput.size(nInput);
        nMetadata[nOutput] =shapeOutput[nOutput];
        nMetadata[c_nDimensions.size()+nOutput] =
            c_mInput.strides()[nInput];
    }
    if( mResult.shape()!=shapeOutput )
    {
        throw std::invalid_argument("cuda_Permute: output shape mismatch");
    }
    if( c_mInput.numel()>static_cast<std::size_t>(INT_MAX) )
    {
        throw std::overflow_error("cuda_Permute: tensor is too large");
    }

    const int nSize =static_cast<int>(mResult.numel());
    if( nSize<=0 ) return;
    cu_memory::Buffer metadataBuffer(nMetadata.size()*sizeof(std::int64_t));
    auto* lpnMetadata =static_cast<std::int64_t*>(metadataBuffer.data());
    cudaError_t cudError =cudaSuccess;
    if( cudError==cudaSuccess )
    {
        cudError =cudaMemcpy(
            lpnMetadata,nMetadata.data(),
            nMetadata.size()*sizeof(std::int64_t),
            cudaMemcpyHostToDevice
        );
    }
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_Permute: metadata transfer failed: ")+
            cudaGetErrorString(cudError)
        );
    }

    const int nThreads =256;
    kernel_Permute<<<(nSize+nThreads-1)/nThreads,nThreads>>>(
        mResult.data(),c_mInput.data(),lpnMetadata,
        lpnMetadata+c_nDimensions.size(),
        static_cast<int>(c_nDimensions.size()),nSize
    );
    cudError =cudaGetLastError();
    if( cudError==cudaSuccess ) cudError =cudaDeviceSynchronize();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_Permute: kernel failed: ")+
            cudaGetErrorString(cudError)
        );
    }
    metadataBuffer.release();
}
void cuda_Softmax_forward(
    cufMat& mResult,
    const cufMat& c_mInput,
    int nOuter,
    int nAxisSize,
    int nInner,
    int nSlices
)
{
    requireContiguousTensor(mResult,"cuda_Softmax_forward");
    requireContiguousTensor(c_mInput,"cuda_Softmax_forward");
    if( (nOuter<0)||(nAxisSize<=0)||(nInner<0)||(nSlices<0)||
        (static_cast<std::size_t>(nOuter)*nAxisSize*nInner!=
         c_mInput.numel())||
        (nSlices!=nOuter*nInner)||(mResult.shape()!=c_mInput.shape()) )
    {
        throw std::invalid_argument(
            "cuda_Softmax_forward: tensor shape mismatch"
        );
    }
    if( nSlices<=0 ) return;

    cu_memory::Buffer errorBuffer(sizeof(int));
    int* lpnError =static_cast<int*>(errorBuffer.data());
    cudaError_t cudError =cudaMemset(lpnError,0,sizeof(int));
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_Softmax_forward: error flag allocation failed: ")+
            cudaGetErrorString(cudError)
        );
    }

    const int nThreads =256;
    kernel_Softmax_forward<<<
        (nSlices+nThreads-1)/nThreads,nThreads
    >>>(
        mResult.data(),c_mInput.data(),lpnError,
        nAxisSize,nInner,nSlices
    );
    cudError =cudaGetLastError();
    int nError =0;
    if( cudError==cudaSuccess )
    {
        cudError =cudaMemcpy(
            &nError,lpnError,sizeof(int),cudaMemcpyDeviceToHost
        );
    }
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_Softmax_forward: CUDA operation failed: ")+
            cudaGetErrorString(cudError)
        );
    }
    errorBuffer.release();
    if( (nError&1)!=0 )
    {
        throw std::invalid_argument(
            "cuda_Softmax_forward: NaN or positive infinity"
        );
    }
    if( (nError&2)!=0 )
    {
        throw std::invalid_argument(
            "cuda_Softmax_forward: all values are negative infinity"
        );
    }
}
void cuda_Softmax_backward(
    cufMat& mInputGrad,
    const cufMat& c_mOutputGrad,
    const cufMat& c_mOutput,
    int nOuter,
    int nAxisSize,
    int nInner,
    int nSlices
)
{
    requireContiguousTensor(mInputGrad,"cuda_Softmax_backward");
    requireContiguousTensor(c_mOutputGrad,"cuda_Softmax_backward");
    requireContiguousTensor(c_mOutput,"cuda_Softmax_backward");
    if( (nOuter<0)||(nAxisSize<=0)||(nInner<0)||(nSlices<0)||
        (static_cast<std::size_t>(nOuter)*nAxisSize*nInner!=
         mInputGrad.numel())||
        (nSlices!=nOuter*nInner)||
        (c_mOutputGrad.shape()!=mInputGrad.shape())||
        (c_mOutput.shape()!=mInputGrad.shape()) )
    {
        throw std::invalid_argument(
            "cuda_Softmax_backward: tensor shape mismatch"
        );
    }
    if( nSlices<=0 ) return;

    const int nThreads =256;
    kernel_Softmax_backward<<<
        (nSlices+nThreads-1)/nThreads,nThreads
    >>>(
        mInputGrad.data(),c_mOutputGrad.data(),c_mOutput.data(),
        nAxisSize,nInner,nSlices
    );
    const cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_Softmax_backward: kernel launch failed: ")+
            cudaGetErrorString(cudError)
        );
    }
}
void cuda_ReLU_forward(cufMat& mResult,const cufMat& c_mValue)
{
    requireContiguousTensor(mResult,"cuda_ReLU_forward");
    requireContiguousTensor(c_mValue,"cuda_ReLU_forward");
    if( mResult.shape()!=c_mValue.shape() )
    {
        throw std::runtime_error(
            "cuda_ReLU_forward: matrix size mismatch"
        );
    }
    //
    int nSize       =static_cast<int>(c_mValue.numel());
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_ReLU_forward<<<nBlocks,nThreads>>>(
        mResult.data(),
        c_mValue.data(),
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
void cuda_ReLU_backward(cufMat& mResult,const cufMat& c_mData,const cufMat& c_mGrad)
{
    requireContiguousTensor(mResult,"cuda_ReLU_backward");
    requireContiguousTensor(c_mData,"cuda_ReLU_backward");
    requireContiguousTensor(c_mGrad,"cuda_ReLU_backward");
    if( mResult.shape()!=c_mData.shape()||c_mData.shape()!=c_mGrad.shape() )
    {
        throw std::runtime_error(
            "cuda_ReLU_backward: matrix size mismatch"
        );
    }
    //
    int nSize       =static_cast<int>(c_mData.numel());
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_ReLU_backward<<<nBlocks,nThreads>>>(
        mResult.data(),
        c_mData.data(),
        c_mGrad.data(),
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
void cuda_GELU_forward(cufMat& mResult,const cufMat& c_mValue)
{
    requireContiguousTensor(mResult,"cuda_GELU_forward");
    requireContiguousTensor(c_mValue,"cuda_GELU_forward");
    if( mResult.shape()!=c_mValue.shape() )
    {
        throw std::runtime_error(
            "cuda_ReLU_forward: matrix size mismatch"
        );
    }
    //
    int nSize       =static_cast<int>(c_mValue.numel());
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_GELU_forward<<<nBlocks,nThreads>>>(
        mResult.data(),
        c_mValue.data(),
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
void cuda_GELU_backward(cufMat& mResult,const cufMat& c_mData,const cufMat& c_mGrad)
{
    requireContiguousTensor(mResult,"cuda_GELU_backward");
    requireContiguousTensor(c_mData,"cuda_GELU_backward");
    requireContiguousTensor(c_mGrad,"cuda_GELU_backward");
    if( mResult.shape()!=c_mData.shape()||c_mData.shape()!=c_mGrad.shape() )
    {
        throw std::runtime_error(
            "cuda_GELU_backward: matrix size mismatch"
        );
    }
    //
    int nSize       =static_cast<int>(c_mData.numel());
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_GELU_backward<<<nBlocks,nThreads>>>(
        mResult.data(),
        c_mData.data(),
        c_mGrad.data(),
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
void cuda_Dropout_forward(cufMat& mResult,const cufMat& c_mValue,cufMat& mMask,float fDropProbability)
{
    requireContiguousTensor(mResult,"cuda_Dropout_forward");
    requireContiguousTensor(c_mValue,"cuda_Dropout_forward");
    requireContiguousTensor(mMask,"cuda_Dropout_forward");
    if( mResult.shape()!=c_mValue.shape()||mMask.shape()!=c_mValue.shape() )
    {
        throw std::runtime_error("cuda_Dropout_forward: matrix size mismatch");
    }
    const int nSize =static_cast<int>(c_mValue.numel());
    if( nSize<=0 ) {return;}
    const int nThreads =256;
    const int nBlocks =(nSize+nThreads-1)/nThreads;
    kernel_Dropout_forward<<<nBlocks,nThreads>>>(
        mResult.data(),c_mValue.data(),mMask.data(),
        fDropProbability,1.0f/(1.0f-fDropProbability),nSize
    );
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(std::string("cuda_Dropout_forward: kernel launch failed: ")+cudaGetErrorString(cudError));
    }
}
void cuda_Dropout_backward(cufMat& mResult,const cufMat& c_mGrad,const cufMat& c_mMask)
{
    requireContiguousTensor(mResult,"cuda_Dropout_backward");
    requireContiguousTensor(c_mGrad,"cuda_Dropout_backward");
    requireContiguousTensor(c_mMask,"cuda_Dropout_backward");
    if( mResult.shape()!=c_mGrad.shape()||c_mMask.shape()!=c_mGrad.shape() )
    {
        throw std::runtime_error("cuda_Dropout_backward: matrix size mismatch");
    }
    const int nSize =static_cast<int>(c_mGrad.numel());
    if( nSize<=0 ) {return;}
    const int nThreads =256;
    const int nBlocks =(nSize+nThreads-1)/nThreads;
    kernel_Dropout_backward<<<nBlocks,nThreads>>>(mResult.data(),c_mGrad.data(),c_mMask.data(),nSize);
    cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(std::string("cuda_Dropout_backward: kernel launch failed: ")+cudaGetErrorString(cudError));
    }
}
void cuda_BatchNorm_forward_training(
    cufMat& mResult,const cufMat& c_mValue,
    const cufMat& c_mGamma,const cufMat& c_mBeta,
    cufMat& mRunningMean,cufMat& mRunningVar,
    cufMat& mNormalized,cufMat& mInvStd,
    float fMomentum,float fEpsilon
)
{
    const int nFeatures =c_mValue.rows();
    const int nBatch    =c_mValue.cols();
    if( (nBatch<=0)||(mResult.rows()!=nFeatures)||(mResult.cols()!=nBatch)||
        (c_mGamma.rows()!=nFeatures)||(c_mGamma.cols()!=1)||
        (c_mBeta.rows()!=nFeatures)||(c_mBeta.cols()!=1)||
        (mRunningMean.rows()!=nFeatures)||(mRunningMean.cols()!=1)||
        (mRunningVar.rows()!=nFeatures)||(mRunningVar.cols()!=1)||
        (mNormalized.rows()!=nFeatures)||(mNormalized.cols()!=nBatch)||
        (mInvStd.rows()!=nFeatures)||(mInvStd.cols()!=1) )
    {
        throw std::runtime_error("cuda_BatchNorm_forward_training: matrix size mismatch");
    }
    if( nFeatures<=0 ) {return;}
    const int nThreads =256;
    const int nBlocks =(nFeatures+nThreads-1)/nThreads;
    //
    kernel_BatchNorm_forward_training<<<nBlocks,nThreads>>>(
        mResult.data(),
        c_mValue.data(),
        c_mGamma.data(),
        c_mBeta.data(),
        mRunningMean.data(),
        mRunningVar.data(),
        mNormalized.data(),
        mInvStd.data(),
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
    cufMat& mResult,const cufMat& c_mValue,
    const cufMat& c_mGamma,const cufMat& c_mBeta,
    const cufMat& c_mRunningMean,const cufMat& c_mRunningVar,
    cufMat& mNormalized,cufMat& mInvStd,float fEpsilon
)
{
    const int nFeatures =c_mValue.rows();
    const int nBatch =c_mValue.cols();
    if( (nBatch<=0)||(mResult.rows()!=nFeatures)||(mResult.cols()!=nBatch)||
        (c_mGamma.rows()!=nFeatures)||(c_mGamma.cols()!=1)||
        (c_mBeta.rows()!=nFeatures)||(c_mBeta.cols()!=1)||
        (c_mRunningMean.rows()!=nFeatures)||(c_mRunningMean.cols()!=1)||
        (c_mRunningVar.rows()!=nFeatures)||(c_mRunningVar.cols()!=1)||
        (mNormalized.rows()!=nFeatures)||(mNormalized.cols()!=nBatch)||
        (mInvStd.rows()!=nFeatures)||(mInvStd.cols()!=1) )
    {
        throw std::runtime_error("cuda_BatchNorm_forward_evaluation: matrix size mismatch");
    }
    if( nFeatures<=0 ) {return;}
    const int nThreads =256;
    const int nBlocks =(nFeatures+nThreads-1)/nThreads;
    //
    kernel_BatchNorm_forward_evaluation<<<nBlocks,nThreads>>>(
        mResult.data(),
        c_mValue.data(),
        c_mGamma.data(),
        c_mBeta.data(),
        c_mRunningMean.data(),
        c_mRunningVar.data(),
        mNormalized.data(),
        mInvStd.data(),
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
    cufMat& mInputGrad,cufMat& mGammaGrad,cufMat& mBetaGrad,
    const cufMat& c_mOutputGrad,const cufMat& c_mGamma,
    const cufMat& c_mNormalized,const cufMat& c_mInvStd,
    bool isTraining
)
{
    const int nFeatures =c_mOutputGrad.rows();
    const int nBatch =c_mOutputGrad.cols();
    if( (nBatch<=0)||(mInputGrad.rows()!=nFeatures)||(mInputGrad.cols()!=nBatch)||
        (mGammaGrad.rows()!=nFeatures)||(mGammaGrad.cols()!=1)||
        (mBetaGrad.rows()!=nFeatures)||(mBetaGrad.cols()!=1)||
        (c_mGamma.rows()!=nFeatures)||(c_mGamma.cols()!=1)||
        (c_mNormalized.rows()!=nFeatures)||(c_mNormalized.cols()!=nBatch)||
        (c_mInvStd.rows()!=nFeatures)||(c_mInvStd.cols()!=1) )
    {
        throw std::runtime_error("cuda_BatchNorm_backward: matrix size mismatch");
    }
    if( nFeatures<=0 ) {return;}
    const int nThreads  =256;
    const int nBlocks   =(nFeatures+nThreads-1)/nThreads;
    //
    kernel_BatchNorm_backward<<<nBlocks,nThreads>>>(
        mInputGrad.data(),
        mGammaGrad.data(),
        mBetaGrad.data(),
        c_mOutputGrad.data(),
        c_mGamma.data(),
        c_mNormalized.data(),
        c_mInvStd.data(),
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
    cufMat& mResult,
    const cufMat& c_mLogits,
    const cufMat& c_mTarget
)
{
    if( (c_mLogits.rows()!=c_mTarget.rows())||(c_mLogits.cols()!=c_mTarget.cols()) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_forward: matrix size mismatch"
        );
    }
    if( (mResult.rows()!=1)||(mResult.cols()!=1) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_forward: result must be 1x1"
        );
    }
    //
    int nClass      =c_mLogits.rows();
    int nBatch      =c_mLogits.cols();
    int nThreads    =256;
    int nBlocks     =(nBatch + nThreads - 1)/nThreads;
    if( (nClass<=0)||(nBatch<=0) )  {return;}
    //
    cuda_fill( mResult,0.0f );      // atomicAddするので最初は0
    kernel_SoftmaxCrossEntropy_forward<<<nBlocks,nThreads>>>(
        mResult.data(),
        c_mLogits.data(),
        c_mTarget.data(),
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
    cufMat& mLogitsGrad,
    const cufMat& c_mLogits,
    const cufMat& c_mTarget,
    const cufMat& c_mGrad
)
{
    if( (c_mLogits.rows()!=c_mTarget.rows())||(c_mLogits.cols()!=c_mTarget.cols()) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_backward: matrix size mismatch"
        );
    }
    if( (c_mLogits.rows()!=mLogitsGrad.rows())||(c_mLogits.cols()!=mLogitsGrad.cols()) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_backward: gradient size mismatch"
        );
    }
    if( (c_mGrad.rows()!=1)||(c_mGrad.cols()!=1) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_backward: grad must be 1x1"
        );
    }
    //
    int nClass      =c_mLogits.rows();
    int nBatch      =c_mLogits.cols();
    int nThreads    =256;
    int nBlocks     =(nBatch + nThreads - 1)/nThreads;
    if( (nClass<=0)||(nBatch<=0) )  {return;}
    //
    kernel_SoftmaxCrossEntropy_backward<<<nBlocks,nThreads>>>(
        mLogitsGrad.data(),
        c_mLogits.data(),
        c_mTarget.data(),
        c_mGrad.data(),
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
void cuda_Conv2D_im2col(
    cufMat& mResult,
    const cufMat& c_mInput,
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
    int nSize       =static_cast<int>(mResult.numel());
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Conv2D_im2col<<<nBlocks,nThreads>>>(
        mResult.data(),
        c_mInput.data(),
        nInputChannels,
        nInputHeight,
        nInputWidth,
        nKernelSize,
        nStride,
        nPadding,
        nOutputHeight,
        nOutputWidth,
        c_mInput.cols(),
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
    cufMat& mResult,
    const cufMat& c_mGemm,
    const cufMat& c_mBias,
    int nOutputHeight,
    int nOutputWidth
)
{
    int nSize       =static_cast<int>(mResult.numel());
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Conv2D_pack<<<nBlocks, nThreads>>>(
        mResult.data(),
        c_mGemm.data(),
        c_mBias.data(),
        c_mGemm.rows(),
        nOutputHeight*nOutputWidth,
        mResult.cols(),
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
void cuda_Conv2D_unpack_grad( cufMat& mResult, const cufMat& c_mOutputGrad, int nOutputChannels,
                              int nOutputHeight, int nOutputWidth )
{
    int nSize       =static_cast<int>(mResult.numel());
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Conv2D_unpack<<<nBlocks, nThreads>>>(
        mResult.data(),
        c_mOutputGrad.data(),
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
    cufMat& mResult,
    const cufMat& c_mColumnGrad,
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
    int nSize       =c_mColumnGrad.rows() * c_mColumnGrad.cols();
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Conv2D_col2im<<<nBlocks, nThreads>>>(
        mResult.data(),
        c_mColumnGrad.data(),
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
    cufMat& mResult,
    const cufMat& c_mGemmGrad
)
{
    int nSize       =c_mGemmGrad.rows() * c_mGemmGrad.cols();
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Conv2D_bias<<<nBlocks, nThreads>>>(
        mResult.data(),
        c_mGemmGrad.data(),
        c_mGemmGrad.rows(),
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
    cufMat& mResult,
    const cufMat& c_mInput,
    int nChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nOutputHeight,
    int nOutputWidth
)
{
    int nSize       =static_cast<int>(mResult.numel());
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Pooling_forward<<<nBlocks, nThreads>>>(
        mResult.data(),
        c_mInput.data(),
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
    cufMat& mResult,
    const cufMat& c_mInput,
    const cufMat& c_mOutputGrad,
    int nChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nOutputHeight,
    int nOutputWidth
)
{
    int nSize       =c_mOutputGrad.rows() * c_mOutputGrad.cols();
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_Pooling_backward<<<nBlocks, nThreads>>>(
        mResult.data(),
        c_mInput.data(),
        c_mOutputGrad.data(),
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

void cuda_Embedding_forward(
    cufMat& mResult,
    const cufMat& c_mWeight,
    const cunMat& c_mIndices
)
{
    if( (c_mWeight.dim()!=2)||(c_mIndices.dim()!=2) )
    {
        throw std::invalid_argument("cuda_Embedding_forward: invalid tensor rank");
    }
    if( !mResult.isContiguous()||!c_mWeight.isContiguous()||
        !c_mIndices.isContiguous() )
    {
        throw std::invalid_argument("cuda_Embedding_forward: contiguous tensors required");
    }
    const auto expected =std::vector<std::int64_t>{
        c_mWeight.size(0),c_mIndices.size(0),c_mIndices.size(1)
    };
    if( mResult.shape()!=expected )
    {
        throw std::invalid_argument("cuda_Embedding_forward: output shape mismatch");
    }
    if( (c_mWeight.size(0)>INT_MAX)||(c_mWeight.size(1)>INT_MAX)||
        (c_mIndices.numel()>static_cast<std::size_t>(INT_MAX))||
        (mResult.numel()>static_cast<std::size_t>(INT_MAX)) )
    {
        throw std::overflow_error("cuda_Embedding_forward: tensor is too large");
    }

    const int nVocabSize =static_cast<int>(c_mWeight.size(1));
    for( std::int32_t tokenId : c_mIndices.toHost() )
    {
        if( (tokenId<0)||(tokenId>=nVocabSize) )
        {
            throw std::out_of_range("cuda_Embedding_forward: token ID out of range");
        }
    }

    const int nSize     =static_cast<int>(mResult.numel());
    if( nSize<=0 ) return;
    const int nThreads  =256;
    const int nBlocks   =(nSize+nThreads-1)/nThreads;
    kernel_Embedding_forward<<<nBlocks,nThreads>>>(
        mResult.data(),
        c_mWeight.data(),
        c_mIndices.data(),
        static_cast<int>(c_mIndices.numel()),
        nVocabSize,nSize
    );
    //
    const cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_Embedding_forward: kernel launch failed: ")+
            cudaGetErrorString(cudError)
        );
    }
}
void cuda_Embedding_backward(
    cufMat& mWeightGrad,
    const cufMat& c_mOutputGrad,
    const cunMat& c_mIndices
)
{
    if( (mWeightGrad.dim()!=2)||(c_mIndices.dim()!=2) )
    {
        throw std::invalid_argument("cuda_Embedding_backward: invalid tensor rank");
    }
    if( !mWeightGrad.isContiguous()||!c_mOutputGrad.isContiguous()||
        !c_mIndices.isContiguous() )
    {
        throw std::invalid_argument("cuda_Embedding_backward: contiguous tensors required");
    }
    const auto expected =std::vector<std::int64_t>{
        mWeightGrad.size(0),c_mIndices.size(0),c_mIndices.size(1)
    };
    if( c_mOutputGrad.shape()!=expected )
    {
        throw std::invalid_argument("cuda_Embedding_backward: gradient shape mismatch");
    }
    if( (mWeightGrad.size(0)>INT_MAX)||(mWeightGrad.size(1)>INT_MAX)||
        (c_mIndices.numel()>static_cast<std::size_t>(INT_MAX))||
        (c_mOutputGrad.numel()>static_cast<std::size_t>(INT_MAX)) )
    {
        throw std::overflow_error("cuda_Embedding_backward: tensor is too large");
    }

    const int nVocabSize =static_cast<int>(mWeightGrad.size(1));
    for( std::int32_t tokenId : c_mIndices.toHost() )
    {
        if( (tokenId<0)||(tokenId>=nVocabSize) )
        {
            throw std::out_of_range("cuda_Embedding_backward: token ID out of range");
        }
    }

    const int nSize =static_cast<int>(mWeightGrad.size(0));
    if( nSize<=0 ) return;
    const int nThreads =256;
    const int nBlocks =(nSize+nThreads-1)/nThreads;
    kernel_Embedding_backward<<<nBlocks,nThreads>>>(
        mWeightGrad.data(),
        c_mOutputGrad.data(),
        c_mIndices.data(),
        static_cast<int>(c_mIndices.numel()),
        nVocabSize,nSize
    );
    //
    const cudaError_t cudError =cudaGetLastError();
    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuda_Embedding_backward: kernel launch failed: ")+
            cudaGetErrorString(cudError)
        );
    }
}
void cuda_Adam_update(
    cufMat& mData,
    const cufMat& c_mGrad,
    cufMat& mFirstMoment,
    cufMat& mSecondMoment,
    float fLearningRate,
    float fBeta1,
    float fBeta2,
    float fBeta1Correction,
    float fBeta2Correction,
    float fEpsilon
)
{
    requireContiguousTensor(mData,"cuda_Adam_update");
    requireContiguousTensor(c_mGrad,"cuda_Adam_update");
    requireContiguousTensor(mFirstMoment,"cuda_Adam_update");
    requireContiguousTensor(mSecondMoment,"cuda_Adam_update");
    if( mData.shape()!=c_mGrad.shape()||mData.shape()!=mFirstMoment.shape()||
        mData.shape()!=mSecondMoment.shape() )
    {
        throw std::runtime_error(
            "cuda_Adam_update: matrix size mismatch"
        );
    }

    int nSize       =static_cast<int>(mData.numel());
    int nThreads    =256;
    int nBlocks     =(nSize+nThreads-1)/nThreads;
    if( nSize<=0 )  {return;}
    //
    kernel_Adam_update<<<nBlocks,nThreads>>>(
        mData.data(),
        c_mGrad.data(),
        mFirstMoment.data(),
        mSecondMoment.data(),
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
__global__ void kernel_BatchMatMul_forward(
    float* lpfResult,
    const float* c_lpfA,
    const float* c_lpfB,
    int nK,
    int nN,
    int nBatch,
    int nSize
)
{
    const int nIndex =blockIdx.x*blockDim.x+threadIdx.x;
    if( nIndex>=nSize ) return;

    // matrix is arrayed in [M,K,Batch] → [1,10, 2,20, 3,30, 4,40]
    const int nBatchIndex   =nIndex%nBatch;
    const int nMatrixIndex  =nIndex/nBatch;
    const int nColumn       =nMatrixIndex%nN;
    const int nRow          =nMatrixIndex/nN;
    float fResult =0.0f;
    for( int nInner=0;nInner<nK;++nInner )
    {
        fResult +=c_lpfA[(nRow*nK+nInner)*nBatch+nBatchIndex]*
                  c_lpfB[(nInner*nN+nColumn)*nBatch+nBatchIndex];
    }
    lpfResult[nIndex] =fResult;
}

__global__ void kernel_BatchMatMul_backward_A(
    float* lpfAGrad,
    const float* c_lpfOutputGrad,
    const float* c_lpfB,
    int nK,
    int nN,
    int nBatch,
    int nSize
)
{
    const int nIndex =blockIdx.x*blockDim.x+threadIdx.x;
    if( nIndex>=nSize ) return;

    // matrix is arrayed in [M,K,Batch] → [1,10, 2,20, 3,30, 4,40]
    const int nBatchIndex   =nIndex%nBatch;
    const int nMatrixIndex  =nIndex/nBatch;
    const int nInner        =nMatrixIndex%nK;
    const int nRow          =nMatrixIndex/nK;
    float fGrad =0.0f;
    for( int nColumn=0;nColumn<nN;++nColumn )
    {
        fGrad +=c_lpfOutputGrad[(nRow*nN+nColumn)*nBatch+nBatchIndex]*
                 c_lpfB[(nInner*nN+nColumn)*nBatch+nBatchIndex];
    }
    lpfAGrad[nIndex] +=fGrad;
}

__global__ void kernel_BatchMatMul_backward_B(
    float* lpfBGrad,
    const float* c_lpfOutputGrad,
    const float* c_lpfA,
    int nM,     // A token
    int nK,     // head
    int nN,     // B token
    int nBatch, // head x batch
    int nSize   // A token x B token x head x batch
)
{
    const int nIndex =blockIdx.x*blockDim.x+threadIdx.x;
    if( nIndex>=nSize ) return;

    // matrix is arrayed in [M,K,Batch] → [1,10, 2,20, 3,30, 4,40]
    const int nBatchIndex   =nIndex%nBatch;
    const int nMatrixIndex  =nIndex/nBatch;
    const int nColumn       =nMatrixIndex%nN;   // 列
    const int nInner        =nMatrixIndex/nN;   // 行
    float fGrad =0.0f;
    for( int nRow=0;nRow<nM;++nRow )
    {
        // dB += A^T x dY
        fGrad +=c_lpfA[(nRow*nK+nInner)*nBatch+nBatchIndex]*
                 c_lpfOutputGrad[(nRow*nN+nColumn)*nBatch+nBatchIndex];
    }
    lpfBGrad[nIndex] +=fGrad;
}

__global__ void kernel_LayerNorm_forward(
    float* lpfResult,
    const float* c_lpfInput,
    const float* c_lpfGamma,
    const float* c_lpfBeta,
    int nFeatures,
    int nPositions,
    float fEpsilon
)
{
    const int nPosition =blockIdx.x*blockDim.x+threadIdx.x;
    if( nPosition>=nPositions ) return;

    double dblMean =0.0;
    double dblM2 =0.0;
    for( int nFeature=0;nFeature<nFeatures;++nFeature )
    {
        const double dblValue =
            c_lpfInput[nFeature*nPositions+nPosition];
        const double dblDelta =dblValue-dblMean;
        dblMean +=dblDelta/static_cast<double>(nFeature+1);
        dblM2 +=dblDelta*(dblValue-dblMean);
    }
    const double dblInvStd =1.0/sqrt(
        dblM2/static_cast<double>(nFeatures)+
        static_cast<double>(fEpsilon)
    );
    for( int nFeature=0;nFeature<nFeatures;++nFeature )
    {
        const int nIndex =nFeature*nPositions+nPosition;
        const double dblNormalized =
            (static_cast<double>(c_lpfInput[nIndex])-dblMean)*dblInvStd;
        const double dblGamma =
            c_lpfGamma==nullptr ? 1.0 : c_lpfGamma[nFeature];
        const double dblBeta =
            c_lpfBeta==nullptr ? 0.0 : c_lpfBeta[nFeature];
        lpfResult[nIndex] =static_cast<float>(
            dblGamma*dblNormalized+dblBeta
        );
    }
}
__global__ void kernel_LayerNorm_backward(
    float* lpfInputGrad,
    float* lpfGammaGrad,
    float* lpfBetaGrad,
    const float* c_lpfOutputGrad,
    const float* c_lpfInput,
    const float* c_lpfGamma,
    int nFeatures,
    int nPositions,
    float fEpsilon
)
{
    // r = 1/sqrt{v + e}
    // xhat_i =(x_i - mean)*r
    // y_i = γ_i*xhat_i + β_i
    //
    const int nPosition =blockIdx.x*blockDim.x+threadIdx.x;
    if( nPosition>=nPositions ) return;

    // Welford's online algorithm for mean and variance
    // mean = 1/F *sum_i{x_i}
    // variance = v = 1/F *sim_i{(d_i - mean)^2}
    double dblMean =0.0;
    double dblM2 =0.0;
    for( int nFeature=0;nFeature<nFeatures;++nFeature )
    {
        const double dblValue =
            c_lpfInput[nFeature*nPositions+nPosition];
        const double dblDelta =dblValue-dblMean;
        dblMean +=dblDelta/static_cast<double>(nFeature+1);
        dblM2 +=dblDelta*(dblValue-dblMean);
    }
    const double dblInvStd =1.0/sqrt(
        dblM2/static_cast<double>(nFeatures)+
        static_cast<double>(fEpsilon)
    );

    // dL/dγ = sum_p{g_(i,p)*xhat_(i,p)}
    // dL/dβ = sum_p{g_(i,p)}
    double dblGradSum =0.0;
    double dblGradNormalizedSum =0.0;
    for( int nFeature=0;nFeature<nFeatures;++nFeature )
    {
        const int nIndex =nFeature*nPositions+nPosition;
        const double dblNormalized =
            (static_cast<double>(c_lpfInput[nIndex])-dblMean)*dblInvStd;
        const double dblOutputGrad =c_lpfOutputGrad[nIndex];
        const double dblGamma =
            c_lpfGamma==nullptr ? 1.0 : c_lpfGamma[nFeature];
        const double dblNormalizedGrad =dblOutputGrad*dblGamma;
        dblGradSum +=dblNormalizedGrad;
        dblGradNormalizedSum +=dblNormalizedGrad*dblNormalized;
        if( lpfGammaGrad!=nullptr )
        {
            lpfGammaGrad[nFeature*nPositions+nPosition] =
                static_cast<float>(dblOutputGrad*dblNormalized);
            lpfBetaGrad[nFeature*nPositions+nPosition] =static_cast<float>(dblOutputGrad);
        }
    }

    // dL/dx_i = r(h_i - mean(h) - xhat_i*mean(h*xhat))
    for( int nFeature=0;nFeature<nFeatures;++nFeature )
    {
        const int nIndex =nFeature*nPositions+nPosition;
        const double dblNormalized =
            (static_cast<double>(c_lpfInput[nIndex])-dblMean)*dblInvStd;
        const double dblGamma =
            c_lpfGamma==nullptr ? 1.0 : c_lpfGamma[nFeature];
        const double dblNormalizedGrad =
            static_cast<double>(c_lpfOutputGrad[nIndex])*dblGamma;
        const double dblInputGrad =
            dblInvStd/static_cast<double>(nFeatures)*
            (static_cast<double>(nFeatures)*dblNormalizedGrad-
             dblGradSum-dblNormalized*dblGradNormalizedSum);
        lpfInputGrad[nIndex] +=static_cast<float>(dblInputGrad);
    }
}
__global__ void kernel_LayerNorm_parameter_reduce(
    float* lpfGammaGrad,float* lpfBetaGrad,const float* c_lpfGammaContributions,
    const float* c_lpfBetaContributions,int nFeatures,int nPositions
)
{
    const int nFeature =blockIdx.x*blockDim.x+threadIdx.x;
    if( nFeature>=nFeatures ) return;
    float fGamma =0.0f;
    float fBeta =0.0f;
    for( int nPosition=0;nPosition<nPositions;++nPosition )
    {
        fGamma +=c_lpfGammaContributions[nFeature*nPositions+nPosition];
        fBeta +=c_lpfBetaContributions[nFeature*nPositions+nPosition];
    }
    lpfGammaGrad[nFeature] +=fGamma;
    lpfBetaGrad[nFeature] +=fBeta;
}
__global__ void kernel_Mask_forward(
    float* lpfResult,
    const float* c_lpfInput,
    const float* c_lpfMask,
    int* lpnError,
    int nKey,
    int nTrailing,
    bool isBroadcast,
    int nRows
)
{
    const int nRow =blockIdx.x*blockDim.x+threadIdx.x;
    if( nRow>=nRows ) return;

    const int nQuery =nRow/nTrailing;
    const int nTrailingIndex =nRow%nTrailing;
    bool isAnyAvailable =false;
    for( int nKeyIndex=0;nKeyIndex<nKey;++nKeyIndex )
    {
        const int nInputIndex =
            (nQuery*nKey+nKeyIndex)*nTrailing+nTrailingIndex;
        const int nMaskIndex =isBroadcast ?
            nQuery*nKey+nKeyIndex : nInputIndex;
        const float fMask =c_lpfMask[nMaskIndex];
        if( !isfinite(fMask) )
        {
            atomicOr(lpnError,1);
            return;
        }
        isAnyAvailable |=fMask!=0.0f;
    }
    if( !isAnyAvailable )
    {
        atomicOr(lpnError,2);
        return;
    }

    for( int nKeyIndex=0;nKeyIndex<nKey;++nKeyIndex )
    {
        const int nInputIndex =
            (nQuery*nKey+nKeyIndex)*nTrailing+nTrailingIndex;
        const int nMaskIndex =isBroadcast ?
            nQuery*nKey+nKeyIndex : nInputIndex;
        lpfResult[nInputIndex] =c_lpfMask[nMaskIndex]!=0.0f ?
            c_lpfInput[nInputIndex] : -CUDART_INF_F;
    }
}
__global__ void kernel_Mask_backward(
    float* lpfInputGrad,
    const float* c_lpfOutputGrad,
    const float* c_lpfMask,
    int nTrailing,
    bool isBroadcast,
    int nSize
)
{
    const int nIndex =blockIdx.x*blockDim.x+threadIdx.x;
    if( nIndex>=nSize ) return;
    const int nMaskIndex =isBroadcast ? nIndex/nTrailing : nIndex;
    if( c_lpfMask[nMaskIndex]!=0.0f )
    {
        lpfInputGrad[nIndex] +=c_lpfOutputGrad[nIndex];
    }
}
__global__ void kernel_Permute(
    float* lpfResult,
    const float* c_lpfInput,
    const std::int64_t* c_lpnOutputShape,
    const std::int64_t* c_lpnInputStrides,
    int nRank,
    int nSize
)
{
    const int nIndex =blockIdx.x*blockDim.x+threadIdx.x;
    if( nIndex>=nSize ) return;

    std::int64_t nRemainder =nIndex;
    std::int64_t nInputIndex =0;
    for( int nDimension=nRank-1;nDimension>=0;--nDimension )
    {
        const std::int64_t nCoordinate =
            nRemainder%c_lpnOutputShape[nDimension];
        nRemainder /=c_lpnOutputShape[nDimension];
        nInputIndex +=
            nCoordinate*c_lpnInputStrides[nDimension];
    }
    lpfResult[nIndex] =c_lpfInput[nInputIndex];
}
__global__ void kernel_Softmax_forward(
    float* lpfResult,
    const float* c_lpfInput,
    int* lpnError,
    int nAxisSize,
    int nInner,
    int nSlices
)
{
    const int nSlice =blockIdx.x*blockDim.x+threadIdx.x;
    if( nSlice>=nSlices ) return;

    const int nOuterIndex =nSlice/nInner;
    const int nInnerIndex =nSlice%nInner;
    const int nBase =
        nOuterIndex*nAxisSize*nInner+nInnerIndex;
    float fMaximum =-CUDART_INF_F;
    for( int nAxisIndex=0;nAxisIndex<nAxisSize;++nAxisIndex )
    {
        const float fValue =
            c_lpfInput[nBase+nAxisIndex*nInner];
        if( isnan(fValue)||(isinf(fValue)&&(fValue>0.0f)) )
        {
            atomicOr(lpnError,1);
            return;
        }
        fMaximum =fmaxf(fMaximum,fValue);
    }
    if( isinf(fMaximum) )
    {
        atomicOr(lpnError,2);
        return;
    }

    float fSum =0.0f;
    for( int nAxisIndex=0;nAxisIndex<nAxisSize;++nAxisIndex )
    {
        const int nIndex =nBase+nAxisIndex*nInner;
        const float fValue =expf(c_lpfInput[nIndex]-fMaximum);
        lpfResult[nIndex] =fValue;
        fSum +=fValue;
    }
    for( int nAxisIndex=0;nAxisIndex<nAxisSize;++nAxisIndex )
    {
        const int nIndex =nBase+nAxisIndex*nInner;
        lpfResult[nIndex] /=fSum;
    }
}
__global__ void kernel_Softmax_backward(
    float* lpfInputGrad,
    const float* c_lpfOutputGrad,
    const float* c_lpfOutput,
    int nAxisSize,
    int nInner,
    int nSlices
)
{
    const int nSlice =blockIdx.x*blockDim.x+threadIdx.x;
    if( nSlice>=nSlices ) return;

    const int nOuterIndex =nSlice/nInner;
    const int nInnerIndex =nSlice%nInner;
    const int nBase =
        nOuterIndex*nAxisSize*nInner+nInnerIndex;
    float fDot =0.0f;
    for( int nAxisIndex=0;nAxisIndex<nAxisSize;++nAxisIndex )
    {
        const int nIndex =nBase+nAxisIndex*nInner;
        fDot +=c_lpfOutputGrad[nIndex]*c_lpfOutput[nIndex];
    }
    for( int nAxisIndex=0;nAxisIndex<nAxisSize;++nAxisIndex )
    {
        const int nIndex =nBase+nAxisIndex*nInner;
        lpfInputGrad[nIndex] +=
            c_lpfOutput[nIndex]*(c_lpfOutputGrad[nIndex]-fDot);
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
        float fMaxLogits    =c_lpfLogits[nBatchIndex];
        for( int nClassIndex=1;nClassIndex<nClass;nClassIndex++ )
        {
            if( c_lpfLogits[nClassIndex*nBatch+nBatchIndex]>fMaxLogits )
            {
                fMaxLogits =c_lpfLogits[nClassIndex*nBatch+nBatchIndex];
            }
        }
        // expの合計
        float fSumExp       =0.0f;
        for( int nClassIndex=0;nClassIndex<nClass;nClassIndex++ )
        {
            fSumExp +=expf(c_lpfLogits[nClassIndex*nBatch+nBatchIndex]-fMaxLogits);
        }
        // CrossEntropy
        float fLoss         =0.0;
        float fSumExpLog    =fMaxLogits + logf( fSumExp );
        for( int nClassIndex=0;nClassIndex<nClass;nClassIndex++ )
        {
            int nIndex =nClassIndex*nBatch+nBatchIndex;

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
        float fMaxLogits    =c_lpfLogits[nBatchIndex];
        for( int nClassIndex=0;nClassIndex<nClass;nClassIndex++ )
        {
            if( c_lpfLogits[nClassIndex*nBatch+nBatchIndex]>fMaxLogits )
            {
                fMaxLogits =c_lpfLogits[nClassIndex*nBatch+nBatchIndex];
            }
        }
        // expの合計
        float fSumExp   =0.0f;
        for( int nClassIndex=0;nClassIndex<nClass;nClassIndex++ )
        {
            fSumExp +=expf(c_lpfLogits[nClassIndex*nBatch+nBatchIndex]-fMaxLogits);
        }
        // gradient
        for( int nClassIndex=0;nClassIndex<nClass;nClassIndex++ )
        {
            int nIndex =nClassIndex*nBatch+nBatchIndex;
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
// Row-major [features,batch]なので x_{f,b} は f*B+b。
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
        fMean +=c_lpfValue[nFeature*nBatch+nBatchIndex];
    }
    fMean /=static_cast<float>(nBatch);

    // ミニバッチ分散（母分散）:
    // sigma_f^2 = (1/B) * sum_b (x_{f,b}-mu_f)^2
    float fVariance =0.0f;
    for( int nBatchIndex=0;nBatchIndex<nBatch;++nBatchIndex )
    {
        const float fCentered =c_lpfValue[nFeature*nBatch+nBatchIndex]-fMean;
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
        const int nIndex        =nFeature*nBatch+nBatchIndex;

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
        const int nIndex        =nFeature*nBatch+nBatchIndex;
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
        const int nIndex    =nFeature*nBatch+nBatchIndex;
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
        const int nIndex    =nFeature*nBatch+nBatchIndex;
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
        const int nColumns      = nBatch * nOutputHeight * nOutputWidth;
        const int nPatchIndex   = nIndex / nColumns;
        const int nColumn       = nIndex % nColumns;
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
                ? c_lpfInput[(( nInputY * nInputWidth + nInputX ) * nChannels + nChannel) * nBatch + nSample]
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
    const int nRow      = nIndex / nBatch;
    const int nSample   = nIndex % nBatch;
    const int nChannel  = nRow % nOutputChannels;
    const int nPosition = nRow / nOutputChannels;
    // Y[n,o,c_o] = (W*C)[c_o,n*(H_out*W_out)+o] + b[c_o]
    lpfOutput[nIndex] =
        c_lpfGemm[nChannel * (nBatch*nPositions) + nSample * nPositions + nPosition] +
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
    const int nColumns  = nSize / nOutputChannels;
    const int nChannel  = nIndex / nColumns;
    const int nColumn   = nIndex % nColumns;
    const int nSample   = nColumn / nPositions;
    const int nPosition = nColumn % nPositions;
    // G[c_o,n*(H_out*W_out)+o] = dL/dY[n,o,c_o]
    const int nBatch = nColumns / nPositions;
    lpfGemmGrad[nIndex] = c_lpfOutputGrad[(nPosition*nOutputChannels+nChannel)*nBatch+nSample];
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
    const int nColumns      = nSize / nPatchSize;
    const int nPatchIndex   = nIndex / nColumns;
    const int nColumn       = nIndex % nColumns;
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
        const int nBatch = nColumns / (nOutputHeight*nOutputWidth);
        atomicAdd( lpfInputGrad + (( nInputY * nInputWidth + nInputX ) * nChannels + nChannel)*nBatch+nSample,
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
        const int nColumns =nSize/nChannels;
        atomicAdd( lpfBiasGrad + nIndex/nColumns, c_lpfGemmGrad[nIndex] );
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
    const int nBatch    = nSize/(nChannels*nPositions);
    const int nFeature  = nIndex/nBatch;
    const int nSample   = nIndex%nBatch;
    const int nChannel  = nFeature%nChannels;
    const int nPosition = nFeature/nChannels;
    const int nOutputY  = nPosition / nOutputWidth;
    const int nOutputX  = nPosition % nOutputWidth;
    // Y[n,o_y,o_x,c] = max_(0<=k_y,k_x<K) X[n,o_y*S+k_y,o_x*S+k_x,c]
    float fMaximum = -FLT_MAX;
    for( int nKernelY = 0; nKernelY < nKernelSize; ++nKernelY )
    {
        for( int nKernelX = 0; nKernelX < nKernelSize; ++nKernelX )
        {
            const int nInputFeature =(( nOutputY*nStride+nKernelY)*nInputWidth+
                                      nOutputX*nStride+nKernelX)*nChannels+nChannel;
            const float fValue =c_lpfInput[nInputFeature*nBatch+nSample];
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
    const int nBatch    = nSize/(nChannels*nPositions);
    const int nFeature  = nIndex/nBatch;
    const int nSample   = nIndex%nBatch;
    const int nChannel  = nFeature%nChannels;
    const int nPosition = nFeature/nChannels;
    const int nOutputY  = nPosition / nOutputWidth;
    const int nOutputX  = nPosition % nOutputWidth;
    // a = argmax_(k_y,k_x) X[n,o_y*S+k_y,o_x*S+k_x,c]
    // 比較は > のみなので、同値なら走査順で最初の位置を保持する。
    int nBestRow = ( nOutputY * nStride * nInputWidth + nOutputX * nStride ) * nChannels + nChannel;
    float fMaximum = c_lpfInput[nBestRow*nBatch+nSample];
    for( int nKernelY = 0; nKernelY < nKernelSize; ++nKernelY )
    {
        for( int nKernelX = 0; nKernelX < nKernelSize; ++nKernelX )
        {
            const int nRow = ( ( nOutputY * nStride + nKernelY ) * nInputWidth +
                               nOutputX * nStride + nKernelX ) *
                                 nChannels +
                             nChannel;
            const float fValue =c_lpfInput[nRow*nBatch+nSample];
            if( fValue > fMaximum )
            {
                fMaximum = fValue;
                nBestRow = nRow;
            }
        }
    }
    // dL/dX[n,a,c] += dL/dY[n,o_y,o_x,c]
    // windowが重なる場合は同じ入力へ複数の勾配が流れるためatomicAddする。
    atomicAdd( lpfInputGrad + nBestRow*nBatch+nSample,
               c_lpfOutputGrad[nIndex] );
}

__global__ void kernel_Embedding_forward(
    float* lpfOutput,
    const float* c_lpfWeight,
    const std::int32_t* c_lpnIndices,
    int nPositions,
    int nVocabSize,
    int nSize
)
{
    const int nIndex =blockIdx.x*blockDim.x+threadIdx.x;
    if( nIndex<nSize )
    {
        const int nEmbedding =nIndex/nPositions;
        const int nPosition =nIndex%nPositions;
        lpfOutput[nIndex] =
            c_lpfWeight[nEmbedding*nVocabSize+c_lpnIndices[nPosition]];
    }
}

__global__ void kernel_Embedding_backward(
    float* lpfWeightGrad,
    const float* c_lpfOutputGrad,
    const std::int32_t* c_lpnIndices,
    int nPositions,
    int nVocabSize,
    int nSize
)
{
    const int nEmbedding =blockIdx.x*blockDim.x+threadIdx.x;
    if( nEmbedding<nSize )
    {
        for( int nPosition=0;nPosition<nPositions;++nPosition )
        {
            lpfWeightGrad[nEmbedding*nVocabSize+c_lpnIndices[nPosition]] +=
                c_lpfOutputGrad[nEmbedding*nPositions+nPosition];
        }
    }
}
