#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include "cuda_matrix.h"

__global__ void kernel_fill(float* rst,float value,int size);
__global__ void kernel_mul_elementwise(float* rst,const float* A,const float* B,int size);
__global__ void kernel_ReLU_forward(float* rst,const float* value,int size);
__global__ void kernel_ReLU_backward(float* rst,const float* data,const float* grad,int size);
__global__ void kernel_GELU_forward(float* rst,const float* value,int size);
__global__ void kernel_GELU_backward(float* rst,const float* data,const float* grad,int size);
__global__ void kernel_SoftmaxCrossEntropy_forward(float* rst,const float* logits,const float* target,int nClass,int nBatch);
__global__ void kernel_SoftmaxCrossEntropy_backward(float* rst,const float* logits,const float* target,const float* grad,int nClass,int nBatch);


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
    cudaError_t err =cudaGetLastError();
    if( err!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "kernel_fill failed: " )
            + cudaGetErrorString( err )
        );
    }
}
void cuda_fill(cuMat& rst,float value)
{
    // R[:] = value

    int nSize       =rst._nRows * rst._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )    {return;}
    //
    kernel_fill<<<nBlocks,nThreads>>>(
        rst._lpfDevice,
        value,nSize
    );
    // 
    cudaError_t err =cudaGetLastError();
    if( err!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "kernel_fill failed: " )
            + cudaGetErrorString( err )
        );
    }
}
void cuda_mul_elementwise(cuMat& rst,const cuMat& A,const cuMat& B)
{
    // R = A ⦿ B
    
    // 行列数の確認
    if( (A._nRows!=B._nRows)||(A._nCols!=B._nCols) )
    {
        throw std::runtime_error(
            "cuda_mul_elementwise: A and B size mismatch"
        );
    }
    if( (rst._nRows!=A._nRows)||(rst._nCols!=A._nCols) )
    {
        throw std::runtime_error(
            "cuda_mul_elementwise: result size mismatch"
        );
    }

    int nSize    =A._nRows * A._nCols;
    int nThreads =256;
    int nBlocks  =(nSize + nThreads - 1) / nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_mul_elementwise<<<nBlocks, nThreads>>>(
        rst._lpfDevice,
        A._lpfDevice,
        B._lpfDevice,
        nSize
    );

    cudaError_t error = cudaGetLastError();
    if( error!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_mul_elementwise: kernel launch failed"
        );
    }
}
void cuda_ReLU_forward(cuMat& rst,const cuMat& value)
{
    if( (rst._nRows!=value._nRows)||(rst._nCols!=value._nCols) )
    {
        throw std::runtime_error(
            "cuda_ReLU_forward: matrix size mismatch"
        );
    }
    //
    int nSize       =value._nRows * value._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_ReLU_forward<<<nBlocks,nThreads>>>(
        rst._lpfDevice,
        value._lpfDevice,
        nSize
    );

    cudaError_t error = cudaGetLastError();
    if( error!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_ReLU_forward: kernel launch failed"
        );
    }
}
void cuda_ReLU_backward(cuMat& rst,const cuMat& data,const cuMat& grad)
{
    if( (rst._nRows!=data._nRows)||(rst._nCols!=data._nCols)||
        (data._nRows!=grad._nRows)||(data._nCols!=grad._nCols) )
    {
        throw std::runtime_error(
            "cuda_ReLU_backward: matrix size mismatch"
        );
    }
    //
    int nSize       =data._nRows * data._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_ReLU_backward<<<nBlocks,nThreads>>>(
        rst._lpfDevice,
        data._lpfDevice,
        grad._lpfDevice,
        nSize
    );
    
    cudaError_t error = cudaGetLastError();
    if( error!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_ReLU_backward: kernel launch failed"
        );
    }
}
void cuda_GELU_forward(cuMat& rst,const cuMat& value)
{
    if( (rst._nRows!=value._nRows)||(rst._nCols!=value._nCols) )
    {
        throw std::runtime_error(
            "cuda_ReLU_forward: matrix size mismatch"
        );
    }
    //
    int nSize       =value._nRows * value._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_GELU_forward<<<nBlocks,nThreads>>>(
        rst._lpfDevice,
        value._lpfDevice,
        nSize
    );
    //
    cudaError_t error = cudaGetLastError();
    if( error!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_ReLU_forward: kernel launch failed"
        );
    }
}
void cuda_GELU_backward(cuMat& rst,const cuMat& data,const cuMat& grad)
{
    if( (rst._nRows!=data._nRows)||(rst._nCols!=data._nCols)||
        (data._nRows!=grad._nRows)||(data._nCols!=grad._nCols) )
    {
        throw std::runtime_error(
            "cuda_GELU_backward: matrix size mismatch"
        );
    }
    //
    int nSize       =data._nRows * data._nCols;
    int nThreads    =256;
    int nBlocks     =(nSize + nThreads - 1)/nThreads;
    if( nSize<=0 )   {return;}
    //
    kernel_GELU_backward<<<nBlocks,nThreads>>>(
        rst._lpfDevice,
        data._lpfDevice,
        grad._lpfDevice,
        nSize
    );
    //
    cudaError_t error = cudaGetLastError();
    if( error!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_GELU_backward: kernel launch failed"
        );
    }
}
void cuda_SoftmaxCrossEntropy_forward(
    cuMat& rst,
    const cuMat& logits,
    const cuMat& target
)
{
    if( (logits._nRows!=target._nRows)||(logits._nCols!=target._nCols) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_forward: matrix size mismatch"
        );
    }
    if( (rst._nRows!=1)||(rst._nCols!=1) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_forward: result must be 1x1"
        );
    }
    //
    int nClass      =logits._nRows;
    int nBatch      =logits._nCols;
    int nThreads    =256;
    int nBlocks     =(nBatch + nThreads - 1)/nThreads;
    if( (nClass<=0)||(nBatch<=0) )  {return;}
    //
    cuda_fill( rst,0.0f );      // atomicAddするので最初は0
    kernel_SoftmaxCrossEntropy_forward<<<nBlocks,nThreads>>>(
        rst._lpfDevice,
        logits._lpfDevice,
        target._lpfDevice,
        nClass,
        nBatch
    );
    //
    cudaError_t error = cudaGetLastError();
    if( error!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_forward: kernel launch failed"
        );
    }
}
void cuda_SoftmaxCrossEntropy_backward(
    cuMat& logitsGrad,
    const cuMat& logits,
    const cuMat& target,
    const cuMat& grad
)
{
    if( (logits._nRows!=target._nRows)||(logits._nCols!=target._nCols) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_backward: matrix size mismatch"
        );
    }
    if( (logits._nRows!=logitsGrad._nRows)||(logits._nCols!=logitsGrad._nCols) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_backward: gradient size mismatch"
        );
    }
    if( (grad._nRows!=1)||(grad._nCols!=1) )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_backward: grad must be 1x1"
        );
    }
    //
    int nClass      =logits._nRows;
    int nBatch      =logits._nCols;
    int nThreads    =256;
    int nBlocks     =(nBatch + nThreads - 1)/nThreads;
    if( (nClass<=0)||(nBatch<=0) )  {return;}
    //
    kernel_SoftmaxCrossEntropy_backward<<<nBlocks,nThreads>>>(
        logitsGrad._lpfDevice,
        logits._lpfDevice,
        target._lpfDevice,
        grad._lpfDevice,
        nClass,
        nBatch
    );
    //
    cudaError_t error = cudaGetLastError();
    if( error!=cudaSuccess )
    {
        throw std::runtime_error(
            "cuda_SoftmaxCrossEntropy_backward: kernel launch failed"
        );
    }
}

__global__ void kernel_fill(float* rst,float value,int size)
{
    int i   =blockIdx.x * blockDim.x + threadIdx.x;

    if( i<size )
    {
        rst[i]    =value;
    }
}

__global__ void kernel_mul_elementwise(float* rst,const float* A,const float* B,int size)
{
    int i   =blockIdx.x * blockDim.x + threadIdx.x;

    if( i<size )
    {
        rst[i]    =A[i] * B[i];
    }
}
__global__ void kernel_ReLU_forward(float* rst,const float* value,int size)
{
    int i   =blockIdx.x * blockDim.x + threadIdx.x;

    if( i<size )
    {
        rst[i]  =value[i]>0.0f ? value[i] : 0.0f;
    }
}
__global__ void kernel_ReLU_backward(float* rst,const float* data,const float* grad,int size)
{
    int i   =blockIdx.x * blockDim.x + threadIdx.x;

    if( i<size )
    {
        if( data[i]>0.0f )
        {
            rst[i]  +=grad[i];
        }
    }
}
__global__ void kernel_GELU_forward(float* rst,const float* value,int size)
{
    int i   =blockIdx.x * blockDim.x + threadIdx.x;

    if( i<size )
    {
        const float v   =value[i];
        const float c   =0.7978845608f;
        const float a   =0.044715f;
        const float u   =c*(v + a*v*v*v);
        const float t   =tanhf( u );
        
        const float derivative  =0.5f*(1.0f + t);

        rst[i]  =value[i]*derivative;
    }
}
__global__ void kernel_GELU_backward(float* rst,const float* data,const float* grad,int size)
{
    int i   =blockIdx.x * blockDim.x + threadIdx.x;

    if( i<size )
    {
        const float v   =data[i];
        const float c   =0.7978845608f;
        const float a   =0.044715f;
        const float u   =c*(v + a*v*v*v);
        const float t   =tanhf( u );

        const float derivative  =0.5f*(1.0f + t) + 0.5f*v*(1.0f - t*t)*c*(1.0f + 3.0f*a*v*v);

        rst[i]  +=grad[i]*derivative;
    }
}
__global__ void kernel_SoftmaxCrossEntropy_forward(float* rst,const float* logits,const float* target,int nClass,int nBatch)
{
    int b   =blockIdx.x * blockDim.x + threadIdx.x;

    if( b<nBatch )
    {
        // max(logits)を求める
        float fMaxLogits    =logits[b*nClass];
        for( int i=1;i<nClass;i++ )
        {
            if( logits[b*nClass+i]>fMaxLogits )
            {
                fMaxLogits    =logits[b*nClass+i];
            }
        }
        // expの合計
        float fSumExp       =0.0f;
        for( int i=0;i<nClass;i++ )
        {
            fSumExp +=expf( logits[b*nClass+i]-fMaxLogits );
        }
        // CrossEntropy
        float fLoss         =0.0;
        float fSumExpLog    =fMaxLogits + logf( fSumExp );
        for( int i=0;i<nClass;i++ )
        {
            int idx =b*nClass+i;

            fLoss   -=target[idx] * (logits[idx] - fSumExpLog);
        }
        // batch平均
        atomicAdd(
            rst,
            fLoss/static_cast<float>(nBatch)
        );
    }
}
__global__ void kernel_SoftmaxCrossEntropy_backward(float* rst,const float* logits,const float* target,const float* grad,int nClass,int nBatch)
{
    int b   =blockIdx.x * blockDim.x + threadIdx.x;

    if( b<nBatch )
    {
        // max(logits)
        float fMaxLogits    =logits[b*nClass];
        for( int i=0;i<nClass;i++ )
        {
            if( logits[b*nClass+i]>fMaxLogits )
            {
                fMaxLogits  =logits[b*nClass+i];
            }
        }
        // expの合計
        float fSumExp   =0.0f;
        for( int i=0;i<nClass;i++ )
        {
            fSumExp +=expf( logits[b*nClass+i]-fMaxLogits );
        }
        // gradient
        for( int i=0;i<nClass;i++ )
        {
            int idx         =b*nClass+i;
            float fSoftmax  =expf( logits[idx]-fMaxLogits )/fSumExp;

            rst[idx]    +=grad[0]*(fSoftmax-target[idx])/static_cast<float>(nBatch);
        }
    }
}