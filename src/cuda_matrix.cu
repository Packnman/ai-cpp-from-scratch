#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include "cuda_matrix.h"

__global__ void kernel_fill(float* rst,float value,int size);
__global__ void kernel_mul_elementwise(float* rst,const float* A,const float* B,int size);
__global__ void kernel_ReLU_forward(float* rst,const float* value,int size);
__global__ void kernel_ReLU_backward(float* rst,const float* data,const float* grad,int size);


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
    if (error != cudaSuccess)
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
    if (error != cudaSuccess)
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
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda_ReLU_backward: kernel launch failed"
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