#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include "cuMat.h"

__global__ void cuMat_pls(float* C,const float* A,const float* B,int size);
__global__ void cuMat_sub(float* C,const float* A,const float* B,int size);

cuMat operator+(const cuMat& L,const cuMat& R)
{
    // 行列サイズ確認
    if( (L._nRows!=R._nRows)||(L._nCols!=R._nCols) )
    {
        throw std::runtime_error(
            "cuMat::operator+: matrix size mismatch"
        );
    }

    cuMat rst( L._nRows,L._nCols );
    // ブロック使用数の計算
    const int c_nSize    =L._nRows*L._nCols;
    const int c_nThreads =256;
    const int c_nBlocks  =(c_nSize+c_nThreads-1)/c_nThreads;
    // GPUへの演算命令
    cuMat_pls<<<c_nBlocks,c_nThreads>>>( rst._lpfDevice,L._lpfDevice,R._lpfDevice,c_nSize );

    cudaError_t err =cudaGetLastError();
    if( err!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "cuMat_pls failed: " )
            + cudaGetErrorString( err )
        );
    }

    return rst;
}
cuMat operator-(const cuMat& L,const cuMat& R)
{
    // 行列サイズ確認
    if( (L._nRows!=R._nRows)||(L._nCols!=R._nCols) )
    {
        throw std::runtime_error(
            "cuMat::operator-: matrix size mismatch"
        );
    }

    cuMat rst( L._nRows,L._nCols );
    // ブロック使用数の計算
    const int c_nSize    =L._nRows*L._nCols;
    const int c_nThreads =256;
    const int c_nBlocks  =(c_nSize+c_nThreads-1)/c_nThreads;
    // GPUへの演算命令
    cuMat_sub<<<c_nBlocks,c_nThreads>>>( rst._lpfDevice,L._lpfDevice,R._lpfDevice,c_nSize );

    cudaError_t err =cudaGetLastError();
    if( err!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string( "cuMat_sub failed: " )
            + cudaGetErrorString( err )
        );
    }

    return rst;
}

__global__ void cuMat_pls(float* rst,const float* L,const float* R,int size)
{
    int i   =blockIdx.x * blockDim.x + threadIdx.x;

    if( i<size )
    {
        rst[i]    =L[i] + R[i];
    }
}
__global__ void cuMat_sub(float* rst,const float* L,const float* R,int size)
{
    int i   =blockIdx.x * blockDim.x + threadIdx.x;

    if( i<size )
    {
        rst[i]    =L[i] - R[i];
    }
}