#include "cuda_matrix.h"
#include "cuda_bublas.h"
#include "matrix.h"
#include <stdexcept>
#include <string>

cuMat::cuMat()
{
    this->_nRows        =0;
    this->_nCols        =0;
    this->_lpfDevice    =nullptr;
}
cuMat::cuMat(int nRows,int nCols)
{
    this->_nRows        =nRows;
    this->_nCols        =nCols;
    // GPU VRAM
    cudaMalloc(
        reinterpret_cast<void**>(&this->_lpfDevice),
        nRows*nCols*sizeof(float)
    );
}

cuMat::cuMat(const cuMat& val)
{
    this->_nRows        =val._nRows;
    this->_nCols        =val._nCols;
    // GPU VRAM
    cudaMalloc(
        reinterpret_cast<void**>(&this->_lpfDevice),
        val._nRows*val._nCols*sizeof(float)
    );

    // 
    cudaError_t err =cudaMemcpy(
        _lpfDevice,
        val._lpfDevice,
        _nRows*_nCols*sizeof(float),
        cudaMemcpyDeviceToDevice
    );
    if( err!=cudaSuccess )
    {
        // メモリ解放
        cudaFree( _lpfDevice ); _lpfDevice  =nullptr;
        // 
        throw std::runtime_error(
            std::string( "cudaMemcpy DeviceToDevice failed: " )
            + cudaGetErrorString( err )
        );
    }
}

cuMat::~cuMat()
{
    // メモリ解放
    cudaFree( _lpfDevice );
    _lpfDevice  =nullptr;
}
cuMat& cuMat::operator=(const cuMat& val)
{
    if( this==&val )    {return *this;}

    // サイズが違うなら再確保
    if( (_nRows!=val._nRows)||(_nCols!=val._nCols) )
    {
        cudaFree( _lpfDevice );

        _nRows  =val._nRows;
        _nCols  =val._nCols;

        cudaMalloc( reinterpret_cast<void**>(&_lpfDevice),_nRows*_nCols*sizeof(float) );
    }

    // GPU → GPU コピー
    cudaError_t err     =cudaMemcpy(
        _lpfDevice,
        val._lpfDevice,
        _nRows*_nCols*sizeof(float),
        cudaMemcpyDeviceToDevice
    );

    if( err!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuMat copy assignment failed: ")
            + cudaGetErrorString(err)
        );
    }

    return *this;
}
cuMat& cuMat::operator=(cuMat&& val) noexcept
{
    if( this!=&val )
    {
        cudaFree( _lpfDevice );

        _nRows      =val._nRows;
        _nCols      =val._nCols;
        _lpfDevice  =val._lpfDevice;

        val._nRows      =0;
        val._nCols      =0;
        val._lpfDevice  =nullptr;
    }

    return *this;
}
void cuMat::upload(Mat& host)
{
    cudaMemcpy(
        host._lpfHost,
        _lpfDevice,
        _nRows*_nCols*sizeof(float),
        cudaMemcpyDeviceToHost
    );
}
void cuMat::download(const Mat& host)
{
    cudaMemcpy(
        _lpfDevice,
        host._lpfHost,
        _nRows*_nCols*sizeof(float),
        cudaMemcpyHostToDevice
    );
}


void cuda_scale(cuMat& rst,float value)
{
    cublasStatus_t status   =cublasSscal(
        getCublasHandle(),
        rst._nRows * rst._nCols,
        &value,
        rst._lpfDevice,
        1
    );

    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::cuda_scale: cublasSscal failed"
        );
    }
}
void cuda_axpy(cuMat& rst,float alpha,const cuMat& A)
{
    // cublasSaxpy
    // Y = αX + Y
    // → 要素ごとの加算・減算向け

    // 行列サイズ確認
    if( (rst._nRows!=A._nRows)||(rst._nCols!=A._nCols) )
    {
        throw std::runtime_error(
            "cuMat::cuda_axpy: matrix size mismatch"
        );
    }

    cublasStatus_t status   =cublasSaxpy(
        getCublasHandle(),
        rst._nRows*rst._nCols,
        &alpha,
        A._lpfDevice,
        1,
        rst._lpfDevice,
        1
    );

    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::cuda_axpy: cublasSaxpy failed"
        );
    }
}
void cuda_geam(cuMat& rst,float alpha,const cuMat& A,float beta,const cuMat& B)
{
    // cublasSgeam
    // R = αA + βB
    
    // 行列サイズ確認
    if( (A._nCols!=B._nCols)||(A._nRows!=B._nRows) )
    {
        throw std::runtime_error(
            "cuMat::cuda_geam: A and B size mismatch"
        );
    }
    if( (rst._nRows!=A._nRows)||(rst._nCols!=A._nCols) )
    {
        throw std::runtime_error(
            "cuda_geam: result size mismatch"
        );
    }

    cublasStatus_t status   =cublasSgeam(
        getCublasHandle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        A._nRows,
        A._nCols,
        &alpha,
        A._lpfDevice,
        A._nRows,
        &beta,
        B._lpfDevice,
        B._nRows,
        rst._lpfDevice,
        rst._nRows
    );

    
    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::cuda_geam: cublasSgeam failed"
        );
    }
}

// R = α*op(A)*op(B) + β*op(B)
void cuda_gemm(cuMat& rst,const cuMat& A,const cuMat& B,
               bool trpA=false,
               bool trpB=false,
               float alpha=1.0f,
               float beta=0.0f
)
{
    // cublasSgemm
    // C = αAB + βC
    // → 行列積向け

    cublasOperation_t opA   =trpA ? CUBLAS_OP_T : CUBLAS_OP_N;
    int aRows   =trpA ? A._nCols : A._nRows;
    int aCols   =trpA ? A._nRows : A._nCols;

    cublasOperation_t opB   =trpB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int bRows   =trpB ? B._nCols : B._nRows;
    int bCols   =trpB ? B._nRows : B._nCols;
    // 行列サイズ確認
    if( aCols!=bRows )
    {
        throw std::runtime_error(
            "cuMat::cuda_gemm: matrix size mismatch"
        );
    }

    cublasStatus_t status   =cublasSgemm(
        getCublasHandle(),
        opA,
        opB,
        aRows,              // m
        bCols,              // n
        aCols,              // k
        &alpha,
        A._lpfDevice,
        A._nRows,           // lda
        B._lpfDevice,
        B._nRows,           // ldb
        &beta,
        rst._lpfDevice,
        rst._nRows          // ldc
    );

    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::cuda_gemm: cublasSgemm failed"
        );
    }
}
void cuda_transpose(cuMat& rst,const cuMat& A)
{
    // R = A^T
    // cublasSgeam
    // R = αA + βB
    
    // 行列サイズ確認
    if( (rst._nCols!=A._nRows)||(rst._nRows!=A._nCols) )
    {
        throw std::runtime_error(
            "cuMat::cuda_geam: matrix size mismatch"
        );
    }

    float fAlpha    =1.0f;
    float fBeta     =0.0f;

    cublasStatus_t status   =cublasSgeam(
        getCublasHandle(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        A._nCols,   // 転置後の行数
        A._nRows,   // 転置後の列数
        &fAlpha,
        A._lpfDevice,
        A._nRows,
        &fBeta,
        A._lpfDevice,
        A._nCols,
        rst._lpfDevice,
        rst._nRows
    );

    
    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::cuda_geam: cublasSgeam failed"
        );
    }
}