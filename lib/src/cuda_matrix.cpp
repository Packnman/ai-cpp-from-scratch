#include "cuda_matrix.h"
#include "cuda_bublas.h"
#include "matrix.h"
#include <stdexcept>
#include <string>

cuMat::cuMat()
    :_lpfDevice( nullptr ),
     _nRows( 0 ),
     _nCols( 0 )
{
    
}
cuMat::cuMat(int nRows,int nCols)
    :_lpfDevice( nullptr ),
     _nRows( nRows ),
     _nCols( nCols )
{
    // GPU VRAM
    cudaMalloc(
        reinterpret_cast<void**>(&_lpfDevice),
        nRows*nCols*sizeof(float)
    );
}

cuMat::cuMat(const cuMat& c_mValue)
    :_lpfDevice( nullptr ),
     _nRows( c_mValue._nRows ),
     _nCols( c_mValue._nCols )
{
    // GPU VRAM
    cudaMalloc(
        reinterpret_cast<void**>(&_lpfDevice),
        c_mValue._nRows*c_mValue._nCols*sizeof(float)
    );

    // 
    cudaError_t cudError =cudaMemcpy(
        _lpfDevice,
        c_mValue._lpfDevice,
        _nRows*_nCols*sizeof(float),
        cudaMemcpyDeviceToDevice
    );
    if( cudError!=cudaSuccess )
    {
        // メモリ解放
        cudaFree( _lpfDevice ); _lpfDevice  =nullptr;
        // 
        throw std::runtime_error(
            std::string( "cudaMemcpy DeviceToDevice failed: " )
            + cudaGetErrorString( cudError )
        );
    }
}
cuMat::cuMat(cuMat&& mValue) noexcept
    :_lpfDevice( mValue._lpfDevice ),
     _nRows( mValue._nRows ),
     _nCols( mValue._nCols )
{
    mValue._nRows      =0;
    mValue._nCols      =0;
    mValue._lpfDevice  =nullptr;
}
cuMat::~cuMat()
{
    // メモリ解放
    cudaFree( _lpfDevice );
    _lpfDevice  =nullptr;
}
cuMat& cuMat::operator=(const cuMat& c_mValue)
{
    if( this==&c_mValue )    {return *this;}

    // サイズが違うなら再確保
    if( (_nRows!=c_mValue._nRows)||(_nCols!=c_mValue._nCols) )
    {
        cudaFree( _lpfDevice );

        _nRows  =c_mValue._nRows;
        _nCols  =c_mValue._nCols;

        cudaMalloc( reinterpret_cast<void**>(&_lpfDevice),_nRows*_nCols*sizeof(float) );
    }

    // GPU → GPU コピー
    cudaError_t cudError     =cudaMemcpy(
        _lpfDevice,
        c_mValue._lpfDevice,
        _nRows*_nCols*sizeof(float),
        cudaMemcpyDeviceToDevice
    );

    if( cudError!=cudaSuccess )
    {
        throw std::runtime_error(
            std::string("cuMat copy assignment failed: ")
            + cudaGetErrorString(cudError)
        );
    }

    return *this;
}
cuMat& cuMat::operator=(cuMat&& mValue) noexcept
{
    if( this!=&mValue )
    {
        cudaFree( _lpfDevice );

        _nRows      =mValue._nRows;
        _nCols      =mValue._nCols;
        _lpfDevice  =mValue._lpfDevice;

        mValue._nRows      =0;
        mValue._nCols      =0;
        mValue._lpfDevice  =nullptr;
    }

    return *this;
}
void cuMat::upload(Mat& mHost) const
{
    cudaMemcpy(
        mHost._lpfHost,
        _lpfDevice,
        _nRows*_nCols*sizeof(float),
        cudaMemcpyDeviceToHost
    );
}
void cuMat::download(const Mat& c_mHost)
{
    cudaMemcpy(
        _lpfDevice,
        c_mHost._lpfHost,
        _nRows*_nCols*sizeof(float),
        cudaMemcpyHostToDevice
    );
}


void cuda_scale(cuMat& mResult,float fValue)
{
    cublasStatus_t cblStatus   =cublasSscal(
        getCublasHandle(),
        mResult._nRows * mResult._nCols,
        &fValue,
        mResult._lpfDevice,
        1
    );

    if( cblStatus!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::cuda_scale: cublasSscal failed"
        );
    }
}
void cuda_axpy(cuMat& mResult,float fAlpha,const cuMat& c_mA)
{
    // cublasSaxpy
    // Y = αX + Y
    // → 要素ごとの加算・減算向け

    // 行列サイズ確認
    if( (mResult._nRows!=c_mA._nRows)||(mResult._nCols!=c_mA._nCols) )
    {
        throw std::runtime_error(
            "cuMat::cuda_axpy: matrix size mismatch"
        );
    }

    cublasStatus_t cblStatus   =cublasSaxpy(
        getCublasHandle(),
        mResult._nRows*mResult._nCols,
        &fAlpha,
        c_mA._lpfDevice,
        1,
        mResult._lpfDevice,
        1
    );

    if( cblStatus!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::cuda_axpy: cublasSaxpy failed"
        );
    }
}
void cuda_geam(cuMat& mResult,float fAlpha,const cuMat& c_mA,float fBeta,const cuMat& c_mB)
{
    // cublasSgeam
    // R = αA + βB
    
    // 行列サイズ確認
    if( (c_mA._nCols!=c_mB._nCols)||(c_mA._nRows!=c_mB._nRows) )
    {
        throw std::runtime_error(
            "cuMat::cuda_geam: c_mA and c_mB size mismatch"
        );
    }
    if( (mResult._nRows!=c_mA._nRows)||(mResult._nCols!=c_mA._nCols) )
    {
        throw std::runtime_error(
            "cuda_geam: result size mismatch"
        );
    }

    cublasStatus_t cblStatus   =cublasSgeam(
        getCublasHandle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        c_mA._nRows,
        c_mA._nCols,
        &fAlpha,
        c_mA._lpfDevice,
        c_mA._nRows,
        &fBeta,
        c_mB._lpfDevice,
        c_mB._nRows,
        mResult._lpfDevice,
        mResult._nRows
    );

    
    if( cblStatus!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::cuda_geam: cublasSgeam failed"
        );
    }
}

// R = α*op(A)*op(B) + β*op(B)
void cuda_gemm(cuMat& mResult,const cuMat& c_mA,const cuMat& c_mB,
               bool isTransposeA,
               bool isTransposeB,
               float fAlpha,
               float fBeta
)
{
    // cublasSgemm
    // C = αAB + βC
    // → 行列積向け

    cublasOperation_t cblOperationA   =isTransposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    int nRowsA   =isTransposeA ? c_mA._nCols : c_mA._nRows;
    int nColsA   =isTransposeA ? c_mA._nRows : c_mA._nCols;

    cublasOperation_t cblOperationB   =isTransposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int nRowsB   =isTransposeB ? c_mB._nCols : c_mB._nRows;
    int nColsB   =isTransposeB ? c_mB._nRows : c_mB._nCols;
    // 行列サイズ確認
    if( nColsA!=nRowsB )
    {
        throw std::runtime_error(
            "cuMat::cuda_gemm: matrix size mismatch"
        );
    }

    cublasStatus_t cblStatus   =cublasSgemm(
        getCublasHandle(),
        cblOperationA,
        cblOperationB,
        nRowsA,              // m
        nColsB,              // n
        nColsA,              // k
        &fAlpha,
        c_mA._lpfDevice,
        c_mA._nRows,           // lda
        c_mB._lpfDevice,
        c_mB._nRows,           // ldb
        &fBeta,
        mResult._lpfDevice,
        mResult._nRows          // ldc
    );

    if( cblStatus!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::cuda_gemm: cublasSgemm failed"
        );
    }
}
void cuda_transpose(cuMat& mResult,const cuMat& c_mA)
{
    // R = A^T
    // cublasSgeam
    // R = αA + βB
    
    // 行列サイズ確認
    if( (mResult._nCols!=c_mA._nRows)||(mResult._nRows!=c_mA._nCols) )
    {
        throw std::runtime_error(
            "cuMat::cuda_geam: matrix size mismatch"
        );
    }

    float fAlpha    =1.0f;
    float fBeta     =0.0f;

    cublasStatus_t cblStatus   =cublasSgeam(
        getCublasHandle(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        c_mA._nCols,   // 転置後の行数
        c_mA._nRows,   // 転置後の列数
        &fAlpha,
        c_mA._lpfDevice,
        c_mA._nRows,
        &fBeta,
        c_mA._lpfDevice,
        c_mA._nCols,
        mResult._lpfDevice,
        mResult._nRows
    );

    
    if( cblStatus!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::cuda_geam: cublasSgeam failed"
        );
    }
}
