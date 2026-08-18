#include "cuMat.h"
#include "cuBublas.h"
#include <stdexcept>
#include <string>

cuMat::cuMat(int nRows,int nCols)
    :_nRows( nRows ),
     _nCols( nCols ),
     _lpfDevice( nullptr ),
     _lpfHost( nullptr )
{
    // CPU RAM
    _lpfHost    =(float*)malloc( _nRows*_nCols*sizeof(float) );
    // GPU VRAM
    cudaMalloc( reinterpret_cast<void**>(&_lpfDevice),_nRows*_nCols*sizeof(float) );
}

cuMat::cuMat(const cuMat& val)
    :_nRows( val._nRows ),
     _nCols( val._nCols ),
     _lpfDevice( nullptr ),
     _lpfHost( nullptr )
{
    // CPU RAM
    _lpfHost    =(float*)malloc( _nRows*_nCols*sizeof(float) );
    // GPU VRAM
    cudaMalloc( reinterpret_cast<void**>(&_lpfDevice),_nRows*_nCols*sizeof(float) );

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
        free( _lpfHost );       _lpfHost    =nullptr;
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
    cudaFree( _lpfDevice ); _lpfDevice  =nullptr;
    free( _lpfHost );       _lpfHost    =nullptr;
}
cuMat& cuMat::operator=(const cuMat& val)
{
    if( this==&val )    {return *this;}

    // サイズが違うなら再確保
    if( (_nRows!=val._nRows)||(_nCols!=val._nCols) )
    {
        cudaFree( _lpfDevice );
        free( _lpfHost );

        _nRows  =val._nRows;
        _nCols  =val._nCols;

        _lpfHost    =(float*)::malloc( _nRows*_nCols*sizeof(float) );
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

    // Host側も必要ならコピー
    std::memcpy( _lpfHost,val._lpfHost,_nRows*_nCols*sizeof(float) );

    return *this;
}
cuMat& cuMat::operator=(cuMat&& val) noexcept
{
    if( this!=&val )
    {
        cudaFree( _lpfDevice );
        free( _lpfHost );

        _nRows      =val._nRows;
        _nCols      =val._nCols;
        _lpfDevice  =val._lpfDevice;
        _lpfHost    =val._lpfHost;

        val._nRows      =0;
        val._nCols      =0;
        val._lpfDevice  =nullptr;
        val._lpfHost    =nullptr;
    }

    return *this;
}
cuMat operator*(const cuMat& L,const cuMat& R)
{
    // cublasSgemm
    // C = αAB + βC
    // → 行列積向け

    // 行列サイズ確認
    if( L._nCols!=R._nRows )
    {
        throw std::runtime_error(
            "cuMat::operator*: matrix size mismatch"
        );
    }

    cuMat rst( L._nRows,R._nCols );

    const float alpha   =1.0f;
    const float beta    =0.0f;

    cublasStatus_t status   =cublasSgemm(
        getCublasHandle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        L._nRows,
        R._nCols,
        L._nCols,
        &alpha,
        L._lpfDevice,
        L._nRows,
        R._lpfDevice,
        R._nRows,
        &beta,
        rst._lpfDevice,
        rst._nRows
    );

    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::operator*: cublasSgemm failed"
        );
    }

    return rst;
}
cuMat& cuMat::operator+=(const cuMat& val)
{
    // cublasSaxpy
    // Y = αX + Y
    // → 要素ごとの加算・減算向け

    // 行列サイズ確認
    if( (_nRows!=val._nRows)||(_nCols!=val._nCols) )
    {
        throw std::runtime_error(
            "cuMat::operator+=: matrix size mismatch"
        );
    }

    const float alpha   =1.0f;

    cublasStatus_t status   =cublasSaxpy(
        getCublasHandle(),
        _nRows*_nCols,
        &alpha,
        val._lpfDevice,
        1,
        _lpfDevice,
        1
    );

    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::operator+=: cublasSaxpy failed"
        );
    }

    return *this;
}
cuMat& cuMat::operator-=(const cuMat& val)
{
    // cublasSaxpy
    // Y = αX + Y
    // → 要素ごとの加算・減算向け
    
    // 行列サイズ確認
    if( (_nRows!=val._nRows)||(_nCols!=val._nCols) )
    {
        throw std::runtime_error(
            "cuMat::operator-=: matrix size mismatch"
        );
    }

    const float alpha   =-1.0f;

    cublasStatus_t status   =cublasSaxpy(
        getCublasHandle(),
        _nRows*_nCols,
        &alpha,
        val._lpfDevice,
        1,
        _lpfDevice,
        1
    );

    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::operator-=: cublasSaxpy failed"
        );
    }

    return *this;
}
cuMat& cuMat::operator*=(const cuMat& val)
{
    // cublasSgemm
    // C = αAB + βC
    // → 行列積向け
    if( _nCols!=val._nRows )
    {
        throw std::runtime_error(
            "cuMat::operator*=: matrix size mismatch"
        );
    }

    cuMat result( _nRows,val._nCols );

    const float alpha   =1.0f;
    const float beta    =0.0f;

    cublasStatus_t status   =cublasSgemm(
        getCublasHandle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        _nRows,
        val._nCols,
        _nCols,
        &alpha,
        _lpfDevice,
        _nRows,
        val._lpfDevice,
        val._nRows,
        &beta,
        result._lpfDevice,
        result._nRows
    );

    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::operator*=: cublasSgemm failed"
        );
    }

    *this = std::move(result);

    return *this;
}
cuMat& cuMat::operator*=(float val)
{
    cublasStatus_t status   =cublasSscal(
        getCublasHandle(),
        _nRows * _nCols,
        &val,
        _lpfDevice,
        1
    );

    if( status!=CUBLAS_STATUS_SUCCESS )
    {
        throw std::runtime_error(
            "cuMat::operator*=: cublasSscal failed"
        );
    }

    return *this;
}
float& cuMat::operator()(int row,int col)
{
    return _lpfHost[IDX2F(row,col,_nRows)];
}
const float& cuMat::operator()(int row,int col) const
{
    return _lpfHost[IDX2F(row,col,_nRows)];
}
void cuMat::upload()
{
    cudaMemcpy(
        _lpfDevice,
        _lpfHost,
        _nRows*_nCols*sizeof(float),
        cudaMemcpyHostToDevice
    );
}
void cuMat::download()
{
    cudaMemcpy(
        _lpfHost,
        _lpfDevice,
        _nRows*_nCols*sizeof(float),
        cudaMemcpyDeviceToHost
    );
}