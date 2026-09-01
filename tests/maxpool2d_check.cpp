#include "cuda_function.h"
#include "cuda_tensor.h"
#include <cuda_runtime_api.h>
#include "matrix.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
namespace
{
void req( bool isCondition, const char* c_lpszMessage )
{
    if( !isCondition )
    {
        throw std::runtime_error( c_lpszMessage );
    }
}
void fill( cuMat& mValue, const std::vector<float>& c_fValues )
{
    Mat mHost( mValue._nRows, mValue._nCols );
    for( size_t nIndex = 0; nIndex < c_fValues.size(); ++nIndex )
    {
        mHost._lpfHost[nIndex] = c_fValues[nIndex];
    }
    mValue.download( mHost );
}
Mat host( const cuMat& c_mValue )
{
    Mat mHost( c_mValue._nRows, c_mValue._nCols );
    c_mValue.upload( mHost );
    return mHost;
}
void pooling()
{
    auto spmInput = std::make_shared<Tensor>( 9, 1 );
    fill( spmInput->_mData, { 5, 5, 1, 5, 0, 4, 2, 3, 6 } );
    cuda_fill( spmInput->_mGrad, .5f );
    MaxPool2D mplPool( 1, 3, 3, 2, 1 );
    auto spmOutput = mplPool( { spmInput } );
    Mat mOutputHost = host( spmOutput->_mData );
    float fExpectedOutput[]{ 5, 5, 5, 6 };
    for( int nIndex = 0; nIndex < 4; ++nIndex )
    {
        req( mOutputHost._lpfHost[nIndex] == fExpectedOutput[nIndex], "forward" );
    }
    cuMat mOutputGrad( 4, 1 );
    fill( mOutputGrad, { 1, 2, 3, 4 } );
    mplPool.backward( { &mOutputGrad }, { spmInput }, { spmOutput } );
    Mat mInputGrad = host( spmInput->_mGrad );
    float fExpectedInputGrad[]{ 1, 2, 0, 3, 0, 0, 0, 0, 4 };
    for( int nIndex = 0; nIndex < 9; ++nIndex )
    {
        req( std::fabs( mInputGrad._lpfHost[nIndex] - fExpectedInputGrad[nIndex] - .5f ) < 1e-6f,
             "backward/tie" );
    }
}
void channels()
{
    auto spmInput = std::make_shared<Tensor>( 8, 2 );
    fill( spmInput->_mData, { 1, 8, 3, 2, 4, 7, 6, 5, 9, 1, 2, 10, 3, 4, 8, 7 } );
    MaxPool2D mplPool( 2, 2, 2 );
    auto spmOutput = mplPool( { spmInput } );
    Mat mOutputHost = host( spmOutput->_mData );
    req( mOutputHost( 0, 0 ) == 6 && mOutputHost( 1, 0 ) == 8 && mOutputHost( 0, 1 ) == 9 &&
             mOutputHost( 1, 1 ) == 10,
         "channels/batch" );
    bool isArgumentRejected = false;
    bool isShapeRejected = false;
    try
    {
        MaxPool2D mplInvalid( 1, 1, 1, 2 );
    }
    catch( const std::invalid_argument& )
    {
        isArgumentRejected = true;
    }
    try
    {
        MaxPool2D mplValid( 1, 2, 2 );
        mplValid( { std::make_shared<Tensor>( 3, 1 ) } );
    }
    catch( const std::runtime_error& )
    {
        isShapeRejected = true;
    }
    req( isArgumentRejected && isShapeRejected, "validation" );
}
} // namespace
void cifarBatchBackward();
int main()
{
    try
    {
        pooling();
        channels();
        cifarBatchBackward();
        std::cout << "maxpool2d check passed\n";
        return 0;
    }
    catch( const std::exception& c_errError )
    {
        std::cerr << c_errError.what() << '\n';
        return 1;
    }
}
void cifarBatchBackward()
{
    constexpr int CHANNELS =32;
    constexpr int HEIGHT =32;
    constexpr int WIDTH =32;
    constexpr int BATCH =128;
    auto spmInput =std::make_shared<Tensor>(
        CHANNELS*HEIGHT*WIDTH,
        BATCH
    );
    cuda_fill( spmInput->_mData,1.0f );
    MaxPool2D mplPool( CHANNELS,HEIGHT,WIDTH );
    auto spmOutput =mplPool( {spmInput} );
    req(
        (spmOutput->_mData._nRows==CHANNELS*16*16)&&
        (spmOutput->_mData._nCols==BATCH),
        "CIFAR batch output shape"
    );
    cuMat mOutputGrad( CHANNELS*16*16,BATCH );
    cuda_fill( mOutputGrad,1.0f );
    mplPool.backward( {&mOutputGrad},{spmInput},{spmOutput} );
    const cudaError_t cudError =cudaDeviceSynchronize();
    req( cudError==cudaSuccess,"CIFAR batch backward CUDA error" );
}
