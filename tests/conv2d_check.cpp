#include "cuda_function.h"
#include "cuda_tensor.h"
#include "matrix.h"
#include "module.h"
#include "optimizer.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
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
void near( float fActual, float fExpected, const char* c_lpszMessage )
{
    if( std::fabs( fActual - fExpected ) > 2e-4f )
    {
        throw std::runtime_error( c_lpszMessage );
    }
}
void fill( cuMat& mValue, const std::vector<float>& c_fValues )
{
    req( c_fValues.size() == static_cast<size_t>( mValue._nRows * mValue._nCols ), "fill" );
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
void basic()
{
    Tensor mWeight( 1, 4 ), mBias( 1, 1 );
    fill( mWeight._mData, { 1, 2, 3, 4 } );
    fill( mBias._mData, { .5f } );
    auto spmInput = std::make_shared<Tensor>( 9, 1 );
    fill( spmInput->_mData, { 1, 2, 3, 4, 5, 6, 7, 8, 9 } );
    cuda_fill( spmInput->_mGrad, .25f );
    cuda_fill( mWeight._mGrad, 1 );
    cuda_fill( mBias._mGrad, 2 );
    Conv2D cnvConv( &mWeight, &mBias, 1, 3, 3, 2 );
    auto spmOutput = cnvConv( { spmInput } );
    Mat mOutputHost = host( spmOutput->_mData );
    float fExpectedOutput[]{ 37.5f, 47.5f, 67.5f, 77.5f };
    for( int nIndex = 0; nIndex < 4; ++nIndex )
    {
        near( mOutputHost._lpfHost[nIndex], fExpectedOutput[nIndex], "forward" );
    }
    cuMat mOutputGrad( 4, 1 );
    fill( mOutputGrad, { 1, 1, 1, 1 } );
    cnvConv.backward( { &mOutputGrad }, { spmInput }, { spmOutput } );
    Mat mInputGrad = host( spmInput->_mGrad ), mWeightGrad = host( mWeight._mGrad ),
        mBiasGrad = host( mBias._mGrad );
    float fExpectedInputGrad[]{ 1, 3, 2, 4, 10, 6, 3, 7, 4 },
        fExpectedWeightGrad[]{ 12, 16, 24, 28 };
    for( int nIndex = 0; nIndex < 9; ++nIndex )
    {
        near( mInputGrad._lpfHost[nIndex], fExpectedInputGrad[nIndex] + .25f, "dx add" );
    }
    for( int nIndex = 0; nIndex < 4; ++nIndex )
    {
        near( mWeightGrad._lpfHost[nIndex], fExpectedWeightGrad[nIndex] + 1, "mWeightGrad add" );
    }
    near( mBiasGrad( 0, 0 ), 6, "mBiasGrad add" );
}
void channels()
{
    Tensor mWeight( 2, 8 ), mBias( 2, 1 );
    fill( mWeight._mData, { 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 } );
    fill( mBias._mData, { 1, -1 } );
    auto spmInput = std::make_shared<Tensor>( 18, 2 );
    std::vector<float> fInputValues( 36 );
    for( int nBatch = 0; nBatch < 2; ++nBatch )
    {
        for( int nRow = 0; nRow < 18; ++nRow )
        {
            fInputValues[nBatch * 18 + nRow] = nRow + 1 + nBatch;
        }
    }
    fill( spmInput->_mData, fInputValues );
    Conv2D cnvConv( &mWeight, &mBias, 2, 3, 3, 2, 2, 1 );
    auto spmOutput = cnvConv( { spmInput } );
    req( spmOutput->_mData._nRows == 8 && spmOutput->_mData._nCols == 2, "shape" );
    Mat mOutputHost = host( spmOutput->_mData );
    for( int nBatch = 0; nBatch < 2; ++nBatch )
    {
        for( int nOutputY = 0; nOutputY < 2; ++nOutputY )
        {
            for( int nOutputX = 0; nOutputX < 2; ++nOutputX )
            {
                for( int nOutputChannel = 0; nOutputChannel < 2; ++nOutputChannel )
                {
                    float fExpected = nOutputChannel ? -1.f : 1.f;
                    for( int nKernelY = 0; nKernelY < 2; ++nKernelY )
                    {
                        for( int nKernelX = 0; nKernelX < 2; ++nKernelX )
                        {
                            int nInputY = nOutputY * 2 + nKernelY - 1,
                                nInputX = nOutputX * 2 + nKernelX - 1;
                            if( nInputY >= 0 && nInputY < 3 && nInputX >= 0 && nInputX < 3 )
                            {
                                fExpected +=
                                    fInputValues[nBatch * 18 + ( nInputY * 3 + nInputX ) * 2 +
                                                 nOutputChannel];
                            }
                        }
                    }
                    near( mOutputHost( ( nOutputY * 2 + nOutputX ) * 2 + nOutputChannel, nBatch ),
                          fExpected, "channels" );
                }
            }
        }
    }
}
void oneByOne()
{
    Tensor mWeight( 2, 2 ), mBias( 2, 1 );
    fill( mWeight._mData, { 1, -1, 2, 1 } );
    fill( mBias._mData, { 0, 0 } );
    auto spmInput = std::make_shared<Tensor>( 4, 1 );
    fill( spmInput->_mData, { 1, 3, 2, 4 } );
    Conv2D cnvConv( &mWeight, &mBias, 2, 1, 2, 1 );
    auto spmOutput = cnvConv( { spmInput } );
    Mat mOutputHost = host( spmOutput->_mData );
    float fExpectedOutput[]{ 7, 2, 10, 2 };
    for( int nIndex = 0; nIndex < 4; ++nIndex )
    {
        near( mOutputHost._lpfHost[nIndex], fExpectedOutput[nIndex], "1x1 kernel" );
    }
}
void validation()
{
    Tensor mWeight( 1, 1 ), mBias( 1, 1 );
    bool isNullRejected = false;
    bool isWeightShapeRejected = false;
    bool isInputShapeRejected = false;
    try
    {
        Conv2D cnvConv( nullptr, &mBias, 1, 1, 1, 1 );
    }
    catch( const std::invalid_argument& )
    {
        isNullRejected = true;
    }
    try
    {
        Conv2D cnvConv( &mWeight, &mBias, 2, 1, 1, 1 );
    }
    catch( const std::invalid_argument& )
    {
        isWeightShapeRejected = true;
    }
    try
    {
        Conv2D cnvConv( &mWeight, &mBias, 1, 1, 1, 1 );
        cnvConv( { std::make_shared<Tensor>( 2, 1 ) } );
    }
    catch( const std::runtime_error& )
    {
        isInputShapeRejected = true;
    }
    req( isNullRejected && isWeightShapeRejected && isInputShapeRejected, "validation" );
}
class ConvModel : public Model
{
  public:
    LayerConv2D _lyrConv;
    ConvModel() : _lyrConv( 2, 3, 4, 5, 3, 2, 1 )
    {
        registerModule( "conv", &_lyrConv );
    }
    std::shared_ptr<Tensor> forward( std::vector<std::shared_ptr<Tensor>>& spmInputs ) override
    {
        return _lyrConv.forward( spmInputs );
    }
};
void layer()
{
    ConvModel mdlModel;
    std::mt19937 rngRandom( 7 );
    mdlModel._lyrConv.init( rngRandom );
    auto nmtNamedParams = mdlModel.namedParameters();
    req( nmtNamedParams.size() == 2 && nmtNamedParams[0].strName == "conv.weight" &&
             nmtNamedParams[1].strName == "conv.bias",
         "parameter names" );
    req( nmtNamedParams[0].lpTensor->_mData._nRows == 3 &&
             nmtNamedParams[0].lpTensor->_mData._nCols == 18,
         "weight shape" );
    req( nmtNamedParams[1].lpTensor->_mData._nRows == 3 &&
             nmtNamedParams[1].lpTensor->_mData._nCols == 1,
         "bias shape" );
    req( mdlModel._lyrConv.outputChannels() == 3 && mdlModel._lyrConv.outputHeight() == 2 &&
             mdlModel._lyrConv.outputWidth() == 3,
         "accessors" );
    Mat mWeightHost = host( nmtNamedParams[0].lpTensor->_mData ),
        mBiasHost = host( nmtNamedParams[1].lpTensor->_mData );
    bool isNonZero = false;
    for( int nIndex = 0; nIndex < 54; ++nIndex )
    {
        isNonZero |= mWeightHost._lpfHost[nIndex] != 0;
    }
    req( isNonZero && mBiasHost( 0, 0 ) == 0 && mBiasHost( 1, 0 ) == 0 && mBiasHost( 2, 0 ) == 0,
         "initialization" );
    auto spmInput = std::make_shared<Tensor>( 40, 1 );
    cuda_fill( spmInput->_mData, 1 );
    std::vector<std::shared_ptr<Tensor>> spmInputs{ spmInput };
    auto spmOutput = mdlModel.forward( spmInputs );
    req( spmOutput->_mData._nRows == 18 && spmOutput->_mData._nCols == 1, "layer output shape" );
    cuda_fill( nmtNamedParams[0].lpTensor->_mGrad, 1 );
    cuda_fill( nmtNamedParams[1].lpTensor->_mGrad, 1 );
    float fBefore = mWeightHost( 0, 0 );
    SGD optOptimizer( &mdlModel, .1f );
    optOptimizer.init();
    optOptimizer.update();
    Mat mAfter = host( nmtNamedParams[0].lpTensor->_mData );
    near( mAfter( 0, 0 ), fBefore - .1f, "optimizer update" );
}
} // namespace
int main()
{
    try
    {
        basic();
        channels();
        oneByOne();
        validation();
        layer();
        std::cout << "conv2d check passed\n";
        return 0;
    }
    catch( const std::exception& c_errError )
    {
        std::cerr << c_errError.what() << '\n';
        return 1;
    }
}
