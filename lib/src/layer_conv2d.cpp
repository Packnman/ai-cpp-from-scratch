#include "module.h"

#include <climits>
#include <cmath>
#include <stdexcept>

#include "matrix.h"
    
int LayerConv2D::_checkedOutputChannels(int nChannels) const
{
    if( nChannels<=0 )
    {
        throw std::invalid_argument(
            "LayerConv2D: output channels must be positive"
        );
    }
    return nChannels;
}
int LayerConv2D::_fanIn(int nChannels,int nKernelSize) const
{
    if( (nChannels<=0)||(nKernelSize<=0) )
    {
        throw std::invalid_argument(
            "LayerConv2D: channels and kernel must be positive"
        );
    }
    const long long nValue  =static_cast<long long>(nChannels) * nKernelSize * nKernelSize;
    if( nValue > INT_MAX )
    {
        throw std::overflow_error( "LayerConv2D: fan-in exceeds int range" );
    }

    return static_cast<int>( nValue );
}
int LayerConv2D::_checkedWeightRows(int nOutputChannels,int nInputChannels,int nKernelSize) const
{
    const int nRows =_checkedOutputChannels( nOutputChannels );
    if( static_cast<long long>(nRows)*_fanIn(nInputChannels,nKernelSize)>INT_MAX )
    {
        throw std::overflow_error( "LayerConv2D: weight exceeds int element range" );
    }

    return nRows;
}
int LayerConv2D::_outputSize(int nInputSize,int nKernelSize,int nStride,int nPadding) const
{
    if( (nInputSize<=0)||(nKernelSize<=0)||(nStride<=0)||(nPadding<0) )
    {
        throw std::invalid_argument( "LayerConv2D: invalid image, kernel, stride, or padding" );
    }
    const long long nPaddedSize     =static_cast<long long>( nInputSize ) + 2LL * nPadding;
    if( nPaddedSize<nKernelSize )
    {
        throw std::invalid_argument( "LayerConv2D: kernel exceeds padded input" );
    }
    const long long nResult     =(nPaddedSize - nKernelSize) / nStride + 1;
    if( nResult > INT_MAX )
    {
        throw std::overflow_error( "LayerConv2D: output exceeds int range" );
    }

    return static_cast<int>( nResult );
}

LayerConv2D::LayerConv2D(int nInputChannels, int nOutputChannels, int nInputHeight,
                         int nInputWidth, int nKernelSize, int nStride, int nPadding)
    :_spmWeight(
        std::make_shared<Tensor>(_checkedWeightRows(nOutputChannels,nInputChannels,nKernelSize),
        _fanIn(nInputChannels,nKernelSize)) ),
     _spmBias( std::make_shared<Tensor>(_checkedOutputChannels(nOutputChannels),1) ),
     _cnvConv2D(
        _spmWeight.get(),
        _spmBias.get(),
        nInputChannels,
        nInputHeight,
        nInputWidth,
        nKernelSize,
        nStride,
        nPadding ),
     _nOutputChannels( nOutputChannels ),
     _nOutputHeight( _outputSize(nInputHeight,nKernelSize,nStride,nPadding) ),
     _nOutputWidth( _outputSize(nInputWidth,nKernelSize,nStride,nPadding) ),
     _nFanIn( _fanIn(nInputChannels,nKernelSize) )
{
    registerParameter( "weight",_spmWeight.get() );
    registerParameter( "bias",_spmBias.get() );
}
LayerConv2D::~LayerConv2D() = default;

void LayerConv2D::init(std::mt19937& rngRandom)
{
    Mat mHost( _spmWeight->_mData._nRows, _spmWeight->_mData._nCols );
    // fan_in = K^2 * C_in
    // W_ij ~ N(0, 2 / fan_in)  (He initialization)
    std::normal_distribution<float> dstNormal(
        0.0f,
        std::sqrt(2.0f/static_cast<float>(_nFanIn))
    );
    const int nCount    =mHost._nRows * mHost._nCols;
    for( int nIndex=0;nIndex<nCount;++nIndex )
    {
        mHost._lpfHost[nIndex]  =dstNormal( rngRandom );
    }
    _spmWeight->_mData.download( mHost );
    cuda_fill( _spmBias->_mData, 0.0f );
}
int LayerConv2D::outputChannels() const
{
    return _nOutputChannels;
}
int LayerConv2D::outputHeight() const
{
    return _nOutputHeight;
}
int LayerConv2D::outputWidth() const
{
    return _nOutputWidth;
}
std::shared_ptr<Tensor> LayerConv2D::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    return _cnvConv2D( spmInputs );
}
