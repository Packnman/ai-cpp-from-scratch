#include "neuralnet_cifar10.h"

#include <cmath>
#include <stdexcept>

#include "matrix.h"

namespace
{
void initWeight(Tensor& mTensor,int nFanIn,std::mt19937& rngRandom)
{
    Mat mHost( mTensor._mData._nRows,mTensor._mData._nCols );
    std::normal_distribution<float> dstNormal(
        0.0f,
        std::sqrt(2.0f/static_cast<float>(nFanIn))
    );
    for( int nIndex=0;
         nIndex<mHost._nRows*mHost._nCols;
         ++nIndex )
    {
        mHost._lpfHost[nIndex] =dstNormal( rngRandom );
    }
    mTensor._mData.download( mHost );
}
}

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

LayerConv2D::LayerConv2D(
    int nInputChannels,
    int nOutputChannels,
    int nInputHeight,
    int nInputWidth,
    int nKernelSize,
    int nStride,
    int nPadding
)
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

Cifar10ConvBlock::Cifar10ConvBlock(
    int nInputChannels,
    int nOutputChannels,
    int nInputHeight,
    int nInputWidth
)
    :_lyrConv(
        nInputChannels,
        nOutputChannels,
        nInputHeight,
        nInputWidth,
        3,
        1,
        1
    ),
     _mplPool(
        nOutputChannels,
        _lyrConv.outputHeight(),
        _lyrConv.outputWidth(),
        2,
        2
    ),
     _nOutputChannels( nOutputChannels ),
     _nOutputHeight( _lyrConv.outputHeight()/2 ),
     _nOutputWidth( _lyrConv.outputWidth()/2 )
{
    registerModule( "conv",&_lyrConv );
}

void Cifar10ConvBlock::init(std::mt19937& rngRandom)
{
    _lyrConv.init( rngRandom );
}

int Cifar10ConvBlock::outputChannels() const
{
    return _nOutputChannels;
}

int Cifar10ConvBlock::outputHeight() const
{
    return _nOutputHeight;
}

int Cifar10ConvBlock::outputWidth() const
{
    return _nOutputWidth;
}

std::shared_ptr<Tensor> Cifar10ConvBlock::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    // Conv2D
    std::shared_ptr<Tensor> spmOutput =_lyrConv.forward( spmInputs );
    std::vector<std::shared_ptr<Tensor>> spmValues{spmOutput};
    // ReLU
    spmOutput =_rluReLU( spmValues );
    spmValues ={spmOutput};
    // Pooling
    return _mplPool( spmValues );
}

Cifar10DenseLayer::Cifar10DenseLayer(
    int nInput,
    int nOutput,
    float fDropoutRate,
    std::uint64_t nDropoutSeed
)
    :_spmWeight( std::make_shared<Tensor>(nOutput,nInput) ),
     _spmBias( std::make_shared<Tensor>(nOutput,1) ),
     _spmGamma( std::make_shared<Tensor>(nOutput,1) ),
     _spmBeta( std::make_shared<Tensor>(nOutput,1) ),
     _spmRunningMean( std::make_shared<Tensor>(nOutput,1) ),
     _spmRunningVar( std::make_shared<Tensor>(nOutput,1) ),
     _lnrLinear( _spmWeight.get(),_spmBias.get() ),
     _bnmBatchNorm(
        _spmGamma.get(),
        _spmBeta.get(),
        _spmRunningMean.get(),
        _spmRunningVar.get()
     ),
     _drpDropout( fDropoutRate,nDropoutSeed )
{
    registerParameter( "weight",_spmWeight.get() );
    registerParameter( "bias",_spmBias.get() );
    registerParameter( "gamma",_spmGamma.get() );
    registerParameter( "beta",_spmBeta.get() );
    registerBuffer( "running_mean",_spmRunningMean.get() );
    registerBuffer( "running_var",_spmRunningVar.get() );
}

void Cifar10DenseLayer::init(std::mt19937& rngRandom)
{
    initWeight( *_spmWeight,_spmWeight->_mData._nCols,rngRandom );
    cuda_fill( _spmBias->_mData,0.0f );
    cuda_fill( _spmGamma->_mData,1.0f );
    cuda_fill( _spmBeta->_mData,0.0f );
    cuda_fill( _spmRunningMean->_mData,0.0f );
    cuda_fill( _spmRunningVar->_mData,1.0f );
}

std::shared_ptr<Tensor> Cifar10DenseLayer::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    // Linear
    std::shared_ptr<Tensor> spmOutput =_lnrLinear( spmInputs );
    // BatchNorm
    _bnmBatchNorm.setTraining( isTraining() );
    std::vector<std::shared_ptr<Tensor>> spmValues{spmOutput};
    spmOutput =_bnmBatchNorm( spmValues );
    spmValues ={spmOutput};
    // ReLU
    spmOutput =_rluReLU( spmValues );
    // Dropout
    if( isTraining() )
    {
        spmValues ={spmOutput};
        spmOutput =_drpDropout( spmValues );
    }
    //
    return spmOutput;
}

Cifar10OutputLayer::Cifar10OutputLayer(int nInput,int nOutput)
    :_spmWeight( std::make_shared<Tensor>(nOutput,nInput) ),
     _spmBias( std::make_shared<Tensor>(nOutput,1) ),
     _lnrLinear( _spmWeight.get(),_spmBias.get() )
{
    registerParameter( "weight",_spmWeight.get() );
    registerParameter( "bias",_spmBias.get() );
}

void Cifar10OutputLayer::init(std::mt19937& rngRandom)
{
    initWeight( *_spmWeight,_spmWeight->_mData._nCols,rngRandom );
    cuda_fill( _spmBias->_mData,0.0f );
}

std::shared_ptr<Tensor> Cifar10OutputLayer::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    // Linear
    return _lnrLinear( spmInputs );
}

NeuralNet_Cifar10::NeuralNet_Cifar10(
    std::uint32_t nSeed,
    float fDropoutRate
)
    :_lyrConv1( 3,32,32,32 ),
     _lyrConv2( 32,64,16,16 ),
     _lyrHidden(
        64*8*8,
        256,
        fDropoutRate,
        static_cast<std::uint64_t>(nSeed)+1
     ),
     _lyrOutput( 256,10 )
{
    registerModule( "conv1",&_lyrConv1 );
    registerModule( "conv2",&_lyrConv2 );
    registerModule( "hidden",&_lyrHidden );
    registerModule( "output",&_lyrOutput );

    std::mt19937 rngRandom( nSeed );
    _lyrConv1.init( rngRandom );
    _lyrConv2.init( rngRandom );
    _lyrHidden.init( rngRandom );
    _lyrOutput.init( rngRandom );
}

std::shared_ptr<Tensor> NeuralNet_Cifar10::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    if( (spmInputs.size()!=1)||(spmInputs[0]==nullptr) )
    {
        throw std::runtime_error(
            "NeuralNet_Cifar10::forward: exactly one non-null input is required"
        );
    }
    if( (spmInputs[0]->_mData._nRows!=32*32*3)||
        (spmInputs[0]->_mData._nCols<=0) )
    {
        throw std::runtime_error(
            "NeuralNet_Cifar10::forward: input must have shape 3072 x batch"
        );
    }

    std::vector<std::shared_ptr<Tensor>> spmValues{spmInputs[0]};
    std::shared_ptr<Tensor> spmOutput =_lyrConv1.forward( spmValues );
    spmValues ={spmOutput};
    spmOutput =_lyrConv2.forward( spmValues );
    spmValues ={spmOutput};
    spmOutput =_lyrHidden.forward( spmValues );
    spmValues ={spmOutput};
    return _lyrOutput.forward( spmValues );
}

std::shared_ptr<Tensor> NeuralNet_Cifar10::loss(
    const std::shared_ptr<Tensor>& c_spmInput,
    const std::shared_ptr<Tensor>& c_spmTarget
)
{
    if( (c_spmInput==nullptr)||(c_spmTarget==nullptr) )
    {
        throw std::runtime_error(
            "NeuralNet_Cifar10::loss: input and target must be non-null"
        );
    }
    if( (c_spmTarget->_mData._nRows!=10)||
        (c_spmTarget->_mData._nCols!=c_spmInput->_mData._nCols) )
    {
        throw std::runtime_error(
            "NeuralNet_Cifar10::loss: target must have shape 10 x batch"
        );
    }

    std::vector<std::shared_ptr<Tensor>> spmInputs{c_spmInput};
    std::shared_ptr<Tensor> spmLogits =forward( spmInputs );
    std::vector<std::shared_ptr<Tensor>> spmArguments{spmLogits,c_spmTarget};
    return _sceEntropy( spmArguments );
}
