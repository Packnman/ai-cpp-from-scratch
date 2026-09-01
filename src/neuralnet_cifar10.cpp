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
    TensorList spmValues{spmOutput};
    // ReLU
    spmOutput =_rluReLU( spmValues );
    spmValues ={spmOutput};
    // MaxPool2D
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
    TensorList spmValues{spmOutput};
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

Cifar10NeuralNet::Cifar10NeuralNet(
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

std::shared_ptr<Tensor> Cifar10NeuralNet::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    if( (spmInputs.size()!=1)||(spmInputs[0]==nullptr) )
    {
        throw std::runtime_error(
            "Cifar10NeuralNet::forward: exactly one non-null input is required"
        );
    }
    if( (spmInputs[0]->_mData._nRows!=32*32*3)||
        (spmInputs[0]->_mData._nCols<=0) )
    {
        throw std::runtime_error(
            "Cifar10NeuralNet::forward: input must have shape 3072 x batch"
        );
    }

    TensorList spmValues{spmInputs[0]};
    std::shared_ptr<Tensor> spmOutput =_lyrConv1.forward( spmValues );
    spmValues ={spmOutput};
    spmOutput =_lyrConv2.forward( spmValues );
    spmValues ={spmOutput};
    spmOutput =_lyrHidden.forward( spmValues );
    spmValues ={spmOutput};
    return _lyrOutput.forward( spmValues );
}

std::shared_ptr<Tensor> Cifar10NeuralNet::loss(
    const std::shared_ptr<Tensor>& c_spmInput,
    const std::shared_ptr<Tensor>& c_spmTarget
)
{
    if( (c_spmInput==nullptr)||(c_spmTarget==nullptr) )
    {
        throw std::runtime_error(
            "Cifar10NeuralNet::loss: input and target must be non-null"
        );
    }
    if( (c_spmTarget->_mData._nRows!=10)||
        (c_spmTarget->_mData._nCols!=c_spmInput->_mData._nCols) )
    {
        throw std::runtime_error(
            "Cifar10NeuralNet::loss: target must have shape 10 x batch"
        );
    }

    TensorList spmInputs{c_spmInput};
    std::shared_ptr<Tensor> spmLogits =forward( spmInputs );
    TensorList spmArguments{spmLogits,c_spmTarget};
    return _sceEntropy( spmArguments );
}
