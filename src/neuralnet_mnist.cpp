#include <iostream>
#include <stdio.h>
#include <random>
#include <math.h>

#include "matrix.h"
#include "neuralnet_mnist.h"

namespace {
    void init_Weight(Tensor& mTensor,int nFan,std::mt19937& rngRandom)
    {
        Mat mHost( mTensor._mData._nRows,mTensor._mData._nCols );
        std::normal_distribution<float> dstNormal(
            0,
            sqrtf(2.0f/nFan)
        );
        for( int nIndex=0;
             nIndex<mTensor._mData._nRows*mTensor._mData._nCols;
             ++nIndex )
        {
            mHost._lpfHost[nIndex] =dstNormal( rngRandom );
        }
        mTensor._mData.download( mHost );
    }
}

// --------------------------
// MnistInputLayer
// --------------------------
// 役割：
//  ・前処理
// --------------------------
MnistInputLayer::MnistInputLayer()
    :Module()
{

}
MnistInputLayer::~MnistInputLayer()
{

}
void MnistInputLayer::init(std::mt19937& rngRandom)
{
    (void)rngRandom;
    // Non
}
std::shared_ptr<Tensor> MnistInputLayer::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    return spmInputs[0];
}
// --------------------------
// MnistHiddenLayer
// --------------------------
// 役割：
//  ・1層目からn-1層目までの処理
// --------------------------
MnistHiddenLayer::MnistHiddenLayer(int nInput,int nOutput,float fDropoutRate,std::uint64_t nDropoutSeed)
    :Module(),
     _spmWeight( std::make_shared<Tensor>(nOutput,nInput) ),
     _spmBias( std::make_shared<Tensor>(nOutput,1) ),
     _spmGamma( std::make_shared<Tensor>(nOutput,1) ),
     _spmBeta( std::make_shared<Tensor>(nOutput,1) ),
     _spmRunningMean( std::make_shared<Tensor>(nOutput,1) ),
     _spmRunningVar( std::make_shared<Tensor>(nOutput,1) ),
     _lnrLinear( _spmWeight.get(),_spmBias.get() ),
     _bnmBatchNorm( _spmGamma.get(),_spmBeta.get(),_spmRunningMean.get(),_spmRunningVar.get() ),
     _drpDropout( fDropoutRate,nDropoutSeed )
{
    registerParameter( "weight",_spmWeight.get() );
    registerParameter( "bias",_spmBias.get() );
    registerParameter( "gamma",_spmGamma.get() );
    registerParameter( "beta",_spmBeta.get() );
    registerBuffer( "running_mean",_spmRunningMean.get() );
    registerBuffer( "running_var",_spmRunningVar.get() );
}
MnistHiddenLayer::~MnistHiddenLayer()
{

}
void MnistHiddenLayer::init(std::mt19937& rngRandom)
{
    init_Weight( *_spmWeight,_spmWeight->_mData._nCols,rngRandom );
    cuda_fill( _spmBias.get()->_mData,0 );
    cuda_fill( _spmGamma.get()->_mData,1.0f );
    cuda_fill( _spmBeta.get()->_mData,0.0f );
    cuda_fill( _spmRunningMean.get()->_mData,0.0f );
    cuda_fill( _spmRunningVar.get()->_mData,1.0f );
}
std::shared_ptr<Tensor> MnistHiddenLayer::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    // Linear
    std::shared_ptr<Tensor> spmOutput =_lnrLinear( spmInputs );
    // BatchNorm
    _bnmBatchNorm.setTraining( isTraining() );
    std::vector<std::shared_ptr<Tensor>> spmOutputs{spmOutput};
    spmOutput =_bnmBatchNorm( spmOutputs );
    // ReLU
    spmOutputs ={spmOutput};
    spmOutput =_rluReLU( spmOutputs );
    // Dropout
    if( isTraining() )
    {
        spmOutputs ={spmOutput};
        spmOutput =_drpDropout( spmOutputs );
    }

    return spmOutput;
}
// --------------------------
// MnistOutputLayer
// --------------------------
// 役割：
//  ・n層目の処理
// --------------------------
MnistOutputLayer::MnistOutputLayer(int nInput,int nOutput)
    :Module(),
     _spmWeight( std::make_shared<Tensor>(nOutput,nInput) ),
     _spmBias( std::make_shared<Tensor>(nOutput,1) ),
     _lnrLinear( _spmWeight.get(),_spmBias.get() )
{
    registerParameter( "weight",_spmWeight.get() );
    registerParameter( "bias",_spmBias.get() );
}
MnistOutputLayer::~MnistOutputLayer()
{

}
void MnistOutputLayer::init(std::mt19937& rngRandom)
{
    init_Weight( *_spmWeight,_spmWeight->_mData._nCols,rngRandom );
    cuda_fill( _spmBias.get()->_mData,0 );
}
std::shared_ptr<Tensor> MnistOutputLayer::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    // Linear
    return _lnrLinear( spmInputs );
}
// --------------------------
// MnistNeuralNet
// --------------------------
// 役割：
//  ・
// --------------------------
MnistNeuralNet::MnistNeuralNet(std::uint32_t nSeed,float fDropoutRate)
    :Model(),
     _lyrInput(),
     _lyrHidden1(784,256,fDropoutRate,static_cast<std::uint64_t>(nSeed)+1),
     _lyrHidden2(256,128,fDropoutRate,static_cast<std::uint64_t>(nSeed)+2),
     _lyrOutput(128,10)
{
    registerModule( "input",&_lyrInput );
    registerModule( "hidden1",&_lyrHidden1 );
    registerModule( "hidden2",&_lyrHidden2 );
    registerModule( "output",&_lyrOutput );

    std::mt19937 rngRandom( nSeed );
    _lyrInput.init( rngRandom );
    _lyrHidden1.init( rngRandom );
    _lyrHidden2.init( rngRandom );
    _lyrOutput.init( rngRandom );
}
MnistNeuralNet::~MnistNeuralNet()
{

}
std::shared_ptr<Tensor> MnistNeuralNet::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    if( (spmInputs.size()!=1)||(spmInputs[0]==nullptr) )
    {
        throw std::runtime_error(
            "MnistNeuralNet::forward: exactly one non-null input is required"
        );
    }
    if( (spmInputs[0]->_mData._nRows!=784)||
        (spmInputs[0]->_mData._nCols<=0) )
    {
        throw std::runtime_error(
            "MnistNeuralNet::forward: input must have shape 784 x batch"
        );
    }
    //
    std::vector<std::shared_ptr<Tensor>> spmOutputs{spmInputs[0]};

    // Layer Inpout
    std::shared_ptr<Tensor> spmOutput 
                =_lyrInput.forward( spmOutputs );
    spmOutputs  ={spmOutput};
    // Hidden Layer 1
    spmOutput   =_lyrHidden1.forward( spmOutputs );
    spmOutputs  ={spmOutput};
    // Hidden Layer 2
    spmOutput   =_lyrHidden2.forward( spmOutputs );
    spmOutputs  ={spmOutput};
    // Output Layer
    return _lyrOutput.forward( spmOutputs );
}
std::shared_ptr<Tensor> MnistNeuralNet::loss(
    const std::shared_ptr<Tensor>& c_spmInput,
    const std::shared_ptr<Tensor>& c_spmTarget
)
{
    if( (c_spmInput==nullptr)||(c_spmTarget==nullptr) )
    {
        throw std::runtime_error(
            "MnistNeuralNet::loss: input and target must be non-null"
        );
    }
    if( (c_spmTarget->_mData._nRows!=10)||
        (c_spmTarget->_mData._nCols!=c_spmInput->_mData._nCols) )
    {
        throw std::runtime_error(
            "MnistNeuralNet::loss: target must have shape 10 x batch"
        );
    }
    //
    std::vector<std::shared_ptr<Tensor>> spmInputs{c_spmInput};
    std::shared_ptr<Tensor> spmLogits =forward( spmInputs );
    std::vector<std::shared_ptr<Tensor>> spmArgs{spmLogits,c_spmTarget};

    return _sceEntropy( spmArgs );
}
