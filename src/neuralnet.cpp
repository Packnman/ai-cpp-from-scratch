#include <iostream>
#include <stdio.h>
#include <random>
#include <math.h>

#include "matrix.h"
#include "neuralnet.h"

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
// LayerInput
// --------------------------
// 役割：
//  ・前処理
// --------------------------
LayerInput::LayerInput()
    :Module()
{

}
LayerInput::~LayerInput()
{

}
void LayerInput::init(std::mt19937& rngRandom)
{
    (void)rngRandom;
    // Non
}
std::shared_ptr<Tensor> LayerInput::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    return spmInputs[0];
}
// --------------------------
// LayerHidden
// --------------------------
// 役割：
//  ・1層目からn-1層目までの処理
// --------------------------
LayerHidden::LayerHidden(int nInput,int nOutput)
    :Module(),
     _spmWeight( std::make_shared<Tensor>(nOutput,nInput) ),
     _spmBias( std::make_shared<Tensor>(nOutput,1) ),
     _lnrLinear( _spmWeight.get(),_spmBias.get() )
{
    registerParameter( "weight",_spmWeight.get() );
    registerParameter( "bias",_spmBias.get() );
}
LayerHidden::~LayerHidden()
{

}
void LayerHidden::init(std::mt19937& rngRandom)
{
    init_Weight( *_spmWeight,_spmWeight->_mData._nCols,rngRandom );
    cuda_fill( _spmBias.get()->_mData,0 );
}
std::shared_ptr<Tensor> LayerHidden::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    // Linear
    std::shared_ptr<Tensor> spmOutput =_lnrLinear( spmInputs );
    // ReLU
    std::vector<std::shared_ptr<Tensor>> spmOutputs{spmOutput};
    spmOutput =_rluReLU( spmOutputs );

    return spmOutput;
}
// --------------------------
// LayerOutput
// --------------------------
// 役割：
//  ・n層目の処理
// --------------------------
LayerOutput::LayerOutput(int nInput,int nOutput)
    :Module(),
     _spmWeight( std::make_shared<Tensor>(nOutput,nInput) ),
     _spmBias( std::make_shared<Tensor>(nOutput,1) ),
     _lnrLinear( _spmWeight.get(),_spmBias.get() )
{
    registerParameter( "weight",_spmWeight.get() );
    registerParameter( "bias",_spmBias.get() );
}
LayerOutput::~LayerOutput()
{

}
void LayerOutput::init(std::mt19937& rngRandom)
{
    init_Weight( *_spmWeight,_spmWeight->_mData._nCols,rngRandom );
    cuda_fill( _spmBias.get()->_mData,0 );
}
std::shared_ptr<Tensor> LayerOutput::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    return _lnrLinear( spmInputs );
}
// --------------------------
// NeuralNet
// --------------------------
// 役割：
//  ・
// --------------------------
NeuralNet::NeuralNet(std::uint32_t nSeed)
    :Model(),
     _lyrInput(),
     _lyrHidden1(784,256),
     _lyrHidden2(256,128),
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
NeuralNet::~NeuralNet()
{

}
std::shared_ptr<Tensor> NeuralNet::forward(
    std::vector<std::shared_ptr<Tensor>>& spmInputs
)
{
    if( (spmInputs.size()!=1)||(spmInputs[0]==nullptr) )
    {
        throw std::runtime_error(
            "NeuralNet::forward: exactly one non-null input is required"
        );
    }
    if( (spmInputs[0]->_mData._nRows!=784)||
        (spmInputs[0]->_mData._nCols<=0) )
    {
        throw std::runtime_error(
            "NeuralNet::forward: input must have shape 784 x batch"
        );
    }
    //
    std::vector<std::shared_ptr<Tensor>> spmOutputs{spmInputs[0]};
    std::shared_ptr<Tensor> spmOutput =_lyrInput.forward( spmOutputs );
    spmOutputs ={spmOutput};
    spmOutput =_lyrHidden1.forward( spmOutputs );
    spmOutputs ={spmOutput};
    spmOutput =_lyrHidden2.forward( spmOutputs );
    spmOutputs ={spmOutput};
    //
    return _lyrOutput.forward( spmOutputs );
}
std::shared_ptr<Tensor> NeuralNet::loss(
    const std::shared_ptr<Tensor>& c_spmInput,
    const std::shared_ptr<Tensor>& c_spmTarget
)
{
    if( (c_spmInput==nullptr)||(c_spmTarget==nullptr) )
    {
        throw std::runtime_error(
            "NeuralNet::loss: input and target must be non-null"
        );
    }
    if( (c_spmTarget->_mData._nRows!=10)||
        (c_spmTarget->_mData._nCols!=c_spmInput->_mData._nCols) )
    {
        throw std::runtime_error(
            "NeuralNet::loss: target must have shape 10 x batch"
        );
    }
    //
    std::vector<std::shared_ptr<Tensor>> spmInputs{c_spmInput};
    std::shared_ptr<Tensor> spmLogits =forward( spmInputs );
    std::vector<std::shared_ptr<Tensor>> spmArgs{spmLogits,c_spmTarget};

    return _sceEntropy( spmArgs );
}
