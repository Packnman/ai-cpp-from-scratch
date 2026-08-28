#include <iostream>
#include <stdio.h>
#include <random>
#include <math.h>

#include "matrix.h"
#include "neuralnet.h"

namespace {
    void init_Weight(Tensor& t,int fan,std::mt19937& rng)
    {
        Mat m( t._mData._nRows,t._mData._nCols );
        std::normal_distribution<float>     dist( 0,sqrtf(2.0f/fan) );
        for( int i=0;i<t._mData._nRows*t._mData._nCols;++i )
        {
            m._lpfHost[i]   =dist( rng );
        }
        t._mData.download( m );
    }
}

// --------------------------
// LayerInput
// --------------------------
// 役割：
//  ・前処理
// --------------------------
LayerInput::LayerInput()
    :Graph()
{

}
LayerInput::~LayerInput()
{

}
void LayerInput::init(std::mt19937& random)
{
    (void)random;
    // Non
}
std::shared_ptr<Tensor> LayerInput::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs
)
{
    return inputs[0];
}
std::vector<Tensor*> LayerInput::getParams()
{
    return {};
}


// --------------------------
// LayerHidden
// --------------------------
// 役割：
//  ・1層目からn-1層目までの処理
// --------------------------
LayerHidden::LayerHidden(int nInput,int nOutput)
    :Graph(),
     _spmWeight( std::make_shared<Tensor>(nOutput,nInput) ),
     _spmBias( std::make_shared<Tensor>(nOutput,1) ),
     _linear( _spmWeight.get(),_spmBias.get() )
{
    
}
LayerHidden::~LayerHidden()
{

}
void LayerHidden::init(std::mt19937& random)
{
    init_Weight( *_spmWeight,_spmWeight->_mData._nCols,random );
    cuda_fill( _spmBias.get()->_mData,0 );
}
std::shared_ptr<Tensor> LayerHidden::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs
)
{
    // Linear
    std::shared_ptr<Tensor> x   =_linear( inputs );
    // ReLU
    std::vector<std::shared_ptr<Tensor>> v{x};
    x   =_relu( v );

    return x;
}
std::vector<Tensor*> LayerHidden::getParams()
{
    return {
        _spmWeight.get(),
        _spmBias.get()
    };
}

// --------------------------
// LayerOutput
// --------------------------
// 役割：
//  ・n層目の処理
// --------------------------
LayerOutput::LayerOutput(int nInput,int nOutput)
    :Graph(),
     _spmWeight( std::make_shared<Tensor>(nOutput,nInput) ),
     _spmBias( std::make_shared<Tensor>(nOutput,1) ),
     _linear( _spmWeight.get(),_spmBias.get() )
{

}
LayerOutput::~LayerOutput()
{

}
void LayerOutput::init(std::mt19937& random)
{
    init_Weight( *_spmWeight,_spmWeight->_mData._nCols,random );
    cuda_fill( _spmBias.get()->_mData,0 );
}
std::shared_ptr<Tensor> LayerOutput::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs
)
{
    return _linear( inputs );
}
std::vector<Tensor*> LayerOutput::getParams()
{
    return {
        _spmWeight.get(),
        _spmBias.get()
    };
}

// --------------------------
// NeuralNet
// --------------------------
// 役割：
//  ・
// --------------------------
NeuralNet::NeuralNet(std::uint32_t seed)
    :Model(),
     _lyrInput(),
     _lyrHidden1(784,256),
     _lyrHidden2(256,128),
     _lyrOutput(128,10)
{
    std::mt19937    random( seed );
    _lyrInput.init( random );
    _lyrHidden1.init( random );
    _lyrHidden2.init( random );
    _lyrOutput.init( random );
}
NeuralNet::~NeuralNet()
{

}
void NeuralNet::save(const char* szFName)
{
    Model::save( szFName );
}
void NeuralNet::load(const char* szFName)
{
    Model::load( szFName );
}
std::shared_ptr<Tensor> NeuralNet::forward(
    std::vector<std::shared_ptr<Tensor>>& inputs
)
{
    if( (inputs.size()!=1)||(inputs[0]==nullptr) )
    {
        throw std::runtime_error(
            "NeuralNet::forward: exactly one non-null input is required"
        );
    }
    if( (inputs[0]->_mData._nRows!=784)||(inputs[0]->_mData._nCols<=0) )
    {
        throw std::runtime_error(
            "NeuralNet::forward: input must have shape 784 x batch"
        );
    }
    //
    std::vector<std::shared_ptr<Tensor>>    v{inputs[0]};
    std::shared_ptr<Tensor> x   =_lyrInput.forward( v );
    v   ={x};
    x   =_lyrHidden1.forward( v );
    v   ={x};
    x   =_lyrHidden2.forward( v );
    v   ={x};
    //
    return _lyrOutput.forward( v );
}
std::shared_ptr<Tensor> NeuralNet::loss(
    const std::shared_ptr<Tensor>& input,
    const std::shared_ptr<Tensor>& target
)
{
    if( (input==nullptr)||(target==nullptr) )
    {
        throw std::runtime_error(
            "NeuralNet::loss: input and target must be non-null"
        );
    }
    if( (target->_mData._nRows!=10)||
        (target->_mData._nCols!=input->_mData._nCols) )
    {
        throw std::runtime_error(
            "NeuralNet::loss: target must have shape 10 x batch"
        );
    }
    //
    std::vector<std::shared_ptr<Tensor>>    inputs{input};
    std::shared_ptr<Tensor>                 logits  =forward( inputs );
    std::vector<std::shared_ptr<Tensor>>    args{ logits,target };

    return _entropy( args );
}
std::vector<Tensor*> NeuralNet::getParams()
{
    std::vector<Tensor*> rsts;
    std::vector<Tensor*> tmp;

    tmp =_lyrInput.getParams();
    for( int i=0;i<(int)tmp.size();i++ )
    {
        rsts.push_back( tmp[i] );
    }

    tmp =_lyrHidden1.getParams();
    for( int i=0;i<(int)tmp.size();i++ )
    {
        rsts.push_back( tmp[i] );
    }
    
    tmp =_lyrHidden2.getParams();
    for( int i=0;i<(int)tmp.size();i++ )
    {
        rsts.push_back( tmp[i] );
    }
    
    tmp =_lyrOutput.getParams();
    for( int i=0;i<(int)tmp.size();i++ )
    {
        rsts.push_back( tmp[i] );
    }

    return rsts;
}
