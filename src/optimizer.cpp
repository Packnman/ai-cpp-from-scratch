#include <iostream>
#include <math.h>

#include "graph.h"
#include "tensor.h"
#include "optimizer.h"


// --------------------------
// OptimizerParams
// --------------------------
// Optimizerごとに必要な
// パラメータ固有の状態を保持する
// --------------------------
OptimizerParams::OptimizerParams()
{

}
OptimizerParams::~OptimizerParams()
{

}

// --------------------------
// Optimizer
// --------------------------
// 役割：
//  ・Modelから学習パラメータを取得する
//  ・各Tensorの勾配を使ってパラメータを更新する
//  ・Optimizer固有の状態を管理する
// --------------------------
Optimizer::Optimizer(Model* lpModel,float fLearningRate)
{
    if( lpModel==nullptr )
    {
        throw std::runtime_error(
            "Optimizer: model is null"
        );
    }
    _lpModel        =lpModel;
    _fLearningRate  =fLearningRate;
    _nStep          =0;
}
Optimizer::~Optimizer()
{

}
void Optimizer::init()
{
    _nStep      =0;
    _lpParams   =_lpModel->getParams();
    _spOptimizerParams.clear();

    for( int i=0;i<static_cast<int>(_lpParams.size());i++ )
    {
        std::shared_ptr<OptimizerParams> spParams   =createOptimizerParams( _lpParams[i] );
        _spOptimizerParams.push_back( spParams );
    }
}
void Optimizer::update()
{
    _nStep++;
    for( int i=0;i<static_cast<int>(_lpParams.size());i++ )
    {
        update_param( _lpParams[i],_spOptimizerParams[i].get() );
    }
}
void Optimizer::zero_grads()
{
    _lpModel->zero_grads();
}
std::shared_ptr<OptimizerParams> Optimizer::createOptimizerParams(Tensor* lpTensor)
{
    (void)lpTensor;     // 警告除去

    throw std::runtime_error(
        "Optimizer::createOptimizerParams is not implemented"
    );
}
void Optimizer::update_param(Tensor* lpTensor,OptimizerParams* lpOptimizerParams)
{
    (void)lpTensor;             // 警告除去
    (void)lpOptimizerParams;    // 警告除去

    throw std::runtime_error(
        "Optimizer::update_param is not implemented"
    );
}


// --------------------------
// SGDParams
// --------------------------
SGDParams::SGDParams()
{

}
SGDParams::~SGDParams()
{

}

// --------------------------
// SGD
// --------------------------
SGD::SGD(Model* lpModel,float fLearningRate)
    :Optimizer( lpModel,fLearningRate )
{

}
SGD::~SGD()
{

}
std::shared_ptr<OptimizerParams>
SGD::createOptimizerParams(Tensor* lpTensor)
{
    SGDParams* lpParamsSGD  =new SGDParams(
        // Non
    );
    std::shared_ptr<SGDParams> spParams =std::shared_ptr<SGDParams>(lpParamsSGD);

    return spParams;
}
void SGD::update_param(Tensor* lpTensor,OptimizerParams* lpOptimizerParams)
{
    // W = W - LearingRate * grad
    cuda_axpy(
        lpTensor->_mData,
        -_fLearningRate,
        lpTensor->_mGrad
    );
}


// --------------------------
// AdamParams
// --------------------------
AdamParams::AdamParams(int nRows,int nCols)
{
    _mM =cuMat( nRows,nCols );  cuda_fill( _mM,0.0f );
    _mV =cuMat( nRows,nCols );  cuda_fill( _mV,0.0f );
}
AdamParams::~AdamParams()
{

}

// --------------------------
// Adam
// --------------------------
Adam::Adam(Model* lpModel,float fLearningRate)
    :Optimizer(lpModel,fLearningRate)
{

}
Adam::~Adam()
{

}
std::shared_ptr<OptimizerParams>
Adam::createOptimizerParams(Tensor* lpTensor)
{
    AdamParams* lpParamsAdam    =new AdamParams(
        lpTensor->_mData._nRows,
        lpTensor->_mData._nCols
    );
    std::shared_ptr<AdamParams> spParams    =std::shared_ptr<AdamParams>(lpParamsAdam);

    return spParams;
}
void Adam::update_param(Tensor* lpTensor,OptimizerParams* lpOptimizerParams)
{
    // 更新手順はREADME参照
    AdamParams* lpAdamParams    =dynamic_cast<AdamParams*>(lpOptimizerParams);

    if( lpAdamParams==nullptr )
    {
        throw std::runtime_error(
            "Adam::update_param: invalid optimizer params"
        );
    }

    const float fBeta1      =0.9f;
    const float fBeta2      =0.999f;
    const float fEpsilon    =1.0e-8f;

    float fBeta1Correction  =1.0f - powf(fBeta1,static_cast<float>(_nStep));
    float fBeta2Correction  =1.0f - powf(fBeta2,static_cast<float>(_nStep));

    cuda_Adam_update(
        lpTensor->_mData,
        lpTensor->_mGrad,
        lpAdamParams->_mM,
        lpAdamParams->_mV,
        _fLearningRate,
        fBeta1,
        fBeta2,
        fBeta1Correction,
        fBeta2Correction,
        fEpsilon
    );
}