#include <iostream>
#include <math.h>

#include "module.h"
#include "cuda_tensor.h"
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
    :_lpModel( lpModel ),
     _fLearningRate( fLearningRate ),
     _nStep( 0 )
{
    if( lpModel==nullptr )
    {
        throw std::runtime_error(
            "Optimizer: model is null"
        );
    }
}
Optimizer::~Optimizer()
{

}
void Optimizer::init()
{
    _nStep      =0;
    _lpParams   =_lpModel->getParams();
    _spOptimizerParams.clear();

    for( auto lpParam : _lpParams )
    {
        auto spParams =createOptimizerParams( lpParam );
        _spOptimizerParams.push_back( spParams );
    }
}
void Optimizer::update()
{
    _nStep++;
    // for( std::size_t i=0;i<_lpParams.size();i++ )    // i++は処理内でコピーが発生するため遅いことがある。
    for( std::size_t i=0;i<_lpParams.size();++i )       // ++iは自信を増加させるためコピーが発生しないことから推奨される
    {
        update_param(
            _lpParams[i],
            _spOptimizerParams[i].get()
        );
    }
}
void Optimizer::zero_grads()
{
    _lpModel->zero_grads();
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
    (void)lpTensor;
    //
    return std::make_shared<SGDParams>();
}
void SGD::update_param(Tensor* lpTensor,OptimizerParams* lpOptimizerParams)
{
    (void)lpOptimizerParams;
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
    :_mM( nRows,nCols ),
     _mV( nRows,nCols )
{
    cuda_fill( _mM,0.0f );
    cuda_fill( _mV,0.0f );
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
    return std::make_shared<AdamParams>(
        lpTensor->_mData._nRows,
        lpTensor->_mData._nCols
    );
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

    const float c_fBeta1      =0.9f;
    const float c_fBeta2      =0.999f;
    const float c_fEpsilon    =1.0e-8f;

    float fBeta1Correction =1.0f -powf(
        c_fBeta1,
        static_cast<float>(_nStep)
    );
    float fBeta2Correction =1.0f -powf(
        c_fBeta2,
        static_cast<float>(_nStep)
    );

    cuda_Adam_update(
        lpTensor->_mData,
        lpTensor->_mGrad,
        lpAdamParams->_mM,
        lpAdamParams->_mV,
        _fLearningRate,
        c_fBeta1,
        c_fBeta2,
        fBeta1Correction,
        fBeta2Correction,
        c_fEpsilon
    );
}
