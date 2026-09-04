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
