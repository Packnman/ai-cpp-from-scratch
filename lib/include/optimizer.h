#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cuda_matrix.h"

class Model;
class Tensor;

template<class ParameterState>
struct OptimizerState
{
    float fLearningRate =0.0f;
    std::uint64_t nStep =0;
    std::vector<ParameterState> prmParameters;
};

// --------------------------
// OptimizerParams
// --------------------------
// Optimizerごとに必要な
// パラメータ固有の状態を保持する
// --------------------------
class OptimizerParams{
public:
    OptimizerParams();
    virtual ~OptimizerParams();
};

// --------------------------
// Optimizer
// --------------------------
// 役割：
//  ・Modelから学習パラメータを取得する
//  ・各Tensorの勾配を使ってパラメータを更新する
//  ・Optimizer固有の状態を管理する
// --------------------------
class Optimizer{
public:
    Optimizer(Model* lpModel,float fLearningRate);
    virtual ~Optimizer();
private:
protected:
    Model*  _lpModel;
    float   _fLearningRate;
    std::uint64_t _nStep;   // update回数
    
    std::vector<Tensor*>    _lpParams;
    std::vector<std::shared_ptr<OptimizerParams>>   _spOptimizerParams;

public:
    virtual void init();
    virtual void update();
    virtual void zero_grads();

protected:
    virtual std::shared_ptr<OptimizerParams> createOptimizerParams(Tensor* lpTensor) =0;
    virtual void update_param(Tensor* lpTensor,OptimizerParams* lpOptimizerParams) =0;
};


