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
    float _fLearningRate =0.0f;
    std::uint64_t _nStep =0;
    std::vector<ParameterState> _prmParameters;
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

// --------------------------
// SGDParams
// --------------------------
class SGDParams : public OptimizerParams{
public:
    SGDParams();
    ~SGDParams();
};

// --------------------------
// SGD
// --------------------------
class SGD : public Optimizer{
public:
    SGD(Model* lpModel,float fLearningRate);
    ~SGD();
protected:
    std::shared_ptr<OptimizerParams> createOptimizerParams(Tensor* lpTensor) override;
    void update_param(Tensor* lpTensor,OptimizerParams* lpOptimizerParams) override;
};


// --------------------------
// Adamのパラメータごとの状態を名前付きで参照する。
// --------------------------
struct NamedAdamState
{
    std::string _strName;
    cuMat* _lpmFirstMoment =nullptr;
    cuMat* _lpmSecondMoment =nullptr;
};

using AdamState =OptimizerState<NamedAdamState>;
// --------------------------
// AdamParams
// --------------------------
class AdamParams : public OptimizerParams{
public:
    AdamParams(int nRows,int nCols);
    ~AdamParams();
public:
    cuMat   _mM;    // 勾配の移動平均
    cuMat   _mV;    // 勾配二乗の移動平均
};

// --------------------------
// Adam
// --------------------------
class Adam : public Optimizer{
public:
    Adam(Model* lpModel,float fLearningRate);
    ~Adam();
protected:
    std::shared_ptr<OptimizerParams> createOptimizerParams(Tensor* lpTensor) override;
    void update_param(Tensor* lpTensor,OptimizerParams* lpOptimizerParams) override;
};
