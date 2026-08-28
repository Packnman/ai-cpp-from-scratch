#pragma once
#include <vector>
#include <memory>
#include "cuda_matrix.h"
class Model;
class Tensor;

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
    int     _nStep;         // update回数
    
    std::vector<Tensor*>    _lpParams;
    std::vector<std::shared_ptr<OptimizerParams>>   _spOptimizerParams;

public:
    virtual void init();
    virtual void update();
    virtual void zero_grads();

protected:
    virtual std::shared_ptr<OptimizerParams> createOptimizerParams(Tensor* lpTensor);
    virtual void update_param(Tensor* lpTensor,OptimizerParams* lpOptimizerParams);
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
// AdamParams
// --------------------------
class AdamParams : public OptimizerParams{
public:
    AdamParams(int nRows,int nCols);
    ~AdamParams();
public:
    cuMat   _mM;
    cuMat   _mV;
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