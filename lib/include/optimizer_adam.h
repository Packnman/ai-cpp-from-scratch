#pragma once

#include "optimizer.h"

// --------------------------
// Adamのパラメータごとの状態を名前付きで参照する。
// --------------------------
struct NamedAdamState
{
    std::string strName;
    cufMat* lpmFirstMoment =nullptr;
    cufMat* lpmSecondMoment =nullptr;
};


using AdamState =OptimizerState<NamedAdamState>;
// --------------------------
// AdamParams
// --------------------------
class AdamParams : public OptimizerParams{
public:
    AdamParams(int nRows,int nCols);
    explicit AdamParams(const std::vector<std::int64_t>& shape);
    ~AdamParams();
public:
    cufMat   _mM;    // 勾配の移動平均
    cufMat   _mV;    // 勾配二乗の移動平均
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
