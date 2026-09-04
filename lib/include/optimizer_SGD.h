#pragma once

#include "optimizer.h"


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
