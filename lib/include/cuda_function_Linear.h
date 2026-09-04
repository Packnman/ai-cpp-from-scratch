#pragma once 

#include "cuda_function.h"


// --------------------------
// Linear
// --------------------------
class Linear: public Function
{
public:
    Linear(Tensor* lpWeight,Tensor* lpBias);
    ~Linear();

public:
    Tensor* _lpmWeight;
    Tensor* _lpmBias;
    cuMat   _mTmp;
public:
    void backward(
        const std::vector<const cuMat*>& c_lpmOutputGrads,
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
        const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
    ) override;
    std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
    ) override;
};
