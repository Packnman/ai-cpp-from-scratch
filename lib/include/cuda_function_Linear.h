#pragma once 

#include "cuda_function.h"


// --------------------------
// Linear
// --------------------------
// Applies W X + b on the first axis; trailing axes are preserved (rank >= 2).
class Linear: public Function
{
public:
    Linear(Tensor* lpWeight,Tensor* lpBias);
    ~Linear();

public:
    Tensor* _lpmWeight;
    Tensor* _lpmBias;
    cufMat   _mTmp;
public:
    void backward(
        const std::vector<const cufMat*>& c_lpmOutputGrads,
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
        const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
    ) override;
    std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
    ) override;
};
