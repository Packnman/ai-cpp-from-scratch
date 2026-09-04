#pragma once

#include "cuda_function.h"

// --------------------------
// SoftmaxCrossEntropy
// --------------------------
class SoftmaxCrossEntropy: public Function
{
public:
    SoftmaxCrossEntropy();
    ~SoftmaxCrossEntropy();

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
