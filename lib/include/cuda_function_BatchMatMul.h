#pragma once

#include "cuda_function.h"


// --------------------------
// BatchMatMul
// --------------------------
class BatchMatMul: public Function
{
public:
    BatchMatMul();
    ~BatchMatMul();

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
