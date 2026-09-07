#pragma once

#include "cuda_function.h"


// --------------------------
// GELU
// --------------------------
class GELU: public Function
{
public:
    GELU();
    ~GELU();

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
