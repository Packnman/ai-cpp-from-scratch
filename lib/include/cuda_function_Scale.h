#pragma once

#include "cuda_function.h"


// --------------------------
// Scale
// --------------------------
class Scale: public Function
{
public:
    explicit Scale(float fScale =1.0f);
    ~Scale();

private:
    float _fScale;

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
