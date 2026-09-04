#pragma once

#include "cuda_function.h"


// --------------------------
// Dropout
// --------------------------
class Dropout: public Function
{
public:
    Dropout(float fDropProbability,std::uint64_t nSeed);
    ~Dropout() override;

private:
    float _fDropProbability;
    curandGenerator_t _crnGenerator;
    cuMat _mMask;

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