#pragma once

#include "cuda_function.h"

// --------------------------
// Embedding
// --------------------------
class Embedding: public IndexFunction
{
public:
    explicit Embedding(Tensor* lpWeight);
    ~Embedding() override =default;

    Tensor* _lpmWeight;

    void backward(
        const TensorGradList& c_lpmOutputGrads,
        const std::shared_ptr<const cunMat>& c_spmIndices,
        const TensorList& c_spmOutputs
    ) override;
    TensorList forward(
        const std::shared_ptr<const cunMat>& c_spmIndices
    ) override;
};
