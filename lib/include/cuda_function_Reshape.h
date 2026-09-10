#pragma once

#include "cuda_function.h"


// --------------------------
// Reshape
// --------------------------
class Reshape: public Function
{
public:
    Reshape();
    explicit Reshape(std::vector<std::int64_t> nShape);
    ~Reshape();

private:
    std::vector<std::int64_t> _nShape;
    static std::vector<std::int64_t> resolveShape(
        const std::vector<std::int64_t>& c_nConfigured,
        const cufMat& c_mInput
    );

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
