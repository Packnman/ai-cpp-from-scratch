#pragma once

#include "cuda_function.h"


// --------------------------
// Permute
// --------------------------
class Permute: public Function
{
public:
    Permute();
    explicit Permute(std::vector<std::size_t> nDimensions);
    ~Permute();

private:
    std::vector<std::size_t> _nDimensions;

    static std::vector<std::size_t> _dimensionsFor(
        const std::vector<std::size_t>& c_nConfigured,
        std::size_t c_nRank
    );
    static std::vector<std::size_t> _inverseOf(
        const std::vector<std::size_t>& c_nDimensions
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
