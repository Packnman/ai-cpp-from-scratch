#include "cuda_function_Permute.h"

#include <stdexcept>

Permute::Permute()
    :Function()
{
    // TODO: Add and validate an axis-permutation argument, then retain both
    // the permutation and its inverse for forward and backward.
}

Permute::~Permute()
{
}

void Permute::backward(
    const std::vector<const cufMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    // TODO: Apply the inverse axis permutation to the output gradient and
    // accumulate the resulting tensor into the input gradient.
    (void)c_lpmOutputGrads;
    (void)c_spmInputs;
    (void)c_spmOutputs;
    throw std::logic_error("Permute::backward is not implemented");
}

std::vector<std::shared_ptr<Tensor>> Permute::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // TODO: Validate one input and the configured axis order, then return a
    // correctly permuted tensor (materializing contiguous storage if needed).
    (void)c_spmInputs;
    throw std::logic_error("Permute::forward is not implemented");
}
