#include "cuda_function_Mask.h"

#include <stdexcept>

Mask::Mask(Tensor* lpMask)
    :Function()
{
    // TODO: Validate and retain the mask tensor, including a clearly defined
    // mask convention and broadcasting rules.
    (void)lpMask;
}

Mask::~Mask()
{
}

void Mask::backward(
    const std::vector<const cufMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    // TODO: Propagate the output gradient only through unmasked positions and
    // accumulate it into the input; the mask itself is not differentiable.
    (void)c_lpmOutputGrads;
    (void)c_spmInputs;
    (void)c_spmOutputs;
    throw std::logic_error("Mask::backward is not implemented");
}

std::vector<std::shared_ptr<Tensor>> Mask::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // TODO: Validate one input against the stored mask, broadcast the mask as
    // specified, and replace or suppress masked values for attention scoring.
    (void)c_spmInputs;
    throw std::logic_error("Mask::forward is not implemented");
}
