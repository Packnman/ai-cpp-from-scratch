#include "cuda_function_Scale.h"

#include <stdexcept>

Scale::Scale()
    :Function()
{
    // TODO: Add a scalar configuration argument and retain it for forward and
    // backward (attention normally uses this for 1/sqrt(headDimension)).
}

Scale::~Scale()
{
}

void Scale::backward(
    const std::vector<const cufMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    // TODO: Multiply the output gradient by the configured scalar and
    // accumulate it into the single input gradient.
    (void)c_lpmOutputGrads;
    (void)c_spmInputs;
    (void)c_spmOutputs;
    throw std::logic_error("Scale::backward is not implemented");
}

std::vector<std::shared_ptr<Tensor>> Scale::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // TODO: Validate one input, allocate a same-shaped output, and multiply
    // every element by the configured scalar on CUDA.
    (void)c_spmInputs;
    throw std::logic_error("Scale::forward is not implemented");
}
