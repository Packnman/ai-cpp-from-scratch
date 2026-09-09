#include "cuda_function_Reshape.h"

#include <stdexcept>

Reshape::Reshape()
    :Function()
{
    // TODO: Add a target-shape argument, validate inferred dimensions, and
    // retain the shape required by forward while backward uses the input shape.
}

Reshape::~Reshape()
{
}

void Reshape::backward(
    const std::vector<const cufMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    // TODO: Reshape the output gradient back to the original input shape and
    // accumulate it without changing element order.
    (void)c_lpmOutputGrads;
    (void)c_spmInputs;
    (void)c_spmOutputs;
    throw std::logic_error("Reshape::backward is not implemented");
}

std::vector<std::shared_ptr<Tensor>> Reshape::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // TODO: Validate one contiguous input and an element-count-preserving
    // target shape, then create the reshaped output or view.
    (void)c_spmInputs;
    throw std::logic_error("Reshape::forward is not implemented");
}
