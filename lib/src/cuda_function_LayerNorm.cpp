#include "cuda_function_LayerNorm.h"

#include <stdexcept>

LayerNorm::LayerNorm()
    :Function()
{
    // TODO: Add configuration for the normalized axes, epsilon, and optional
    // learnable scale and bias tensors before implementing this operation.
}

LayerNorm::~LayerNorm()
{
}

void LayerNorm::backward(
    const std::vector<const cufMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    // TODO: Use the saved normalization statistics to compute the input
    // gradient and, when enabled, accumulate scale and bias gradients.
    (void)c_lpmOutputGrads;
    (void)c_spmInputs;
    (void)c_spmOutputs;
    throw std::logic_error("LayerNorm::backward is not implemented");
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // TODO: Validate one input, calculate stable mean and variance over the
    // configured axes, normalize it, and apply optional scale and bias.
    (void)c_spmInputs;
    throw std::logic_error("LayerNorm::forward is not implemented");
}
