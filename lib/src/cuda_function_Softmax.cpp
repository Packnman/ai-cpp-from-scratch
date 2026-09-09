#include "cuda_function_Softmax.h"

#include <stdexcept>

Softmax::Softmax()
    :Function()
{
    // TODO: Add an axis configuration argument and validate it against each
    // input rank before launching the operation.
}

Softmax::~Softmax()
{
}

void Softmax::backward(
    const std::vector<const cufMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    // TODO: Use the saved softmax output to compute the Jacobian-vector
    // product y*(dY-sum(dY*y)) along the configured axis.
    (void)c_lpmOutputGrads;
    (void)c_spmInputs;
    (void)c_spmOutputs;
    throw std::logic_error("Softmax::backward is not implemented");
}

std::vector<std::shared_ptr<Tensor>> Softmax::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // TODO: Validate one input and compute a numerically stable softmax along
    // the configured axis by subtracting the axis maximum before exponentiation.
    (void)c_spmInputs;
    throw std::logic_error("Softmax::forward is not implemented");
}
