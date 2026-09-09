#include "cuda_function_BatchMatMul.h"

#include <stdexcept>

BatchMatMul::BatchMatMul()
    :Function()
{
}

BatchMatMul::~BatchMatMul()
{
}

void BatchMatMul::backward(
    const std::vector<const cufMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    // TODO: Validate the saved operands and output gradient, compute
    // dA=dY*B^T and dB=A^T*dY for every batch, and accumulate both gradients.
    (void)c_lpmOutputGrads;
    (void)c_spmInputs;
    (void)c_spmOutputs;
    throw std::logic_error("BatchMatMul::backward is not implemented");
}

std::vector<std::shared_ptr<Tensor>> BatchMatMul::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // TODO: Accept two batch-compatible tensors, validate their inner matrix
    // dimensions, allocate the batched result, and multiply them on CUDA.
    (void)c_spmInputs;
    throw std::logic_error("BatchMatMul::forward is not implemented");
}
