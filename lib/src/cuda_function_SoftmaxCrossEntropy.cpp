#include "cuda_tensor.h"
#include "cuda_function_SoftmaxCrossEntropy.h"


// --------------------------
// SoftmaxCrossEntropy
// --------------------------
SoftmaxCrossEntropy::SoftmaxCrossEntropy()
{

}
SoftmaxCrossEntropy::~SoftmaxCrossEntropy()
{

}
void SoftmaxCrossEntropy::backward(
    const std::vector<const cuMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    (void)c_spmOutputs;
    // c_spmInputs[0] = logits
    // c_spmInputs[1] = target(one-hot)
    //
    // L = SoftmaxCrossEntropy(logits, target)
    //
    // dL/dlogits
    //     = (softmax(logits) - target) / batch_size
    //
    // logits.grad += grad * dL/dlogits
    // 
    if( c_spmInputs.size()!=2 )
    {
        throw std::runtime_error(
            "SoftmaxCrossEntropy::backward: "
            "SoftmaxCrossEntropy requires exactly two inputs"
        );
    }
    //
    cuda_SoftmaxCrossEntropy_backward(
        c_spmInputs[0]->_mGrad,
        c_spmInputs[0]->_mData,
        c_spmInputs[1]->_mData,
        requireSingleOutputGrad(c_lpmOutputGrads,"SoftmaxCrossEntropy::backward")
    );
}
std::vector<std::shared_ptr<Tensor>>
SoftmaxCrossEntropy::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // c_spmInputs[0] = logits
    // c_spmInputs[1] = target(one-hot)
    if( c_spmInputs.size()!=2 )
    {
        throw std::runtime_error(
            "SoftmaxCrossEntropy::forward: "
            "SoftmaxCrossEntropy requires exactly two inputs"
        );
    }
    if( (c_spmInputs[0]->_mData._nRows!=c_spmInputs[1]->_mData._nRows)||
        (c_spmInputs[0]->_mData._nCols!=c_spmInputs[1]->_mData._nCols) )
    {
        throw std::runtime_error(
            "SoftmaxCrossEntropy::forward: "
            "logits and target size mismatch"
        );
    }
    //
    auto spmResult =std::make_shared<Tensor>( 1,1 );

    cuda_SoftmaxCrossEntropy_forward(
        spmResult->_mData,
        c_spmInputs[0]->_mData,
        c_spmInputs[1]->_mData
    );

    return {spmResult};
}
