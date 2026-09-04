#include "cuda_tensor.h"
#include "cuda_function_GELU.h"


// --------------------------
// GELU
// --------------------------
GELU::GELU()
{

}
GELU::~GELU()
{

}
void GELU::backward(
    const std::vector<const cuMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    (void)c_spmOutputs;
    // 内容はREADMEを参照すること
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error(
            "GELU::backward: GELU requires exactly one input"
        );
    }
    //
    cuda_GELU_backward(
        c_spmInputs[0]->_mGrad,
        c_spmInputs[0]->_mData,
        requireSingleOutputGrad(c_lpmOutputGrads,"GELU::backward")
    );
}
std::vector<std::shared_ptr<Tensor>>
GELU::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // 内容はREADMEを参照すること
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error(
            "GELU::forward: GELU requires exactly one input"
        );
    }

    auto spmResult =std::make_shared<Tensor>(
        c_spmInputs[0]->_mData._nRows,
        c_spmInputs[0]->_mData._nCols
    );
    //
    cuda_GELU_forward(
        spmResult->_mData,
        c_spmInputs[0]->_mData
    );

    return {spmResult};
}
