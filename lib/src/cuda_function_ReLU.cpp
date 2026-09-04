#include "cuda_tensor.h"
#include "cuda_function_ReLU.h"


// --------------------------
// ReLU
// --------------------------
ReLU::ReLU()
{
    // nothing
}
ReLU::~ReLU()
{
    // nothing
}
void ReLU::backward(
    const std::vector<const cuMat*>& c_lpmOutputGrads,
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs,
    const std::vector<std::shared_ptr<Tensor>>& c_spmOutputs
)
{
    (void)c_spmOutputs;
    // 逆伝播
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error(
            "ReLU::backward: ReLU requires exactly one input"
        );
    }
    //
    cuda_ReLU_backward(
        c_spmInputs[0]->_mGrad,
        c_spmInputs[0]->_mData,
        requireSingleOutputGrad(c_lpmOutputGrads,"ReLU::backward")
    );
}

std::vector<std::shared_ptr<Tensor>>
ReLU::forward(
    const std::vector<std::shared_ptr<Tensor>>& c_spmInputs
)
{
    // ReLU(x) =max(0,x)
    if( c_spmInputs.size()!=1 )
    {
        throw std::runtime_error(
            "ReLU::forward: ReLU requires exactly one input"
        );
    }
    //
    auto spmResult =std::make_shared<Tensor>(
        c_spmInputs[0]->_mData._nRows,
        c_spmInputs[0]->_mData._nCols
    );
    // 順伝播
    cuda_ReLU_forward(
        spmResult->_mData,
        c_spmInputs[0]->_mData
    );

    return {spmResult};
}

